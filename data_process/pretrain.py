#!/usr/bin/env python3
"""Deduplicate documents, split stable groups, and pack independent datasets."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time
from typing import NamedTuple
import unicodedata

import numpy as np


FORMAT = 'mindlm_packed_pretrain_v1'
MANIFEST_FORMAT = 'mindlm_pretrain_v1'
CHAT_EOS_TOKEN = '<|im_end|>'
_WORKER_TOKENIZER = None


class Document(NamedTuple):
    text: str
    source: str
    input_row: int
    full_hash: bytes
    group_hash: bytes
    heldout: bool
    order_hash: bytes


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def tokenizer_fingerprint(path):
    path = Path(path).resolve()
    files = {}
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            raise ValueError(f'Tokenizer symlinks are not supported: {item}')
        if item.is_file():
            files[item.relative_to(path).as_posix()] = {
                'size_bytes': item.stat().st_size,
                'sha256': sha256_file(item),
            }
    if not files:
        raise ValueError(f'Empty tokenizer directory: {path}')
    serialized = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'path': str(path), 'files': files, 'sha256': hashlib.sha256(serialized).hexdigest()}


def document_keys(text, seed, prefix_chars, heldout_ppm):
    normalized = unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n')).strip()
    full_hash = hashlib.sha256(normalized.encode('utf-8')).digest()
    prefix = ' '.join(normalized.split())[:prefix_chars]
    group_hash = hashlib.sha256(prefix.encode('utf-8')).digest()
    seed_bytes = str(seed).encode('ascii')
    split_hash = hashlib.sha256(seed_bytes + b'/heldout/' + group_hash).digest()
    heldout = int.from_bytes(split_hash[:8], 'big') < (heldout_ppm * 2**64 // 1_000_000)
    order_hash = hashlib.sha256(seed_bytes + b'/mix/' + full_hash).digest()
    return full_hash, group_hash, heldout, order_hash


def scan_documents(args):
    documents, seen, groups = [], set(), {}
    counts = defaultdict(Counter)
    source_bits = {name: 1 << i for i, name in enumerate(sorted(args.sources))}
    rows = blank_rows = 0
    csv.field_size_limit(100_000_000)
    with args.input_csv.open('r', encoding='utf-8', errors='strict', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if not header or len(header) != len(set(header)) or not {'text', 'source'} <= set(header):
            raise ValueError('CSV requires distinct text and source columns')
        text_index, source_index = header.index('text'), header.index('source')
        for record in reader:
            if args.limit_documents and rows >= args.limit_documents:
                break
            rows += 1
            if not record:
                blank_rows += 1
                continue
            if len(record) != len(header):
                raise ValueError(f'Wrong column count at logical data row {rows}')
            source = record[source_index]
            if source not in source_bits:
                raise ValueError(f'Unexpected source {source!r} at logical data row {rows}')
            counts[source]['rows_read'] += 1
            counts[source]['characters_read'] += len(record[text_index])
            text = record[text_index].strip()
            if not text:
                counts[source]['empty_rows'] += 1
                continue
            full_hash, group_hash, heldout, order_hash = document_keys(
                text, args.seed, args.prefix_chars, args.heldout_ppm,
            )
            group = groups.setdefault(group_hash, [0, 0, 0])
            group[0] += 1
            group[2] |= source_bits[source]
            if full_hash in seen:
                counts[source]['exact_duplicates_dropped'] += 1
                continue
            seen.add(full_hash)
            group[1] += 1
            counts[source]['documents_kept'] += 1
            counts[source]['characters_kept'] += len(text)
            documents.append(Document(text, source, rows, full_hash, group_hash, heldout, order_hash))
            if rows % args.progress_every == 0:
                print(f'scan rows={rows:,} kept={len(documents):,}', flush=True)
    if args.expected_rows is not None and rows != args.expected_rows:
        raise ValueError(f'Expected {args.expected_rows} rows, found {rows}')
    if not documents:
        raise ValueError('Input has no nonempty unique documents')
    heldout_hashes = {doc.full_hash for doc in documents if doc.heldout}
    heldout_groups = {doc.group_hash for doc in documents if doc.heldout}
    exact_overlap = {doc.full_hash for doc in documents if not doc.heldout and doc.full_hash in heldout_hashes}
    group_overlap = {doc.group_hash for doc in documents if not doc.heldout and doc.group_hash in heldout_groups}
    assert not exact_overlap and not group_overlap, 'Train/heldout document or group overlap'
    audit = {
        'logical_rows_read': rows,
        'blank_csv_rows': blank_rows,
        'retained_documents': len(documents),
        'exact_duplicates_dropped': sum(value['exact_duplicates_dropped'] for value in counts.values()),
        'groups': len(groups),
        'groups_with_multiple_retained_documents': sum(value[1] > 1 for value in groups.values()),
        'largest_retained_group_documents': max(value[1] for value in groups.values()),
        'cross_source_groups_including_dropped_duplicates': sum(bin(value[2]).count('1') > 1 for value in groups.values()),
        'train_groups': len(groups) - len(heldout_groups),
        'heldout_groups': len(heldout_groups),
        'train_heldout_exact_intersection': len(exact_overlap),
        'train_heldout_group_intersection': len(group_overlap),
        'sources': {
            source: {key: counts[source][key] for key in (
                'rows_read', 'empty_rows', 'exact_duplicates_dropped',
                'documents_kept', 'characters_read', 'characters_kept',
            )}
            for source in args.sources
        },
    }
    documents.sort(key=lambda doc: (doc.order_hash, doc.full_hash))
    return documents, audit


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def chat_eos_id(tokenizer):
    """Return the explicit Qwen3 chat EOS token used as document boundary."""

    convert = getattr(tokenizer, 'convert_tokens_to_ids', None)
    if not callable(convert):
        raise ValueError('tokenizer must expose convert_tokens_to_ids')
    vocabulary = tokenizer.get_vocab()
    if CHAT_EOS_TOKEN not in vocabulary:
        raise ValueError(f'tokenizer does not contain {CHAT_EOS_TOKEN}')
    token_id = convert(CHAT_EOS_TOKEN)
    if token_id is None or isinstance(token_id, (list, tuple)):
        raise ValueError(f'tokenizer does not contain {CHAT_EOS_TOKEN}')
    token_id = int(token_id)
    if not 0 <= token_id < len(tokenizer):
        raise ValueError(f'invalid {CHAT_EOS_TOKEN} id: {token_id}')
    return token_id


def init_worker(tokenizer_path):
    global _WORKER_TOKENIZER
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    _WORKER_TOKENIZER = load_tokenizer(tokenizer_path)


def encode_worker(texts):
    return _WORKER_TOKENIZER(texts, add_special_tokens=False)['input_ids']


def document_batches(documents, max_rows, max_chars):
    batch, characters = [], 0
    for doc in documents:
        if batch and (len(batch) >= max_rows or characters + len(doc.text) > max_chars):
            yield batch
            batch, characters = [], 0
        batch.append(doc)
        characters += len(doc.text)
    if batch:
        yield batch


def encoded_batches(documents, args, tokenizer):
    batches = iter(document_batches(documents, args.batch_rows, args.batch_chars))
    if args.workers == 0:
        for batch in batches:
            yield batch, tokenizer([doc.text for doc in batch], add_special_tokens=False)['input_ids']
        return
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=init_worker,
        initargs=(str(args.tokenizer_path),),
    ) as pool:
        pending = deque()
        for batch in batches:
            pending.append((batch, pool.submit(encode_worker, [doc.text for doc in batch])))
            if len(pending) >= args.workers * 2:
                ready, future = pending.popleft()
                yield ready, future.result()
        while pending:
            ready, future = pending.popleft()
            yield ready, future.result()


class PackedWriter:
    def __init__(self, directory, split, length, vocab_size, boundary_token_id):
        self.directory, self.split = directory, split
        self.width = length + 1
        self.vocab_size = vocab_size
        self.boundary_token_id = boundary_token_id
        self.path = directory / f'{split}.bin.tmp'
        self.stream = self.path.open('wb')
        self.buffer, self.buffer_sources = [], []
        self.sources = defaultdict(Counter)
        self.documents = self.source_tokens = self.boundary_tokens_added = self.records = 0

    def append(self, doc, token_ids):
        if not token_ids:
            raise ValueError(f'Nonempty document produced no tokens: input row {doc.input_row}')
        if min(token_ids) < 0 or max(token_ids) >= self.vocab_size:
            raise ValueError(f'Token id outside vocabulary: input row {doc.input_row}')
        start = self.source_tokens + self.boundary_tokens_added
        add_boundary = token_ids[-1] != self.boundary_token_id
        stats = self.sources[doc.source]
        stats['documents'] += 1
        stats['source_tokens'] += len(token_ids)
        stats['boundary_tokens_in_source'] += token_ids.count(self.boundary_token_id)
        stats['boundary_appended'] += int(add_boundary)
        stats['stream_tokens'] += len(token_ids) + int(add_boundary)
        self.documents += 1
        self.source_tokens += len(token_ids)
        self.boundary_tokens_added += int(add_boundary)
        self.buffer.extend(token_ids)
        self.buffer_sources.extend([doc.source] * len(token_ids))
        if add_boundary:
            self.buffer.append(self.boundary_token_id)
            self.buffer_sources.append(doc.source)
        complete = len(self.buffer) // self.width * self.width
        if complete:
            np.asarray(self.buffer[:complete], dtype='<u4').tofile(self.stream)
            self.records += complete // self.width
            self.buffer = self.buffer[complete:]
            self.buffer_sources = self.buffer_sources[complete:]
        return {
            'stream_offset': start,
            'source_tokens': len(token_ids),
            'boundary_appended': bool(add_boundary),
        }

    def finish(self, input_path, input_sha):
        self.stream.close()
        if not self.records:
            raise ValueError(f'{self.split} contains no full packed sequence')
        tail_by_source = Counter(self.buffer_sources)
        for source, stats in self.sources.items():
            stats['discarded_tail_tokens'] = tail_by_source[source]
            stats['packed_tokens'] = stats['stream_tokens'] - tail_by_source[source]
        packed_tokens = self.records * self.width
        assert packed_tokens + len(self.buffer) == self.source_tokens + self.boundary_tokens_added
        assert sum(value['packed_tokens'] for value in self.sources.values()) == packed_tokens
        assert self.path.stat().st_size == packed_tokens * 4
        tokens = np.memmap(self.path, dtype='<u4', mode='r')
        minimum, maximum = self.vocab_size, -1
        for start in range(0, len(tokens), 1_000_000):
            block = tokens[start:start + 1_000_000]
            minimum, maximum = min(minimum, int(block.min())), max(maximum, int(block.max()))
        del tokens
        assert 0 <= minimum <= maximum < self.vocab_size
        return {
            'format': FORMAT, 'dtype': '<u4', 'loss_mask_dtype': 'uint8',
            'sequence_length': self.width - 1,
            'tokens_per_record': self.width, 'num_sequences': self.records,
            'tokenizer_vocab_size': self.vocab_size,
            'boundary_token': CHAT_EOS_TOKEN,
            'boundary_token_id': self.boundary_token_id,
            'documents': self.documents, 'source_tokens': self.source_tokens,
            'boundary_appended': self.boundary_tokens_added,
            'stream_tokens': self.source_tokens + self.boundary_tokens_added,
            'packed_tokens': packed_tokens, 'supervised_tokens': self.records * (self.width - 1),
            'discarded_tail_tokens': len(self.buffer), 'discarded_tail_by_source': dict(tail_by_source),
            'source_statistics': dict(self.sources),
            'add_boundary': True,
            'min_token_id': minimum, 'max_token_id': maximum, 'sha256': sha256_file(self.path),
            'size_bytes': self.path.stat().st_size, 'source_csv': str(input_path), 'source_sha256': input_sha,
        }


def prepare(args):
    started = time.monotonic()
    args.input_csv = Path(args.input_csv).resolve()
    args.tokenizer_path = Path(args.tokenizer_path).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    writers = []
    try:
        input_stat = args.input_csv.stat()
        input_sha = sha256_file(args.input_csv)
        tokenizer_info = tokenizer_fingerprint(args.tokenizer_path)
        print(f'input sha256={input_sha} bytes={input_stat.st_size}', flush=True)
        scan_started = time.monotonic()
        documents, audit = scan_documents(args)
        scan_seconds = time.monotonic() - scan_started
        print(f'scan audit={json.dumps(audit)}', flush=True)
        tokenizer = load_tokenizer(str(args.tokenizer_path))
        vocab_size, boundary_token_id = len(tokenizer), chat_eos_id(tokenizer)
        if not 0 < vocab_size <= np.iinfo(np.uint32).max or not 0 <= boundary_token_id < vocab_size:
            raise ValueError('Tokenizer requires uint32-compatible vocabulary and a valid Qwen3 chat EOS token')
        writers = [PackedWriter(args.output_dir, split, args.max_seq_len, vocab_size, boundary_token_id) for split in ('train', 'heldout')]
        index_tmp = args.output_dir / 'documents.jsonl.tmp'
        processed = 0
        tokenization_started = time.monotonic()
        with index_tmp.open('w', encoding='utf-8') as index:
            for batch, encoded in encoded_batches(documents, args, tokenizer):
                if len(batch) != len(encoded):
                    raise ValueError('Tokenizer returned a different number of documents')
                for doc, ids in zip(batch, encoded):
                    writer = writers[int(doc.heldout)]
                    position = writer.append(doc, ids)
                    index.write(json.dumps({
                        'input_row': doc.input_row, 'source': doc.source, 'split': writer.split,
                        'text_sha256': doc.full_hash.hex(), 'group_sha256': doc.group_hash.hex(), **position,
                    }, separators=(',', ':')) + '\n')
                processed += len(batch)
                if processed % args.progress_every < len(batch):
                    print(f'tokenize documents={processed:,}/{len(documents):,} source_tokens={sum(w.source_tokens for w in writers):,} seconds={time.monotonic() - started:.1f}', flush=True)
        tokenization_seconds = time.monotonic() - tokenization_started
        after = args.input_csv.stat()
        if (after.st_size, after.st_mtime_ns) != (input_stat.st_size, input_stat.st_mtime_ns):
            raise RuntimeError('Input CSV changed during preparation')
        split_metadata = {writer.split: writer.finish(args.input_csv, input_sha) for writer in writers}
        manifest = {
            'format': MANIFEST_FORMAT, 'status': 'complete',
            'input': {'path': str(args.input_csv), 'size_bytes': input_stat.st_size, 'mtime_ns': input_stat.st_mtime_ns, 'sha256': input_sha},
            'tokenizer': tokenizer_info,
            'policy': {
                'seed': args.seed, 'heldout_ppm': args.heldout_ppm, 'prefix_chars': args.prefix_chars,
                'exact_deduplication': 'SHA256(NFC, CRLF/CR to LF, outer strip); keep first input occurrence',
                'grouping': 'SHA256(first prefix_chars of whitespace-collapsed normalized text); group only, never prefix-drop',
                'split': 'uint64_bigendian(SHA256(ascii(seed)+/heldout/+group_digest)[:8]) < floor(heldout_ppm*2**64/1000000)',
                'mixing': 'sort by SHA256(ascii(seed)+/mix/+full_digest), then full_digest',
                'tokenized_text': 'original text with outer strip; internal whitespace and code indentation preserved',
                'packing': 'independent split streams; append Qwen3 chat EOS unless last token is chat EOS; nonoverlapping sequence_length+1 records; drop final incomplete record',
                'limitation': 'Exact/group overlap audited; arbitrary semantic or non-prefix near duplicates are not guaranteed removed',
            },
            'audit': audit, 'splits': split_metadata,
            'document_index': {'file': 'documents.jsonl', 'sha256': sha256_file(index_tmp), 'records': processed},
            'execution': {'workers': args.workers, 'max_inflight_batches': args.workers * 2 if args.workers else 1, 'batch_rows': args.batch_rows, 'batch_chars': args.batch_chars, 'limit_documents': args.limit_documents, 'scan_sort_seconds': scan_seconds, 'tokenization_seconds': tokenization_seconds, 'seconds': time.monotonic() - started},
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'transformers': importlib.metadata.version('transformers'), 'tokenizers': importlib.metadata.version('tokenizers')},
            'builder_sha256': sha256_file(__file__),
        }
        for split, metadata in split_metadata.items():
            os.replace(args.output_dir / f'{split}.bin.tmp', args.output_dir / f'{split}.bin')
            atomic_json(args.output_dir / f'{split}.json', metadata)
        os.replace(index_tmp, args.output_dir / 'documents.jsonl')
        atomic_json(args.output_dir / 'manifest.json', manifest)
        print(f'COMPLETE {args.output_dir} seconds={time.monotonic() - started:.1f}', flush=True)
        return manifest
    except BaseException as error:
        for writer in writers:
            writer.stream.close()
        atomic_json(args.output_dir / 'failed.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--tokenizer-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sources', nargs='+', default=['minimind', 'open-perfectblend'])
    parser.add_argument('--max-seq-len', type=int, default=4096)
    parser.add_argument('--heldout-ppm', type=int, default=5000)
    parser.add_argument('--prefix-chars', type=int, default=150)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-rows', type=int, default=256)
    parser.add_argument('--batch-chars', type=int, default=500_000)
    parser.add_argument('--progress-every', type=int, default=100_000)
    parser.add_argument('--limit-documents', type=int, default=0)
    parser.add_argument('--expected-rows', type=int)
    args = parser.parse_args(argv)
    if not 0 < args.heldout_ppm < 1_000_000:
        parser.error('heldout-ppm must be between 1 and 999999')
    if min(args.max_seq_len, args.prefix_chars, args.batch_rows, args.batch_chars, args.progress_every) < 1 or args.workers < 0 or args.limit_documents < 0:
        parser.error('lengths/batch/progress must be positive; workers/limit must be nonnegative')
    if len(args.sources) != len(set(args.sources)) or not all(args.sources):
        parser.error('sources must be distinct nonempty labels')
    return args


if __name__ == '__main__':
    prepare(parse_args())
