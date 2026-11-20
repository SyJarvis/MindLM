#!/usr/bin/env python3
"""Stage B step 3: streaming JSONL tokenizing/packing loader (mindlm contract v2-stream).

Deterministic online replacement for the offline PackedWriter path:
  - reads corpus JSONL in order (rows accepted in file order),
  - splits train/heldout by SHA256(seed + '/heldout/' + group_hash) ppm rule
    (group hash = first prefix_chars of whitespace-normalized text, NFC),
  - mixes by SHA256(seed + '/mix/' + full_hash) sort key per fixed-size window
    (streaming: memory-bounded reservoir of window rows) instead of global sort,
  - tokenizes incrementally, appends boundary token (tokenizer EOS by default,
    configurable), packs 4097-token records (input 4096 + target shift),
  - exposes per-epoch deterministic shuffling of packed records via index map.

The packer is the same fixed-width/no-padding contract as mindlm_packed_pretrain_v1
but the boundary token is taken from the live tokenizer instead of the hard-coded
Qwen3 chat EOS, and writes nothing to disk: records are yielded in memory chunks.
"""
import hashlib
import json
import unicodedata
from collections import Counter, deque

WINDOW_ROWS = 50_000          # deterministic mixing window (rows sorted by order_hash inside)
CHUNK_RECORDS = 512           # records per yielded chunk (512 * 4097 * 4B ≈ 8 MiB)


def normalize_text(text):
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def document_keys(text, seed, prefix_chars, heldout_ppm):
    normalized = normalize_text(text)
    full_hash = hashlib.sha256(normalized.encode("utf-8")).digest()
    prefix = " ".join(normalized.split())[:prefix_chars]
    group_hash = hashlib.sha256(prefix.encode("utf-8")).digest()
    seed_bytes = str(seed).encode("ascii")
    split_hash = hashlib.sha256(seed_bytes + b"/heldout/" + group_hash).digest()
    heldout = int.from_bytes(split_hash[:8], "big") < (heldout_ppm * 2**64 // 1_000_000)
    order_hash = hashlib.sha256(seed_bytes + b"/mix/" + full_hash).digest()
    return full_hash, group_hash, heldout, order_hash


class StreamPacker:
    """Fixed-width packer, in-memory twin of the offline PackedWriter."""

    def __init__(self, seq_len, vocab_size, boundary_token_id):
        self.seq_len = seq_len
        self.width = seq_len + 1
        self.vocab_size = vocab_size
        self.boundary_token_id = boundary_token_id
        self.buffer = []
        self.records = 0
        self.stats = Counter()

    def append(self, token_ids):
        if min(token_ids) < 0 or max(token_ids) >= self.vocab_size:
            raise ValueError("token id outside vocabulary")
        self.stats["source_tokens"] += len(token_ids)
        self.buffer.extend(token_ids)
        if token_ids and token_ids[-1] != self.boundary_token_id:
            self.buffer.append(self.boundary_token_id)
            self.stats["boundary_appended"] += 1
        self.stats["documents"] += 1

    def take_records(self):
        """Pop complete records from the buffer; keep the tail for the next append."""
        out = []
        complete = len(self.buffer) // self.width * self.width
        if complete:
            flat = self.buffer[:complete]
            self.records += complete // self.width
            self.buffer = self.buffer[complete:]
            # emit as (width,) records: row-major reshape
            for i in range(0, complete, self.width):
                out.append(flat[i:i + self.width])
        return out

    @property
    def tail_tokens(self):
        return len(self.buffer)


class StreamingPretrainLoader:
    """One-pass streaming JSONL -> packed records, deterministic, memory-bounded."""

    def __init__(self, corpus_path, tokenizer, seq_len=4096, seed=1337,
                 prefix_chars=30, heldout_ppm=5000, boundary_token_id=None,
                 window_rows=WINDOW_ROWS, sources=None):
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.prefix_chars = prefix_chars
        self.heldout_ppm = heldout_ppm
        self.vocab_size = tokenizer.vocab_size
        if boundary_token_id is None:
            boundary_token_id = tokenizer.eos_token_id
            if boundary_token_id is None:
                raise ValueError("tokenizer has no eos_token_id; pass boundary_token_id explicitly")
        self.boundary_token_id = int(boundary_token_id)
        self.window_rows = window_rows
        self.sources = sources

    def _scan_row(self, row_bytes):
        obj = json.loads(row_bytes)
        text, source = obj["text"], obj["source"]
        if self.sources is not None and source not in self.sources:
            return None
        return text, source

    def stream(self, split="train", max_records=None):
        """Yield (record_index, tokens list of width) for the requested split.

        split='train' or 'heldout'. One pass over the corpus produces both
        splits' stats via scan_stats(); this generator filters rows by the
        deterministic heldout rule and mixes inside each window.
        """
        packer = StreamPacker(self.seq_len, self.vocab_size, self.boundary_token_id)
        window = []            # rows in current window: (order_hash, full_hash, text, source)
        emitted = 0
        with open(self.corpus_path, "rb", buffering=1024 * 1024) as fh:
            for raw in fh:
                try:
                    text, source = self._scan_row(raw)
                except Exception:
                    continue
                if text is None or not text:
                    continue
                full_hash, group_hash, heldout, order_hash = document_keys(
                    text, self.seed, self.prefix_chars, self.heldout_ppm)
                if (split == "heldout") != bool(heldout):
                    continue
                window.append((order_hash, full_hash, text, source))
                if len(window) >= self.window_rows:
                    for (_o, _f, t, s) in sorted(window, key=lambda r: (r[0], r[1])):
                        ids = self.tokenizer(t, add_special_tokens=False)["input_ids"]
                        if not ids:
                            continue
                        packer.append(ids)
                        for rec in packer.take_records():
                            emitted += 1
                            yield rec
                            if max_records is not None and emitted >= max_records:
                                return
                    window = []
        for (_o, _f, t, s) in sorted(window, key=lambda r: (r[0], r[1])):
            ids = self.tokenizer(t, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            packer.append(ids)
            for rec in packer.take_records():
                emitted += 1
                yield rec
                if max_records is not None and emitted >= max_records:
                    return


def determinstic_record_shuffle(num_records, rng_seed):
    """Deterministic permutation of packed record indices (numpy-free)."""
    import random
    rng = random.Random(rng_seed)
    order = list(range(num_records))
    rng.shuffle(order)
    return order
