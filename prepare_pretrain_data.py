"""Pack CSV text into fixed-length, EOS-delimited pretraining token blocks."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parent
PACKED_DATASET_FORMAT = "mindlm_packed_pretrain_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Pack a text CSV for MindLM pretraining")
    parser.add_argument("--input_csv", required=True, help="CSV containing a text column")
    parser.add_argument("--output_prefix", required=True, help="Writes <prefix>.bin and <prefix>.json")
    parser.add_argument("--tokenizer_path", default=str(REPOSITORY_ROOT / "qwen3_tokenizer"))
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_rows", type=int, default=1024)
    parser.add_argument("--no_add_eos", action="store_true", help="Do not append EOS between source documents")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2")
    if args.batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    return args


def batched_rows(reader, text_column, batch_rows):
    batch = []
    for row in reader:
        value = row.get(text_column)
        if value is None:
            raise ValueError(f"CSV is missing required column '{text_column}'")
        text = str(value).strip()
        if text:
            batch.append(text)
        if len(batch) == batch_rows:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    args = parse_args()
    prefix = Path(args.output_prefix)
    tokens_path = prefix.with_suffix(".bin")
    metadata_path = prefix.with_suffix(".json")
    for path in (tokens_path, metadata_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    if not args.no_add_eos and eos_token_id is None:
        raise ValueError("The tokenizer has no eos_token_id; pass --no_add_eos only if intentional")
    if len(tokenizer) > np.iinfo(np.uint32).max:
        raise ValueError("Tokenizer vocabulary exceeds uint32 storage")

    tokens_per_record = args.max_seq_len + 1
    prefix.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = tokens_path.with_suffix(".bin.tmp")
    sequence_count = 0
    source_documents = 0
    source_tokens = 0
    buffer = []
    buffer_start = 0

    try:
        with Path(args.input_csv).open("r", encoding="utf-8", errors="replace", newline="") as input_file, temporary_path.open("wb") as output_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None or args.text_column not in reader.fieldnames:
                raise ValueError(f"CSV is missing required column '{args.text_column}'")

            for texts in batched_rows(reader, args.text_column, args.batch_rows):
                encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
                for token_ids in encoded:
                    if not token_ids:
                        continue
                    source_documents += 1
                    source_tokens += len(token_ids)
                    buffer.extend(token_ids)
                    if not args.no_add_eos and token_ids[-1] != eos_token_id:
                        buffer.append(eos_token_id)

                    while len(buffer) - buffer_start >= tokens_per_record:
                        record = np.asarray(
                            buffer[buffer_start : buffer_start + tokens_per_record],
                            dtype=np.uint32,
                        )
                        record.tofile(output_file)
                        sequence_count += 1
                        buffer_start += tokens_per_record
                    # Keep memory bounded when an individual document is very long.
                    if buffer_start >= 1_000_000:
                        buffer = buffer[buffer_start:]
                        buffer_start = 0
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    if sequence_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("The input did not contain one full packed sequence")

    os.replace(temporary_path, tokens_path)
    metadata = {
        "format": PACKED_DATASET_FORMAT,
        "dtype": "uint32",
        "sequence_length": args.max_seq_len,
        "tokens_per_record": tokens_per_record,
        "num_sequences": sequence_count,
        "tokenizer_vocab_size": len(tokenizer),
        "eos_token_id": eos_token_id,
        "documents": source_documents,
        "source_tokens": source_tokens,
        "discarded_tail_tokens": len(buffer) - buffer_start,
        "add_eos": not args.no_add_eos,
        "source_csv": str(Path(args.input_csv).resolve()),
    }
    with metadata_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=False, indent=2)

    usable_tokens = sequence_count * tokens_per_record
    print(f"Packed {source_documents} documents into {sequence_count} sequences")
    print(f"Usable tokens: {usable_tokens}; discarded tail: {metadata['discarded_tail_tokens']}")
    print(f"Token file: {tokens_path}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
