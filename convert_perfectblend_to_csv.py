"""Flatten mlabonne/open-perfectblend parquet conversations into a `text`-column CSV.

Each multi-turn conversation is rendered with the Qwen3 chat-template markers
(`<|im_start|>role\\n...<|im_end|>\\n`) so the pretraining run sees properly
delimited dialogue. The result is consumed by prepare_data.py --type pretrain,
which tokenizes with add_special_tokens=False and inserts EOS between documents.
"""

import argparse
import csv
import glob
import os
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parent

# open-perfectblend uses "human"/"gpt"; map to the Qwen3 chat roles.
ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert open-perfectblend parquet to a text CSV")
    parser.add_argument("--parquet_glob", required=True, help="Glob matching the parquet shards")
    parser.add_argument("--output_csv", required=True, help="Output CSV with a single `text` column")
    parser.add_argument("--tokenizer_path", default=str(REPOSITORY_ROOT / "qwen3_tokenizer"))
    parser.add_argument("--max_turns", type=int, default=0, help="0 = keep all turns")
    parser.add_argument("--min_chars", type=int, default=1, help="Skip rows whose rendered text is shorter")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def render_text(conversation, max_turns):
    parts = []
    turns = conversation if max_turns <= 0 else conversation[:max_turns]
    for message in turns:
        role = ROLE_MAP.get(message.get("from", ""), "user")
        value = str(message.get("value", "")).strip()
        if not value:
            continue
        parts.append(f"<|im_start|>{role}\n{value}<|im_end|>\n")
    return "".join(parts)


def main():
    args = parse_args()
    output = Path(args.output_csv)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer only to sanity-check that the chat markers are real tokens,
    # so the rendered text is not split into ordinary subwords.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    for marker in ("<|im_start|>", "<|im_end|>"):
        ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            print(f"warning: {marker!r} tokenizes to {len(ids)} pieces: {ids}")

    files = sorted(glob.glob(args.parquet_glob))
    if not files:
        raise FileNotFoundError(f"no parquet files matched {args.parquet_glob}")

    total_rows = 0
    skipped = 0
    with output.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["text"])
        for shard in files:
            table = pq.read_table(shard, columns=["conversations"])
            conversations = table.column("conversations").to_pylist()
            for conv in conversations:
                text = render_text(conv, args.max_turns)
                if len(text) < args.min_chars:
                    skipped += 1
                    continue
                writer.writerow([text])
                total_rows += 1
            print(f"  {os.path.basename(shard)}: cumulative rows={total_rows} skipped={skipped}")

    print(f"Wrote {total_rows} rows ({skipped} skipped) to {output}")


if __name__ == "__main__":
    main()
