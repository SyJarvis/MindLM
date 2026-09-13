"""Render an SFT messages-CSV into packed token bins for grouped SFT training."""
import argparse
import csv
import json
import struct

from transformers import AutoTokenizer

REPO = "/home/runke.zhong.srv/workspace/MindLM"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=f"{REPO}/data/sft_qwen3_combined.csv")
    parser.add_argument("--tokenizer", default=f"{REPO}/qwen3_tokenizer")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--bin", default=f"{REPO}/data/sft_tokens_4096.bin")
    parser.add_argument("--meta", default=f"{REPO}/data/sft_tokens_4096.rows.jsonl")
    parser.add_argument("--stats", default=f"{REPO}/data/sft_tokens_4096.json")
    parser.add_argument("--progress_every", type=int, default=500000)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    stats = {"total": 0, "errors": 0, "over_long": 0, "tokens": 0, "answer_tokens": 0}
    lengths = []

    bos_ids = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids

    def find_last_sublist(hay, needle):
        n = len(needle)
        for i in range(len(hay) - n, -1, -1):
            if hay[i:i + n] == needle:
                return i
        return -1

    out = open(args.bin, "wb")
    meta = open(args.meta, "w", encoding="utf-8")
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            if stats["total"] % args.progress_every == 0:
                print(f"... {stats['total']} rows, over_long={stats['over_long']}", flush=True)
            try:
                messages = json.loads(row["messages"])
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
                ids = tok(text, add_special_tokens=False).input_ids
            except Exception:
                stats["errors"] += 1
                continue

            marker = find_last_sublist(ids, bos_ids)
            answer_start = marker + len(bos_ids)
            if marker < 0 or answer_start >= len(ids):
                stats["errors"] += 1
                continue

            keep = min(len(ids), args.max_tokens)
            if len(ids) > args.max_tokens:
                stats["over_long"] += 1

            out.write(struct.pack(f"<{keep}I", *ids[:keep]))
            meta.write(json.dumps({
                "off": stats["tokens"],
                "n": keep,
                "ans": keep - answer_start,
                "src": row.get("source", ""),
            }) + "\n")
            stats["tokens"] += keep
            stats["answer_tokens"] += keep - answer_start
            lengths.append(keep)
    out.close()
    meta.close()

    lengths.sort()
    n = len(lengths)
    stats.update({
        "kept": n,
        "mean_len": round(sum(lengths) / n, 1),
        "p50": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
        "le_256": sum(l <= 256 for l in lengths),
        "le_512": sum(256 < l <= 512 for l in lengths),
        "le_1024": sum(512 < l <= 1024 for l in lengths),
        "le_2048": sum(1024 < l <= 2048 for l in lengths),
        "le_4096": sum(2048 < l <= 4096 for l in lengths),
    })
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
