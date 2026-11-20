#!/usr/bin/env python3
"""Stage D: sample the SFT subset from sft_e2_combined_v5.csv per the frozen
mix contract in docs/project/2026-11-20/stageD-sft-data-contract-draft.md §3.5.

Per-source sampling (deterministic, seeded):
- t2t_qa, legacy (zh QA):        Bernoulli keep --zh_qa_keep_frac
- perfectblend_en:               target = cap_fraction x available rows
- ultra, multilingual_zh, pb_zh: keep all
- xlam_fc, toolace (tools):      keep all (under the mindlm template these
                                  train as tools-free conversations with the
                                  tool-call XML bodies preserved)

Two passes over the CSV: pass 1 counts rows per source; the keep decisions are
made on line numbers; pass 2 streams the file again and writes the chosen
rows in stable line order. Memory holds only the chosen line-number set.
The sampled CSV feeds data_process/sft.py --template_style mindlm.
"""
import argparse
import csv
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="sft_e2_combined_v5.csv path")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--stats", required=True, help="output sidecar JSON path")
    parser.add_argument("--zh_qa_keep_frac", type=float, default=0.07,
                        help="fraction of t2t_qa+legacy rows to keep (default 0.07)")
    parser.add_argument("--perfectblend_en_cap", type=float, default=0.10,
                        help="perfectblend_en target as a fraction of its own rows")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--progress_every", type=int, default=2000000)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.zh_qa_keep_frac <= 1:
        raise SystemExit("--zh_qa_keep_frac must be in (0, 1]")
    if not 0 < args.perfectblend_en_cap <= 1:
        raise SystemExit("--perfectblend_en_cap must be in (0, 1]")
    rng = random.Random(args.seed)

    # Pass 1: count rows per source (line numbers are 1-based, header is not a row).
    counts = {}
    line_no = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            line_no += 1
            src = row.get("source", "") or "?"
            counts[src] = counts.get(src, 0) + 1
            if line_no % args.progress_every == 0:
                print(f"... pass1 scanned {line_no}", flush=True)
    allowed = set(counts)
    unknown = allowed - {"t2t_qa", "legacy", "perfectblend_en", "ultra",
                         "multilingual_zh", "pb_zh", "xlam_fc", "toolace"}
    if unknown:
        raise SystemExit(f"unmapped sources {sorted(unknown)}; update the mix contract first")

    # Decide keep counts.
    targets = {}
    for src in ("t2t_qa", "legacy"):
        targets[src] = max(1, round(counts.get(src, 0) * args.zh_qa_keep_frac))
    targets["perfectblend_en"] = max(1, round(counts.get("perfectblend_en", 0) * args.perfectblend_en_cap))
    for src in ("ultra", "multilingual_zh", "pb_zh", "xlam_fc", "toolace"):
        targets[src] = counts.get(src, 0)

    # Choose line numbers per source by sampling from the observed count.
    # Because pass 2 re-reads in the same order, the k-th row of a source
    # always sits at the same line number.
    picks = {}
    for src, target in targets.items():
        available = counts.get(src, 0)
        picks[src] = set(rng.sample(range(1, available + 1), target)) if available else set()

    # Pass 2: stream and write chosen rows.
    fieldnames = None
    kept_counts = {src: 0 for src in targets}
    written = 0
    with open(args.csv, newline="", encoding="utf-8") as fin, \
            open(args.out, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        src_seen = {src: 0 for src in targets}
        line_no = 0
        for row in reader:
            line_no += 1
            src = row.get("source", "") or "?"
            if src not in targets:
                continue
            src_seen[src] += 1
            if src_seen[src] in picks[src]:
                writer.writerow(row)
                kept_counts[src] += 1
                written += 1
            if line_no % args.progress_every == 0:
                print(f"... pass2 scanned {line_no}, written {written}", flush=True)

    stats = {
        "input_rows": line_no,
        "output_rows": written,
        "input_counts": counts,
        "realized": kept_counts,
        "params": {
            "zh_qa_keep_frac": args.zh_qa_keep_frac,
            "perfectblend_en_cap": args.perfectblend_en_cap,
            "seed": args.seed,
        },
    }
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
