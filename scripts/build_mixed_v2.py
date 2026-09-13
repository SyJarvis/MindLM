#!/usr/bin/env python3
"""Build the V2 mixed pretraining pack (mindlm_packed_pretrain_v1).

Pipeline (docs/pretraining_v2_plan.md §3-4, docs/handoff_v2_worklog.md §5):
  1. pretrain_data.csv         (Chinese web): prefix-150-MD5 dedup -> uniform sample to quota
  2. pretrain_t2t.jsonl        (Chinese t2t): full-MD5 dedup -> classify -> weighted keep -> sample to quota
  3. perfectblend_mathcode.csv (English math/code, ChatML already stripped): full pass
  -> shuffle -> tokenize (qwen3, add_special_tokens=False) -> pack into 4096+1 uint32
  blocks with EOS 151645 between documents -> <prefix>.bin + <prefix>.json

The manifest passes PackedPretrainDataset validation (dataset.py): format,
sequence_length, tokens_per_record, tokenizer_vocab_size, dtype, and
num_sequences * tokens_per_record * 4 must equal the .bin size in bytes.

Dependencies: tokenizers + numpy only (no torch/transformers), so it runs both
on the local base env and on the packaging server.

Usage:
  # sanity pass: first 1000 docs per source, small bin, full manifest
  /Users/mac/base/bin/python3 scripts/build_mixed_v2.py --dry-run \
      --output-prefix data/packed_mixed_v2_dry

  # full build (write to a disk with >=20GB free; ~15GB expected)
  /Users/mac/base/bin/python3 scripts/build_mixed_v2.py \
      --output-prefix /mnt/pack/packed_mixed_v2 --target-tokens-per-source 1.04e9 0.8e9 0.43e9
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
csv.field_size_limit(10**9)

PACKED_FORMAT = "mindlm_packed_pretrain_v1"
EOS_TOKEN_ID = 151645  # qwen3 tokenizer_config.json: eos_token <|im_end|> = 151645

# Same first-hit rules as scripts/profile_pretrain_data.py (T2T_RULES).
T2T_RULES = [
    ("code", ["```", "def ", "代码", "编程", "Python", "python", "函数", "算法"]),
    ("math", ["计算", "数学", "方程", "求解", "积分", "概率", "几何", "等于", "表达式"]),
    ("multi_turn", ["好的。现在", "好的，现在", "接下来请", "现在我会", "下一", "第二", "再次"]),
    ("writing", ["写作", "撰写", "写一篇", "写一封", "文章", "作文", "邮件", "摘要", "总结", "故事", "小说", "标题"]),
]
# Plan §3.3: keep all code/math/multi_turn, writing 50%, other 70%.
T2T_KEEP_PROB = {"code": 1.0, "math": 1.0, "multi_turn": 1.0, "writing": 0.5, "other": 0.7}


def parse_args():
    p = argparse.ArgumentParser(description="Build the V2 mixed pretraining pack")
    p.add_argument("--output-prefix", type=Path, default=BASE / "data" / "packed_mixed_v2")
    p.add_argument("--tokenizer-path", type=Path, default=BASE / "qwen3_tokenizer")
    p.add_argument("--seed", type=int, default=42)
    # Per-source token budgets (estimated tokens; the manifest reports the real
    # per-source token counts after tokenization). Defaults = handoff option A
    # (~2.2B). Option B (total ~3.5B): 0.95e9 1.2e9 0.43e9 -- t2t oversampled.
    p.add_argument("--target-tokens-per-source", type=float, nargs=3,
                   metavar=("CN_GENERAL", "T2T", "PB_MATHCODE"),
                   default=[1.04e9, 0.8e9, 0.43e9])
    p.add_argument("--min-doc-tokens", type=int, default=20,
                   help="Drop documents estimated under this many tokens.")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--batch-rows", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true",
                   help="Only the first 1000 docs per source; quotas ignored.")
    p.add_argument("--limit-per-source", type=int, default=None,
                   help="Hard cap on documents read per source (debug).")
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.max_seq_len < 2:
        p.error("--max-seq-len must be at least 2")
    if args.batch_rows < 1:
        p.error("--batch-rows must be positive")
    return args


# ---------------------------------------------------------------- readers

def iter_csv_texts(path):
    with open(path, "r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"CSV missing 'text' column: {path}")
        for row in reader:
            yield str(row.get("text") or "")


def iter_jsonl_texts(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield str(obj.get("text", ""))


def classify_t2t(text):
    for label, patterns in T2T_RULES:
        if any(p in text for p in patterns):
            return label
    return "other"


def md5_head(text, n=150):
    return hashlib.md5(text[:n].encode("utf-8", "ignore")).digest()[:8]


def md5_full(text):
    return hashlib.md5(text.encode("utf-8", "ignore")).digest()[:8]


def estimate_tokens(text):
    """Cheap token estimate for quota math only (real counts at tokenize time).

    Calibrated on 2000-doc samples per source with the qwen3 tokenizer:
    Chinese ~0.73 tok/char, Latin/code ~0.26 tok/char (measured est/real bias
    1.38x / 1.12x before calibration).
    """
    head = text[:512]
    cjk = sum(1 for ch in head if "\u4e00" <= ch <= "\u9fff")
    ratio = cjk / max(len(head), 1)
    return len(text) * (ratio * 0.73 + (1 - ratio) * 0.26)


# ---------------------------------------------------------------- loaders

class SourceStats:
    def __init__(self, name, path):
        self.name = name
        self.path = str(path)
        self.docs_read = 0
        self.dup_dropped = 0
        self.short_dropped = 0
        self.weight_dropped = 0
        self.docs_kept = 0
        self.chars_kept = 0
        self.est_tokens_kept = 0.0
        self.category_kept = {}

    def summary(self):
        return {
            "source": self.name,
            "input": self.path,
            "docs_read": self.docs_read,
            "dup_dropped": self.dup_dropped,
            "short_dropped": self.short_dropped,
            "weight_dropped": self.weight_dropped,
            "docs_kept": self.docs_kept,
            "chars_kept": self.chars_kept,
            "est_tokens_kept": round(self.est_tokens_kept),
            "category_kept": self.category_kept,
        }


def load_source(name, path, reader, dedup, quota_tokens, min_chars, limit, rng, dry,
                classify=None, keep_prob=None):
    """Shared loader: dedup -> (classify -> weighted keep) -> min-length -> quota reservoir.

    The reservoir keeps a uniform sample of accepted docs with an estimated
    total of ~quota_tokens: append until the budget fills, then apply standard
    reservoir replacement so late documents still get in with fair probability.
    """
    st = SourceStats(name, path)
    seen = set()
    reservoir = []
    n_accepted = 0  # docs that passed every filter (denominator for replacement)

    for raw in reader(path):
        if limit and st.docs_read >= limit:
            break
        if dry and st.docs_read >= 1000:
            break
        st.docs_read += 1
        text = raw.strip()
        if not text:
            continue
        h = md5_head(text) if dedup == "prefix" else md5_full(text)
        if h in seen:
            st.dup_dropped += 1
            continue
        seen.add(h)
        if classify is not None:
            cat = classify(text)
            if rng.random() > keep_prob[cat]:
                st.weight_dropped += 1
                continue
            st.category_kept[cat] = st.category_kept.get(cat, 0) + 1
        if len(text) < min_chars:
            st.short_dropped += 1
            continue

        n_accepted += 1
        st.docs_kept += 1
        st.chars_kept += len(text)
        st.est_tokens_kept += estimate_tokens(text)
        if st.est_tokens_kept <= quota_tokens:
            reservoir.append(text)
        else:
            j = rng.randrange(n_accepted)
            if j < len(reservoir):
                reservoir[j] = text
    del seen
    print(f"  [{name}] read={st.docs_read:,} dup={st.dup_dropped:,} short={st.short_dropped:,} "
          f"weight={st.weight_dropped:,} kept={st.docs_kept:,} -> sampled={len(reservoir):,}", flush=True)
    st.docs_kept = len(reservoir)
    return reservoir, st


# ---------------------------------------------------------------- packing

def pack(corpus, tokenizer_path, tokens_path, max_seq_len, batch_rows):
    """corpus: list of (text, source_name). Writes tokens_path (.bin.tmp), returns stats."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
    vocab_size = tok.get_vocab_size()
    tokens_per_record = max_seq_len + 1

    tokens_path = Path(tokens_path)
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tokens_path.with_suffix(".bin.tmp")

    sequence_count = 0
    documents = 0
    source_tokens = 0
    per_source = {}
    buffer = []
    buffer_start = 0
    t0 = time.time()

    def flush_records(out_file):
        nonlocal sequence_count, buffer, buffer_start
        while len(buffer) - buffer_start >= tokens_per_record:
            np.asarray(buffer[buffer_start:buffer_start + tokens_per_record],
                       dtype=np.uint32).tofile(out_file)
            sequence_count += 1
            buffer_start += tokens_per_record
        if buffer_start >= 1_000_000:  # keep memory bounded on long documents
            buffer = buffer[buffer_start:]
            buffer_start = 0

    with tmp_path.open("wb") as out_file:
        for i in range(0, len(corpus), batch_rows):
            batch = corpus[i:i + batch_rows]
            encodings = tok.encode_batch([t for t, _ in batch], add_special_tokens=False)
            for (text, src), enc in zip(batch, encodings):
                ids = enc.ids
                if not ids:
                    continue
                documents += 1
                source_tokens += len(ids)
                per_source[src] = per_source.get(src, 0) + len(ids)
                buffer.extend(ids)
                if ids[-1] != EOS_TOKEN_ID:
                    buffer.append(EOS_TOKEN_ID)
                flush_records(out_file)
            if (i // batch_rows) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  packed {min(i + batch_rows, len(corpus)):,}/{len(corpus):,} docs, "
                      f"{sequence_count:,} seqs, {elapsed:.0f}s", flush=True)

    stats = {
        "vocab_size": vocab_size,
        "documents": documents,
        "source_tokens": source_tokens,
        "source_token_breakdown": per_source,
        "discarded_tail_tokens": len(buffer) - buffer_start,
        "pack_seconds": round(time.time() - t0, 1),
    }
    return sequence_count, stats


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    cn_path = BASE / "data" / "pretrain_data.csv"
    t2t_path = BASE / "data" / "pretrain_t2t.jsonl"
    pb_path = BASE / "data" / "perfectblend_mathcode.csv"
    missing = [str(p) for p in (cn_path, t2t_path, pb_path) if not p.is_file()]
    if missing:
        sys.exit("Missing input files: " + ", ".join(missing))

    # min length gate: 20 tokens ~ 20 Chinese chars (worst case, ~1 tok/char)
    min_chars = args.min_doc_tokens
    q = args.target_tokens_per_source
    print(f"targets: cn_general={q[0]/1e9:.2f}B t2t={q[1]/1e9:.2f}B pb_mathcode={q[2]/1e9:.2f}B "
          f"(total ~{(q[0]+q[1]+q[2])/1e9:.2f}B est)", flush=True)

    t0 = time.time()
    cn_docs, cn_st = load_source("pretrain_data", cn_path, iter_csv_texts, dedup="prefix",
                                 quota_tokens=q[0], min_chars=min_chars,
                                 limit=args.limit_per_source, rng=rng, dry=args.dry_run)
    t2t_docs, t2t_st = load_source("pretrain_t2t", t2t_path, iter_jsonl_texts, dedup="full",
                                   quota_tokens=q[1], min_chars=min_chars,
                                   limit=args.limit_per_source, rng=rng, dry=args.dry_run,
                                   classify=classify_t2t, keep_prob=T2T_KEEP_PROB)
    pb_docs, pb_st = load_source("perfectblend_mathcode", pb_path, iter_csv_texts, dedup="full",
                                 quota_tokens=q[2], min_chars=min_chars,
                                 limit=args.limit_per_source, rng=rng, dry=args.dry_run)
    print(f"load done in {time.time() - t0:.0f}s", flush=True)

    corpus = ([(t, "pretrain_data") for t in cn_docs]
              + [(t, "pretrain_t2t") for t in t2t_docs]
              + [(t, "perfectblend_mathcode") for t in pb_docs])
    del cn_docs, t2t_docs, pb_docs
    print(f"corpus: {len(corpus):,} documents", flush=True)
    if not args.no_shuffle:
        rng.shuffle(corpus)
        print("shuffled", flush=True)

    prefix = Path(args.output_prefix)
    tokens_path = prefix.with_suffix(".bin")
    meta_path = prefix.with_suffix(".json")
    for p in (tokens_path, meta_path):
        if p.exists() and not args.overwrite:
            sys.exit(f"{p} already exists; pass --overwrite")

    print(f"packing -> {tokens_path}", flush=True)
    sequence_count, pack_stats = pack(corpus, args.tokenizer_path, tokens_path,
                                      args.max_seq_len, args.batch_rows)
    os.replace(tokens_path.with_suffix(".bin.tmp"), tokens_path)

    manifest = {
        "format": PACKED_FORMAT,
        "dtype": "uint32",
        "sequence_length": args.max_seq_len,
        "tokens_per_record": args.max_seq_len + 1,
        "num_sequences": sequence_count,
        "tokenizer_vocab_size": pack_stats["vocab_size"],
        "eos_token_id": EOS_TOKEN_ID,
        "documents": pack_stats["documents"],
        "source_tokens": pack_stats["source_tokens"],
        "source_token_breakdown": pack_stats["source_token_breakdown"],
        "discarded_tail_tokens": pack_stats["discarded_tail_tokens"],
        "add_eos": True,
        "dry_run": args.dry_run,
        "mix_targets_tokens_est": {
            "pretrain_data": q[0],
            "pretrain_t2t": q[1],
            "perfectblend_mathcode": q[2],
        },
        "source_stats": [cn_st.summary(), t2t_st.summary(), pb_st.summary()],
        "seed": args.seed,
    }
    meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    usable = sequence_count * (args.max_seq_len + 1)
    total = max(pack_stats["source_tokens"], 1)
    print(f"\nPacked {pack_stats['documents']:,} documents into {sequence_count:,} sequences")
    print(f"source tokens: {pack_stats['source_tokens']:,} | usable: {usable:,} "
          f"| tail discarded: {manifest['discarded_tail_tokens']:,}")
    for k, v in sorted(pack_stats["source_token_breakdown"].items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v:,} tokens ({100.0 * v / total:.1f}%)")
    print(f"bin -> {tokens_path}\nmanifest -> {meta_path}")


if __name__ == "__main__":
    main()
