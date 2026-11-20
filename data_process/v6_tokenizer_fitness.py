#!/usr/bin/env python3
"""Stage B step 2: tokenizer fitness probe on V6 corpus (per-source stratified sample).

Loads the local mindlm_tokenizer, samples N rows per source deterministically
(line-offset sampling with fixed seed), reports:
  - token count, chars, fertility (tokens/char), unique token coverage
  - UNK occurrences and asserted special tokens found
Writes JSON report; no corpus modification.
"""
import hashlib
import json
import random
import sys
import time

CORPUS = "/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.jsonl"
OUT = sys.argv[1]
SAMPLE_PER_SOURCE = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SEED = 20261120

sys.path.insert(0, "/mnt/data2/reexen_datasets/llm_training/MindLM")
from transformers import AutoTokenizer

t0 = time.time()
tok = AutoTokenizer.from_pretrained("/mnt/data2/reexen_datasets/llm_training/MindLM/tokenizers/mindlm_tokenizer",
                                    local_files_only=True, trust_remote_code=False)
vocab_size = tok.vocab_size
unk_id = tok.unk_token_id
eos_id = tok.eos_token_id
print("vocab:", vocab_size, "unk:", unk_id, unk_id and tok.convert_ids_to_tokens(unk_id), "eos:", eos_id,
      eos_id and tok.convert_ids_to_tokens(eos_id), flush=True)

# Pass 1: collect line offsets per source (cheap: parse only source field via small reads)
offsets = {}
with open(CORPUS, "rb") as fh:
    off = 0
    import io
    for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
        try:
            src = json.loads(raw)["source"]
        except Exception:
            src = "__bad__"
        offsets.setdefault(src, []).append(off)
        off += len(raw.encode("utf-8", errors="replace"))
print("pass1 done", {s: len(v) for s, v in offsets.items()}, flush=True)

rng = random.Random(SEED)
report = {"tokenizer": "mindlm_tokenizer", "vocab_size": vocab_size,
          "unk_token": unk_id, "eos_token": eos_id,
          "sample_per_source": SAMPLE_PER_SOURCE, "seed": SEED, "by_source": {}}
global_tokens = 0
global_chars = 0
global_unk = 0
for src in sorted(offsets):
    if src == "__bad__":
        continue
    n = min(SAMPLE_PER_SOURCE, len(offsets[src]))
    picks = sorted(rng.sample(offsets[src], n))
    texts = []
    with open(CORPUS, "rb") as fh:
        for p in picks:
            fh.seek(p)
            try:
                texts.append(json.loads(fh.readline())["text"])
            except Exception:
                pass
    tokens = 0
    chars = 0
    unk = 0
    used = set()
    for text in texts:
        ids = tok.encode(text)
        tokens += len(ids)
        chars += len(text)
        unk += sum(1 for i in ids if i == unk_id)
        used.update(ids)
    rec = {"sampled_rows": len(texts), "chars": chars, "tokens": tokens,
           "fertility_tokens_per_char": round(tokens / chars, 4) if chars else None,
           "unk_occurrences": unk, "unk_rate": round(unk / tokens, 6) if tokens else None,
           "unique_tokens_used": len(used),
           "unique_ratio_of_vocab": round(len(used) / vocab_size, 4),
           "est_total_tokens_full_corpus": None}
    if tokens:
        src_rows = len(offsets[src])
        rec["est_total_tokens_full_corpus"] = int(tokens / len(texts) * src_rows)
    report["by_source"][src] = rec
    global_tokens += tokens
    global_chars += chars
    global_unk += unk
    print(src, json.dumps(rec), flush=True)

report["global"] = {
    "sampled_chars": global_chars, "sampled_tokens": global_tokens,
    "fertility_tokens_per_char": round(global_tokens / global_chars, 4),
    "unk_occurrences": global_unk,
    "unk_rate": round(global_unk / global_tokens, 6),
    "est_total_tokens_full_corpus": sum(r["est_total_tokens_full_corpus"] for r in report["by_source"].values()),
}
report["elapsed_sec"] = round(time.time() - t0, 1)
verdict = "PASS" if report["global"]["unk_rate"] is not None and report["global"]["unk_rate"] < 0.001 else "REVIEW"
report["verdict"] = verdict
json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=2)
print("VERDICT", verdict, json.dumps(report["global"]), flush=True)
with open(OUT + ".exit", "w") as fh:
    fh.write(json.dumps({"rc": 0, "verdict": verdict}) + "\n")
