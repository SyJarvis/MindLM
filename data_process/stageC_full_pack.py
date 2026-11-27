#!/usr/bin/env python3
"""Stage C: materialize the FULL packed dataset (train + heldout) from the
streaming loader for the formal pretraining run.

Same generator contract as the smoke script (seed 20261120, heldout_ppm 5000,
window 50k, mindlm_tokenizer </s> boundary, seq 4096, record width 4097).
Writes <out>/train.{bin,json} and <out>/heldout.{bin,json} plus a plan json.
Resumable: skips a split whose manifest already reports the expected record
count (train est ~2.198M records, heldout ~11k at 5,000 ppm).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from v6_stream_loader import StreamingPretrainLoader

CORPUS = "/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.jsonl"
TOKDIR = "/mnt/data2/reexen_datasets/llm_training/MindLM/tokenizers/mindlm_tokenizer"
SEQ = 4096
SEED = 20261120
HELDOUT_PPM = 5000
WINDOW = 50_000


def write_split(tok, split, out_prefix, max_records=None):
    done_path = out_prefix + ".json"
    if os.path.exists(done_path):
        try:
            have = json.load(open(done_path)).get("num_sequences")
        except Exception:
            have = None
        if max_records is None and have:
            print(f"[skip] {split}: manifest exists with {have} records", flush=True)
            return have
    loader = StreamingPretrainLoader(CORPUS, tok, seq_len=SEQ, seed=SEED,
                                     heldout_ppm=HELDOUT_PPM, window_rows=WINDOW)
    records = tokens = 0
    t0 = time.time()
    with open(out_prefix + ".bin.tmp", "wb") as fh, open(out_prefix + ".rows.tmp", "w") as _:
        for rec in loader.stream(split=split, max_records=max_records):
            if len(rec) != SEQ + 1:
                raise ValueError(f"bad record width {len(rec)}")
            np.asarray(rec, dtype="<u4").tofile(fh)
            records += 1
            tokens += len(rec)
            if records % 100_000 == 0:
                rate = records / (time.time() - t0)
                print(f"[{split}] {records} records, {rate:.0f} rec/s, "
                      f"eta {((2_200_000 - records) / max(rate, 1)) / 3600:.1f}h", flush=True)
    os.replace(out_prefix + ".bin.tmp", out_prefix + ".bin")
    meta = {
        "format": "mindlm_packed_pretrain_v1",
        "dtype": "<u4",
        "loss_mask_dtype": "uint8",
        "sequence_length": SEQ,
        "tokens_per_record": SEQ + 1,
        "num_sequences": records,
        "tokenizer_vocab_size": tok.vocab_size,
        "boundary_token": tok.eos_token,
        "boundary_token_id": tok.eos_token_id,
        "packed_tokens": tokens,
        "supervised_tokens": records * SEQ,
        "add_boundary": True,
        "generator": "v6_stream_loader stage-C full pack",
        "corpus": CORPUS,
        "seed": SEED,
        "heldout_ppm": HELDOUT_PPM,
        "window_rows": WINDOW,
        "split": split,
    }
    with open(done_path, "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return records


def main():
    base = sys.argv[1]
    os.makedirs(base, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKDIR, local_files_only=True)
    tr = write_split(tok, "train", base + "/train")
    va = write_split(tok, "heldout", base + "/heldout")
    plan = {"train_records": tr, "heldout_records": va, "seq_len": SEQ,
            "seed": SEED, "heldout_ppm": HELDOUT_PPM, "window_rows": WINDOW,
            "boundary_token": tok.eos_token, "vocab": tok.vocab_size}
    with open(base + "/pack-plan.json", "w") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
