#!/usr/bin/env python3
"""Stage C: materialize a small packed smoke dataset from the streaming loader.

Produces <prefix>.bin + <prefix>.json in mindlm_packed_pretrain_v1 format with
the mindlm_tokenizer boundary (</s>), so pretrain.py can run CPU/GPU smoke
training through the exact production data path.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from v6_stream_loader import StreamingPretrainLoader

CORPUS = "/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.jsonl"


def write_split(tok, split, out_prefix, max_records):
    loader = StreamingPretrainLoader(CORPUS, tok, seq_len=4096, seed=20261120,
                                     heldout_ppm=5000, window_rows=50_000)
    buf = []
    records = 0
    tokens = 0
    boundary = tok.eos_token_id
    with open(out_prefix + ".bin.tmp", "wb") as fh:
        for rec in loader.stream(split=split, max_records=max_records):
            if len(rec) != 4097:
                raise ValueError(f"bad record width {len(rec)}")
            np.asarray(rec, dtype="<u4").tofile(fh)
            records += 1
            tokens += len(rec)
    os.replace(out_prefix + ".bin.tmp", out_prefix + ".bin")
    meta = {
        "format": "mindlm_packed_pretrain_v1",
        "dtype": "<u4",
        "loss_mask_dtype": "uint8",
        "sequence_length": 4096,
        "tokens_per_record": 4097,
        "num_sequences": records,
        "tokenizer_vocab_size": tok.vocab_size,
        "boundary_token": tok.eos_token,
        "boundary_token_id": boundary,
        "documents": None,
        "packed_tokens": tokens,
        "supervised_tokens": records * 4096,
        "add_boundary": True,
        "generator": "v6_stream_loader (stage C smoke)",
        "seed": 20261120,
        "heldout_ppm": 5000,
        "window_rows": 50_000,
        "split": split,
    }
    with open(out_prefix + ".json", "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return records, tokens


def main():
    base = sys.argv[1]
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    tok = AutoTokenizer.from_pretrained(
        "/mnt/data2/reexen_datasets/llm_training/MindLM/tokenizers/mindlm_tokenizer",
        local_files_only=True)
    tr, tr_tok = write_split(tok, "train", base + "/train", n_train)
    va, va_tok = write_split(tok, "heldout", base + "/heldout", n_val)
    print(json.dumps({"train_records": tr, "train_tokens": tr_tok,
                      "val_records": va, "val_tokens": va_tok,
                      "boundary_token": tok.eos_token, "vocab": tok.vocab_size}))


if __name__ == "__main__":
    main()
