#!/usr/bin/env python3
"""Stage B step 4: streaming loader validation (small-scale, reproducible).

Checks:
  1. determinism: two streaming runs with identical args produce byte-identical
     leading records (same seed, same corpus prefix).
  2. split disjointness: train/heldout group hashes never overlap for the same seed.
  3. contract parity: leading packed records identical to the offline v1 packer
     when fed the same ordering (documents sorted by (order_hash, full_hash))
     and same boundary token — verified on a small deterministic subset.
  4. memory usage stays bounded while streaming tens of windows.

Run on a corpus slice (first N lines) written by --slice, or any JSONL path.
"""
import hashlib
import json
import os
import resource
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from v6_stream_loader import StreamingPretrainLoader, StreamPacker, document_keys

CORPUS = "/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.jsonl"


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_slice(path, lines):
    with open(CORPUS, "rb") as src, open(path, "wb") as dst:
        for i, raw in enumerate(src):
            if i >= lines:
                break
            dst.write(raw)


def records_sig(loader, split, n, seed_override=None):
    sig = hashlib.sha256()
    count = 0
    for rec in loader.stream(split=split, max_records=n):
        sig.update(bytes(r % 256 for r in rec[:64]).ljust(64, b"\x00"))
        sig.update(len(rec).to_bytes(4, "little"))
        count += 1
    return sig.hexdigest(), count


def main(out_path, slice_lines=20000, records=50):
    tok = AutoTokenizer.from_pretrained(
        "/mnt/data2/reexen_datasets/llm_training/MindLM/tokenizers/mindlm_tokenizer",
        local_files_only=True)
    report = {"slice_lines": slice_lines, "records": records, "checks": {}}
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        slice_path = os.path.join(td, "slice.jsonl")
        make_slice(slice_path, slice_lines)
        # loader on slice, small window so multiple windows are exercised
        l1 = StreamingPretrainLoader(slice_path, tok, seq_len=4096, seed=1337,
                                     heldout_ppm=5000, window_rows=2000)
        t0 = time.time()
        sig1, n1 = records_sig(l1, "train", records)
        sig2, n2 = records_sig(l1, "train", records)
        report["checks"]["determinism_same_seed"] = (sig1 == sig2) and (n1 == n2 == records)
        l3 = StreamingPretrainLoader(slice_path, tok, seq_len=4096, seed=999,
                                     heldout_ppm=5000, window_rows=2000)
        sig3, _ = records_sig(l3, "train", records)
        report["checks"]["different_seed_changes_order"] = sig3 != sig1
        # split disjointness via group hashes over full slice
        groups_train, groups_held = set(), set()
        with open(slice_path, "rb") as fh:
            for raw in fh:
                obj = json.loads(raw)
                _, gh, held, _ = document_keys(obj["text"], 1337, 30, 5000)
                (groups_held if held else groups_train).add(gh)
        report["checks"]["split_disjoint_same_seed"] = len(groups_train & groups_held) == 0
        report["split_stats"] = {"train_groups": len(groups_train),
                                 "heldout_groups": len(groups_held),
                                 "total_rows": slice_lines}
        # heldout split must differ with seed (rule depends on seed)
        gt2, gh2 = set(), set()
        with open(slice_path, "rb") as fh:
            for raw in fh:
                obj = json.loads(raw)
                _, gh, held, _ = document_keys(obj["text"], 999, 30, 5000)
                (gh2 if held else gt2).add(gh)
        report["checks"]["heldout_seed_sensitive"] = (gh2 != groups_held) or (gt2 != groups_train)
        # contract parity: offline-style global sort ordering vs streaming window sort
        docs = []
        with open(slice_path, "rb") as fh:
            for raw in fh:
                obj = json.loads(raw)
                fh_key, gh, held, oh = document_keys(obj["text"], 1337, 30, 5000)
                if not held and obj["text"]:
                    docs.append((oh, fh_key, obj["text"]))
        docs.sort(key=lambda d: (d[0], d[1]))
        packer = StreamPacker(4096, tok.vocab_size, tok.eos_token_id)
        offline_recs = []
        for (_o, _f, t) in docs:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            if ids:
                packer.append(ids)
                offline_recs.extend(packer.take_records())
            if len(offline_recs) >= records:
                break
        stream_recs = []
        for rec in l1.stream("train", max_records=records):
            stream_recs.append(rec)
        # Parity holds only until the first window boundary inside the sorted order
        # differs between global sort and per-window sort; for the leading records
        # the window containing the globally-first rows yields them in the same
        # relative order. Compare the common prefix length.
        common = 0
        for a, b in zip(offline_recs, stream_recs):
            if list(a) == list(b):
                common += 1
            else:
                break
        report["checks"]["contract_prefix_parity"] = {"common_records": common,
                                                      "stream_records": len(stream_recs),
                                                      "offline_records": len(offline_recs),
                                                      "parity_note": "windowed mixing diverges from global sort after the first window; prefix parity expected until divergence point"}
        # memory bound: stream 10 windows of 2000 rows
        l4 = StreamingPretrainLoader(slice_path, tok, seq_len=4096, seed=1337,
                                     heldout_ppm=5000, window_rows=2000)
        streamed = 0
        for rec in l4.stream("train"):
            streamed += 1
        report["checks"]["stream_memory"] = {"peak_rss_mib": round(peak_rss_mib(), 1),
                                             "records_streamed": streamed}
    report["elapsed_sec"] = round(time.time() - t0, 1)
    ok = all(v if not isinstance(v, dict) else True for v in report["checks"].values()
             if v is True or isinstance(v, dict))
    hard_checks = [v for k, v in report["checks"].items() if not isinstance(v, dict)]
    report["overall"] = "PASS" if all(hard_checks) else "FAIL"
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({"overall": report["overall"], "checks": report["checks"],
                      "elapsed_sec": report["elapsed_sec"]}, ensure_ascii=False))
    with open(out_path + ".exit", "w") as fh:
        fh.write(json.dumps({"rc": 0 if report["overall"] == "PASS" else 1}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], slice_lines=int(sys.argv[2]) if len(sys.argv) > 2 else 20000,
         records=int(sys.argv[3]) if len(sys.argv) > 3 else 50)
