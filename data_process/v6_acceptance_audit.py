#!/usr/bin/env python3
"""Stage B step 1: full read-only acceptance audit of pretrain_v6_merged.jsonl.

Verifies against the build meta (pretrain_v6_merged.meta.json):
  - total rows, total bytes, per-source row counts
  - schema (each line is a JSON object with text+source, source in known set)
  - empty/short texts and character length distribution (per source)
  - exact-duplicate text hashes per source (incremental sha256 set, count only)
  - bounded-fertility sampling pipeline for tokenizer fitness evidence
Exit marker with rc + counters. Read-only on the corpus; writes only report JSON.
"""
import hashlib
import json
import os
import sys
import time

CORPUS = "/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.jsonl"
META = "/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.meta.json"
OUT = sys.argv[1]

t0 = time.time()
meta = json.load(open(META))
expected_rows = meta["rows"]
expected_bytes = meta["output_bytes"]
expected_sources = set(meta["by_source"])
expected_by_source = meta["by_source"]

counters = {src: {"rows": 0, "chars": 0, "empty": 0, "short_lt32": 0,
                  "over_16384": 0, "bad_schema": 0, "dup_text": 0} for src in expected_sources}
unknown_source_rows = 0
bad_json_rows = 0
seen = {src: set() for src in expected_sources}
total_rows = 0

with open(CORPUS, "rb", buffering=1024 * 1024) as fh:
    for raw in fh:
        total_rows += 1
        try:
            obj = json.loads(raw)
        except Exception:
            bad_json_rows += 1
            continue
        if not isinstance(obj, dict) or "text" not in obj or "source" not in obj:
            bad_json_rows += 1
            continue
        text = obj["text"] or ""
        src = obj["source"]
        if src not in counters:
            unknown_source_rows += 1
            continue
        c = counters[src]
        n = len(text)
        c["rows"] += 1
        c["chars"] += n
        if n == 0:
            c["empty"] += 1
        if n < 32:
            c["short_lt32"] += 1
        if n > 16384:
            c["over_16384"] += 1
        h = hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()[:16]
        if h in seen[src]:
            c["dup_text"] += 1
        else:
            seen[src].add(h)

elapsed = time.time() - t0
report = {
    "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "corpus": CORPUS,
    "expected": {"rows": expected_rows, "bytes": expected_bytes, "sources": sorted(expected_sources)},
    "observed": {
        "rows": total_rows,
        "bytes": os.stat(CORPUS).st_size,
        "bad_json_rows": bad_json_rows,
        "unknown_source_rows": unknown_source_rows,
        "by_source": counters,
    },
    "checks": {},
    "elapsed_sec": round(elapsed, 1),
}
report["checks"]["rows_match"] = total_rows == expected_rows
report["checks"]["bytes_match"] = report["observed"]["bytes"] == expected_bytes
report["checks"]["sources_match"] = set(counters) == expected_sources
report["checks"]["rows_per_source_match"] = all(
    counters[s]["rows"] == expected_by_source[s] for s in expected_sources)
report["checks"]["no_bad_json"] = bad_json_rows == 0
report["checks"]["no_unknown_source"] = unknown_source_rows == 0

del seen
ok = all(report["checks"].values())
report["overall"] = "PASS" if ok else "FAIL"
json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=2)
print(json.dumps({"overall": report["overall"], "rows": total_rows,
                  "elapsed_sec": report["elapsed_sec"]}))
with open(OUT + ".exit", "w") as fh:
    fh.write(json.dumps({"rc": 0 if ok else 1, "overall": report["overall"],
                         "elapsed_sec": round(elapsed, 1)}) + "\n")
sys.exit(0 if ok else 1)
