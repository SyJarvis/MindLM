#!/usr/bin/env python3
"""Convert workspace/dir/sft_t2t.jsonl into the e2 CSV schema.

Input row: {"conversations": [{"role","content","reasoning_content"},...]}
Output CSV columns: messages, tools, source, domain  (source=t2t_qa)

Rules:
  - reasoning_content is KEPT on assistant messages (the Qwen3 chat template
    renders it as a <think>...</think> block, verified empirically; ~30% of
    rows carry real Chinese math/reasoning CoT)
  - rows without any assistant turn are dropped (unusable for SFT)
  - content None -> ""
"""
import csv
import json
import sys
import time
from collections import Counter

csv.field_size_limit(10 ** 9)

IN = "/home/runke.zhong.srv/workspace/dir/sft_t2t.jsonl"
OUT = "/home/runke.zhong.srv/MindLM/full_data_v2/sft_t2t_qa.csv"

stats = Counter()
t0 = time.time()
with open(IN, encoding="utf-8") as fin, \
     open(OUT, "w", newline="", encoding="utf-8") as fout:
    w = csv.writer(fout)
    w.writerow(["messages", "tools", "source", "domain"])
    for line in fin:
        stats["in"] += 1
        try:
            d = json.loads(line)
            conv = d.get("conversations")
            if not isinstance(conv, list) or not conv:
                stats["bad_shape"] += 1
                continue
            msgs = []
            ok = True
            for m in conv:
                role = m.get("role")
                if role not in ("system", "user", "assistant", "tool"):
                    ok = False
                    break
                msg = {"role": role, "content": m.get("content") or ""}
                rc = m.get("reasoning_content") or ""
                if role == "assistant" and rc.strip():
                    msg["reasoning_content"] = rc
                    stats["rc_nonempty"] += 1
                msgs.append(msg)
            if not ok:
                stats["bad_role"] += 1
                continue
            if not any(m["role"] == "assistant" for m in msgs):
                stats["no_assistant"] += 1
                continue
            if msgs[-1]["role"] != "assistant":
                stats["end_not_asst"] += 1
                continue
            w.writerow([json.dumps(msgs, ensure_ascii=False), "", "t2t_qa", ""])
            stats["out"] += 1
        except Exception:
            stats["parse"] += 1
        if stats["in"] % 1000000 == 0:
            print(f"[{time.time()-t0:5.0f}s] in={stats['in']} out={stats['out']}",
                  flush=True)

print(dict(stats), "elapsed", round(time.time() - t0, 1))
