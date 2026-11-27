#!/usr/bin/env python3
"""Build sft_e2_combined_v4.csv from v3: dedup + segmentation + token-budget mix.

Alignment with training (data_process/sft.py + trainers/sft_grouped.py):
  - rows render exactly like the packing script: apply_chat_template(
    add_generation_prompt=False, enable_thinking=False, tools=<row tools>)
  - every emitted row renders to <= --max-tokens (default 4000; bin cap 4096)
  - every segment ends with an assistant turn so supervision intervals exist
  - CSV schema unchanged: messages, tools, source, domain

Data fixes applied on the fly:
  - tools stored as Python repr (single quotes) -> reparsed and re-serialized
    as strict JSON (would otherwise be dropped by the packing script)
  - oversized tool schemas (system+tools alone > budget) -> tools cell emptied
    (the packing script would drop them anyway)

Segmentation note: the Qwen3 template merges consecutive tool messages into
user blocks, so message/block counts differ. Boundaries are therefore located
via assistant markers (count-validated against the messages list).

Pipeline
  pass 1: stream v3; MinHash/LSH near-dup removal on the first user prompt
          (global, keep-first); real-tokenizer counting; non-ultra rows
          segmented (if over budget) and written; ultra rows only measured.
  sampling: trajectory-level uniform sample of ultra rows (fixed seed) until
          the ultra token budget is reached.
  pass 2: re-stream v3, segment only the selected ultra rows, append to v4.
"""
import argparse
import bisect
import csv
import json
import multiprocessing as mp
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(10 ** 9)

# ----------------------------- MinHash constants -----------------------------
N_PERM = 32
N_BAND = 8
ROWS_PER_BAND = N_PERM // N_BAND      # 4
NGRAM = 8
MAX_GRAMS = 512
SIG_CHARS = 4000                      # first-user chars fed to the signature
SIG_AGREE = 28                        # >=28/32 equal perms => near-duplicate

_rs = np.random.RandomState(7)
A = _rs.randint(1, 2 ** 63 - 1, size=N_PERM).astype(np.uint64)
B = _rs.randint(0, 2 ** 63 - 1, size=N_PERM).astype(np.uint64)

_WS = re.compile(r"\s+")
ASST_MARK = "<|im_start|>assistant\n"
BLOCK_MARK = "<|im_start|>"
MAXTOK = 4000        # set from args in main() before workers are forked
TOOLS_CHAR_CAP = 4000  # schemas bigger than this (≈1k tokens) are dropped
SYS_TOKEN_CAP = 600    # prompts with a system block bigger than this get a
                        # generic system when segmentation is needed (agent
                        # systems embed full tool manuals, ~7k tokens)
GENERIC_SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


def sig_of(text):
    """32-perm MinHash signature of normalized char 8-grams."""
    t = _WS.sub("", text or "").lower()[:SIG_CHARS]
    # strip the shared agent boilerplate header so distinct tasks don't collide
    cut = t.find("</uploadedfiles>")
    if cut >= 0:
        t = t[cut + len("</uploadedfiles>"):]
    n = len(t)
    if n < NGRAM + 1:
        grams = np.array([zlib.crc32(t.encode("utf-8", "ignore"))], dtype=np.uint64)
    else:
        step = max(1, (n - NGRAM) // MAX_GRAMS)
        count = len(range(0, n - NGRAM + 1, step))
        grams = np.fromiter(
            (zlib.crc32(t[i:i + NGRAM].encode("utf-8", "ignore"))
             for i in range(0, n - NGRAM + 1, step)),
            dtype=np.uint64, count=count)
    with np.errstate(over="ignore"):
        h = A[None, :] * grams[:, None] + B[None, :]
    return h.min(axis=0)


def parse_tools(raw):
    """Decode the tools cell; returns (tools_list_or_None, json_str, was_fixed,
    dropped_big)."""
    if not raw or not str(raw).strip():
        return None, "", False, False
    try:
        value = json.loads(raw)
        fixed = False
    except Exception:
        import ast
        try:
            value = ast.literal_eval(raw)
        except Exception:
            return None, "", False, False
        fixed = True
    if not isinstance(value, list) or not value:
        return None, "", False, False
    out = json.dumps(value, ensure_ascii=False)
    if len(out) > TOOLS_CHAR_CAP:
        return None, "", fixed, True
    return value, out, fixed, False


# ------------------------------- worker state --------------------------------
_TOK = None


def _init_worker(tok_path):
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tok_path)


def _render_text(messages, tools):
    text = _TOK.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False, tools=tools or None)
    if not isinstance(text, str):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in text)
    return text


def _measure(messages, tools):
    """Render + tokenize once with offsets. Returns (ids, tok_start, text)."""
    text = _render_text(messages, tools)
    enc = _TOK(text, add_special_tokens=False, return_offsets_mapping=True)
    tok_start = [o[0] for o in enc.offset_mapping]
    return enc.input_ids, tok_start, text


def _truncate_assistant(messages, tools, budget):
    """Last resort: shrink the trailing assistant content until it fits."""
    for _ in range(5):
        last = messages[-1]
        c = last.get("content") or ""
        cut = int(len(c) * 0.6)
        if cut <= 0:
            return None
        messages = messages[:-1] + [dict(last, content=c[:cut])]
        ids = _TOK(_render_text(messages, tools), add_special_tokens=False).input_ids
        if len(ids) <= budget:
            return messages, len(ids)
    return None


def segment(messages, tools, budget):
    """Split into <=budget-token pieces, each ending with an assistant turn.

    Returns (segments, truncated_flag); segments None when unrecoverable.
    """
    if messages and messages[0].get("role") == "system":
        sys_ids = _TOK(_render_text([messages[0]], None),
                       add_special_tokens=False).input_ids
        if len(sys_ids) > SYS_TOKEN_CAP:
            messages = [GENERIC_SYSTEM] + messages[1:]
    ids, tok_start, text = _measure(messages, tools)
    if len(ids) <= budget:
        return [(messages, len(ids))], False

    asst_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    marks = []
    p = text.find(ASST_MARK)
    while p >= 0:
        marks.append(p)
        p = text.find(ASST_MARK, p + 1)
    if len(marks) != len(asst_idx):
        return None, False              # fake assistant markers inside content

    # end char/token of each assistant block
    ends = []
    for p in marks:
        e_char = text.find(BLOCK_MARK, p + len(ASST_MARK))
        if e_char < 0:
            e_char = len(text)
        ends.append(bisect.bisect_left(tok_start, e_char))

    sys_msg = [messages[0]] if messages and messages[0].get("role") == "system" else []
    segs, truncated = [], False
    s_tok, s_msg = 0, 0

    def emit_window(last_k):
        """Emit messages[s_msg : asst_idx[last_k]+1]; returns True on success."""
        nonlocal s_msg, s_tok, truncated
        end = asst_idx[last_k] + 1
        piece = messages[s_msg:end]
        if sys_msg and s_msg > 0:
            piece = sys_msg + piece
        n_ids = _TOK(_render_text(piece, tools),
                     add_special_tokens=False).input_ids
        if len(n_ids) <= budget:
            segs.append((piece, len(n_ids)))
        else:
            fixed = _truncate_assistant(piece, tools, budget)
            if fixed is None:
                return False
            segs.append(fixed)
            truncated = True
        s_msg = end
        s_tok = ends[last_k]
        return True

    k = 0
    while k < len(marks):
        e = None
        j = k
        while j < len(marks) and ends[j] - s_tok <= budget:
            e = j
            j += 1
        if e is not None:
            if not emit_window(e):
                # window cannot fit even after truncation: skip past it
                s_msg = asst_idx[e] + 1
                s_tok = ends[e]
                truncated = True
            k = j
        else:
            # assistant k alone exceeds the budget (e.g. a giant tool_call
            # whose content lives in tool_calls, not truncatable) -> skip it
            s_msg = asst_idx[k] + 1
            s_tok = ends[k]
            truncated = True
            k += 1
    return segs, truncated


def process_row(payload):
    """payload: (idx, source, domain, messages_raw, tools_raw, do_segment)
    result: (idx, source, domain, tools_json, sig, status, rows, n_tokens)
      rows = [(messages_list, ntok), ...] when do_segment else None"""
    idx, src, dom, msg_raw, tools_raw, do_seg = payload
    try:
        messages = json.loads(msg_raw)
    except Exception:
        return (idx, src, dom, "", None, "parse", None, 0)
    tools, tools_json, fixed, big = parse_tools(tools_raw)
    fu = ""
    for m in messages:
        if m.get("role") == "user":
            fu = (m.get("content") or "")[:SIG_CHARS]
            break
    sig = sig_of(fu)
    try:
        if do_seg:
            segs, truncated = segment(messages, tools, MAXTOK)
            if segs is None:
                return (idx, src, dom, "", sig, "seg_fail", None, 0)
            status = "ok_trunc" if truncated else "ok"
            if fixed:
                status += "_toolsfix"
            if big:
                status += "_toolsdrop"
            return (idx, src, dom, tools_json, sig, status, segs,
                    sum(n for _, n in segs))
        _, ids, _ = _measure(messages, tools)
        return (idx, src, dom, tools_json, sig, "ok", None, len(ids))
    except Exception:
        return (idx, src, dom, "", sig, "render", None, 0)


# ------------------------------- orchestration -------------------------------
# any status starting with "ok" (possibly combined with _trunc/_toolsfix/
# _toolsdrop flags) is emitted


def percentile(vals, p):
    if not vals:
        return 0
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p / 100))]


def main():
    global MAXTOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="/home/runke.zhong.srv/MindLM/full_data_v2/sft_e2_combined_v3.csv,"
                            "/home/runke.zhong.srv/MindLM/full_data_v2/sft_t2t_qa.csv")
    ap.add_argument("--output", default="/home/runke.zhong.srv/MindLM/full_data_v2/sft_e2_combined_v5.csv")
    ap.add_argument("--stats", default="/home/runke.zhong.srv/MindLM/full_data_v2/prepare_stats_v5.json")
    ap.add_argument("--tokenizer", default="/home/runke.zhong.srv/MindLM/qwen3_tokenizer")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--min-tokens", type=int, default=16)
    ap.add_argument("--ultra-source", default="ultra")
    ap.add_argument("--ultra-budget", type=int, default=1_500_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()
    MAXTOK = args.max_tokens
    min_tok = args.min_tokens
    input_list = [p.strip() for p in args.input.split(",") if p.strip()]

    stats = Counter()
    tok_by_src = defaultdict(list)
    rows_out_by_src = Counter()
    ultra_meta = {}
    sigs = {}
    bands = [dict() for _ in range(N_BAND)]
    t0 = time.time()

    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers, initializer=_init_worker,
                    initargs=(args.tokenizer,))

    def is_dup(idx, sig):
        keys = []
        for b in range(N_BAND):
            key = sig[b * ROWS_PER_BAND:(b + 1) * ROWS_PER_BAND].tobytes()
            keys.append(key)
            other = bands[b].get(key)
            if other is not None:
                prev = np.frombuffer(sigs[other], dtype=np.uint64)
                if int((prev == sig).sum()) >= SIG_AGREE:
                    return True
        for b, key in enumerate(keys):
            bands[b].setdefault(key, idx)
        sigs[idx] = sig.tobytes()
        return False

    fout = open(args.output, "w", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    writer.writerow(["messages", "tools", "source", "domain"])

    def emit(rows, tools_json, src, dom):
        for msg_obj, ntok in rows:
            writer.writerow([json.dumps(msg_obj, ensure_ascii=False),
                             tools_json or "", src, dom])
        rows_out_by_src[src] += len(rows)
        tok_by_src[src].extend(n for _, n in rows)

    def iter_rows():
        idx = 0
        for path in input_list:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    yield idx, row
                    idx += 1

    def gen_pass1():
        for idx, row in iter_rows():
            yield (idx, row.get("source", ""), row.get("domain", ""),
                   row["messages"], row.get("tools", ""),
                   row.get("source", "") != args.ultra_source)

    # ---------------- pass 1 ----------------
    n_in = 0
    for idx, src, dom, tools_json, sig, status, rows, ntok in pool.imap(
            process_row, gen_pass1(), chunksize=128):
        n_in += 1
        stats["scanned"] += 1
        stats["status_" + status] += 1
        if not status.startswith("ok"):
            stats["drop_" + status] += 1
            continue
        if rows is not None and len(rows) == 1 and rows[0][1] < min_tok:
            stats["drop_short"] += 1
            continue
        if rows is None and ntok < min_tok:
            stats["drop_short"] += 1
            continue
        if is_dup(idx, sig):
            stats["drop_dup_" + src] += 1
            continue
        if src == args.ultra_source:
            ultra_meta[idx] = ntok
            continue
        emit(rows, tools_json, src, dom)
        if n_in % 500000 == 0:
            print(f"[p1 {time.time()-t0:6.0f}s] in={n_in} out_rows={sum(rows_out_by_src.values())}",
                  flush=True)

    # ---------------- ultra sampling ----------------
    idxs = list(ultra_meta)
    random.Random(args.seed).shuffle(idxs)
    selected, acc = set(), 0
    for i in idxs:
        if acc >= args.ultra_budget:
            break
        selected.add(i)
        acc += ultra_meta[i]
    stats["ultra_traj_total"] = len(ultra_meta)
    stats["ultra_traj_selected"] = len(selected)
    stats["ultra_tokens_selected"] = acc
    print(f"sampling: {len(selected)}/{len(ultra_meta)} trajectories, "
          f"{acc/1e9:.2f}B tokens (budget {args.ultra_budget/1e9:.2f}B)", flush=True)

    # ---------------- pass 2 ----------------
    def gen_pass2():
        for idx, row in iter_rows():
            if idx in selected:
                yield (idx, row.get("source", ""), row.get("domain", ""),
                       row["messages"], row.get("tools", ""), True)

    for idx, src, dom, tools_json, sig, status, rows, ntok in pool.imap(
            process_row, gen_pass2(), chunksize=8):
        stats["p2_status_" + status] += 1
        if not status.startswith("ok"):
            stats["drop_p2_" + status] += 1
            continue
        emit(rows, tools_json, src, dom)

    pool.close()
    pool.join()
    fout.close()

    # ---------------- report ----------------
    report = {
        "scanned": stats["scanned"],
        "emitted_rows": sum(rows_out_by_src.values()),
        "emitted_rows_by_source": dict(rows_out_by_src),
        "emitted_tokens_by_source": {s: sum(v) for s, v in tok_by_src.items()},
        "token_percentiles_by_source": {
            s: {"p50": percentile(v, 50), "p90": percentile(v, 90),
                "p99": percentile(v, 99), "max": max(v) if v else 0}
            for s, v in tok_by_src.items() if v},
        "drops": {k: v for k, v in stats.items()
                  if k.startswith(("drop_", "ultra_"))},
        "status_counts": {k: v for k, v in stats.items()
                          if k.startswith(("status_", "p2_status_"))},
        "config": {"inputs": input_list,
                   "max_tokens": args.max_tokens, "min_tokens": min_tok,
                   "ultra_budget": args.ultra_budget, "seed": args.seed,
                   "tools_char_cap": TOOLS_CHAR_CAP},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    report["emitted_tokens_total"] = sum(report["emitted_tokens_by_source"].values())
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("token_percentiles_by_source",)},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
