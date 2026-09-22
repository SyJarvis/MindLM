"""Render an SFT messages-CSV into packed token bins for grouped SFT training.

E2: rows may carry a ``tools`` column (JSON array of OpenAI-style function
schemas). It is passed to ``apply_chat_template`` so the training input renders
the same ``<tools>`` block that serving and ``eval_sft_tool*.py`` produce.
"""
import argparse
import csv
import json
import struct
from pathlib import Path

from transformers import AutoTokenizer

REPO = str(Path(__file__).resolve().parents[1])


def render_row(tok, messages, tools=None):
    """Apply the Qwen3 chat template exactly like the serving/eval path.

    Returns ``(text, ids)``. ``tools`` must be a non-empty schema list or None;
    the Qwen3 template treats ``[]`` as "no tools".
    """
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False, tools=tools,
    )
    if not isinstance(text, str):
        text = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in text)
    ids = tok(text, add_special_tokens=False).input_ids
    return text, ids


def parse_tools(raw):
    """Decode the CSV ``tools`` cell. Returns a schema list or None.

    Empty cells and ``[]`` both mean "no tools"; malformed JSON also returns
    None so the row degrades to the plain rendering path instead of being lost.
    """
    if not raw or not str(raw).strip():
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return value or None


def assistant_sup_intervals(ids, assistant_ids, im_end_ids, keep):
    """Supervised intervals in Y=ids[1:] coordinates.

    Each interval covers one assistant body (including ``<tool_call>`` blocks)
    plus its closing ``<|im_end|>``, stored as ``[start, length)`` pairs.
    """
    sup = []
    cursor = 0
    while cursor <= keep - len(assistant_ids):
        if ids[cursor:cursor + len(assistant_ids)] != assistant_ids:
            cursor += 1
            continue
        body_start = cursor + len(assistant_ids)
        end = next((i for i in range(body_start, keep - len(im_end_ids) + 1)
                    if ids[i:i + len(im_end_ids)] == im_end_ids), None)
        if end is None:
            break
        # seq index s is target index s-1 in Y; ignore the header itself.
        target_start = max(0, body_start - 1)
        target_end = min(keep - 1, end + len(im_end_ids) - 1)
        if target_start < target_end:
            sup.append([target_start, target_end - target_start])
        cursor = end + len(im_end_ids)
    return sup


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=f"{REPO}/data/sft_qwen3_combined.csv")
    parser.add_argument("--tokenizer", default=f"{REPO}/qwen3_tokenizer")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--bin", default=f"{REPO}/data/sft_tokens_4096.bin")
    parser.add_argument("--meta", default=f"{REPO}/data/sft_tokens_4096.rows.jsonl")
    parser.add_argument("--stats", default=f"{REPO}/data/sft_tokens_4096.json")
    parser.add_argument("--progress_every", type=int, default=500000)
    args = parser.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    stats = {"total": 0, "errors": 0, "over_long": 0, "tokens": 0, "answer_tokens": 0,
             "rows_with_tools": 0, "tools_block_rendered": 0, "tools_parse_errors": 0}
    lengths = []

    assistant_ids = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    im_end_ids = tok("<|im_end|>", add_special_tokens=False).input_ids

    out = open(args.bin, "wb")
    meta = open(args.meta, "w", encoding="utf-8")
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            if stats["total"] % args.progress_every == 0:
                print(f"... {stats['total']} rows, over_long={stats['over_long']}", flush=True)
            tools = parse_tools(row.get("tools"))
            if tools:
                stats["rows_with_tools"] += 1
            try:
                messages = json.loads(row["messages"])
                text, ids = render_row(tok, messages, tools)
            except Exception:
                stats["errors"] += 1
                continue
            if tools and "</tools>" not in text:
                # A schema list that renders without the tools block means the
                # tokenizer template ignored it; training would silently diverge
                # from eval, so count it and fall back to the plain rendering.
                stats["tools_parse_errors"] += 1
                tools = None
                try:
                    text, ids = render_row(tok, messages, None)
                except Exception:
                    stats["errors"] += 1
                    continue
            if tools:
                stats["tools_block_rendered"] += 1

            keep = min(len(ids), args.max_tokens)
            if len(ids) > args.max_tokens:
                stats["over_long"] += 1

            sup = assistant_sup_intervals(ids, assistant_ids, im_end_ids, keep)
            if not sup:
                stats["errors"] += 1
                continue
            out.write(struct.pack(f"<{keep}I", *ids[:keep]))
            meta.write(json.dumps({
                "off": stats["tokens"],
                "n": keep,
                "sup": sup,
                "src": row.get("source", ""),
            }) + "\n")
            stats["tokens"] += keep
            stats["answer_tokens"] += sum(length for _, length in sup)
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
