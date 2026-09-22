"""Convert heterogeneous SFT sources into a unified Qwen3-chat-template messages CSV.

Sources:
  1. Legacy history/q/a CSVs (single & multi turn)  -> messages column
  2. perfectblend_sft_zh.jsonl                      -> messages column
  3. UltraData-SFT-Agent-2609 Tool_Use subset       -> tool-call trajectories
  4. UltraData Chinese samples across all subsets   -> short zh trajectories

Output: data/sft_qwen3_combined.csv with a single `messages` column
(JSON string of OpenAI-style message list, rendered by qwen3 chat template).
"""

import argparse
import glob
import json
import random

import pandas as pd
from transformers import AutoTokenizer

REPO = "/home/runke.zhong.srv/workspace/MindLM"
ULTRA = "/mnt/data2/reexen_datasets/llm_training/sft/UltraData-SFT-Agent-2609/data"


def zh_ratio(text: str) -> float:
    if not text:
        return 0.0
    han = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return han / max(len(text), 1)


def called_function_names(record: dict) -> set:
    names = set()
    for m in record.get("messages", []):
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name"):
                names.add(fn["name"])
    return names


def cap_tools(tools: list, called: set, max_tools: int) -> list:
    """Cap a row's schema list before rendering, keeping it self-consistent.

    Called functions are always kept (dropping one would make a supervised
    ``<tool_call>`` reference an undefined tool); remaining slots are filled
    with the row's own schemas in their original order so the distractor set
    stays domain-coherent. Rows whose called set alone exceeds the cap are
    passed through unchanged rather than corrupted.
    """
    if max_tools <= 0 or not tools or len(tools) <= max_tools:
        return tools
    kept, seen = [], set()
    for tool in tools:
        name = (tool.get("function") or {}).get("name")
        if name in called and name not in seen:
            kept.append(tool)
            seen.add(name)
    for tool in tools:
        if len(kept) >= max_tools:
            break
        name = (tool.get("function") or {}).get("name")
        if name not in seen:
            kept.append(tool)
            seen.add(name)
    return kept


def legacy_rows(df: pd.DataFrame) -> list:
    rows = []
    for sample in df.itertuples(index=False):
        history = sample.history if hasattr(sample, "history") else "[]"
        try:
            history = json.loads(history) if isinstance(history, str) and history.strip().startswith("[") else []
        except (json.JSONDecodeError, ValueError):
            history = []
        messages = []
        for turn in history if isinstance(history, list) else []:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                messages.append({"role": "user", "content": str(turn[0])})
                messages.append({"role": "assistant", "content": str(turn[1])})
        messages.append({"role": "user", "content": str(sample.q)})
        messages.append({"role": "assistant", "content": str(sample.a)})
        rows.append({"messages": json.dumps(messages, ensure_ascii=False), "source": "legacy"})
    return rows


def normalize_ultra(record: dict, max_tools: int = 0) -> dict | None:
    """Map an UltraData record onto OpenAI message conventions for the qwen3 template.

    E2: the record's ``tools`` schema list is preserved in a dedicated CSV
    column so the bin renderer can pass it through ``apply_chat_template(tools=...)``
    — the same rendering path used by serving and the tool eval probes.
    """
    messages = []
    for m in record.get("messages", []):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                fn = call.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                calls.append({"type": "function", "function": {"name": fn.get("name"), "arguments": args or "{}"}})
            messages.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        elif role in ("user", "system", "tool") and m.get("content"):
            messages.append({"role": role, "content": m["content"]})
    if not any(m["role"] == "assistant" for m in messages):
        return None
    tools = record.get("tools")
    if not isinstance(tools, list) or not tools:
        tools = None
    row = {"messages": json.dumps(messages, ensure_ascii=False), "source": "ultra"}
    if tools:
        tools = cap_tools(tools, called_function_names(record), max_tools)
        row["tools"] = json.dumps(tools, ensure_ascii=False)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=f"{REPO}/qwen3_tokenizer")
    parser.add_argument("--max_tokens", type=int, default=3600)
    parser.add_argument("--tool_limit", type=int, default=60000)
    parser.add_argument("--zh_limit", type=int, default=20000)
    parser.add_argument("--max_tools", type=int, default=10,
                        help="cap per-row tool schemas before rendering; called functions always kept; 0 = no cap")
    parser.add_argument("--output", default=f"{REPO}/data/sft_qwen3_combined.csv")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = []

    # 1) legacy CSVs + zh jsonl
    for path in [f"{REPO}/data/sft_data_single.csv", f"{REPO}/data/sft_data_multi.csv"]:
        df = pd.read_csv(path, usecols=["history", "q", "a"])
        rows.extend(legacy_rows(df))
        print(f"legacy {path}: total {len(rows)}")
    jl = pd.read_json(f"{REPO}/data/perfectblend_sft_zh.jsonl", lines=True)
    for sample in jl.itertuples(index=False):
        messages = [{"role": "user", "content": str(sample.q)}, {"role": "assistant", "content": str(sample.a)}]
        rows.append({"messages": json.dumps(messages, ensure_ascii=False), "source": "pb_zh"})

    # 2) UltraData: tool-call trajectories + chinese samples
    random.seed(1337)
    tool_rows, zh_rows = [], []
    for path in sorted(glob.glob(f"{ULTRA}/*/*.jsonl")):
        is_tool_file = "Tool_Use" in path
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user_text = " ".join(
                    m["content"] for m in record.get("messages", []) if m.get("role") == "user" and m.get("content")
                )
                # zh candidate: chinese-dominant query, any subset
                if len(zh_rows) < args.zh_limit and zh_ratio(user_text[:300]) > 0.3:
                    row = normalize_ultra(record, args.max_tools)
                    if row:
                        zh_rows.append(row)
                # tool candidate: must contain assistant tool_calls
                if is_tool_file and len(tool_rows) < args.tool_limit:
                    if any(m.get("role") == "assistant" and m.get("tool_calls") for m in record.get("messages", [])):
                        row = normalize_ultra(record, args.max_tools)
                        if row:
                            tool_rows.append(row)
        print(f"scanned {path}: tool={len(tool_rows)} zh={len(zh_rows)}")

    # 3) length filter for ultra rows with the real tokenizer (rendered length).
    # E2: render WITH the row's tools schema — the <tools> block adds tokens, so
    # filtering must happen after it exists (docs/sft.md §2 requirement 1).
    def keep_short(row):
        messages = json.loads(row["messages"])
        tools = json.loads(row["tools"]) if row.get("tools") else None
        try:
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, tools=tools,
            )
        except Exception:
            return False
        return len(tok(rendered, add_special_tokens=False).input_ids) <= args.max_tokens

    tool_kept = [r for r in tool_rows if keep_short(r)]
    zh_kept = [r for r in zh_rows if keep_short(r)]
    print(f"ultra tool: {len(tool_rows)} -> {len(tool_kept)} after <={args.max_tokens} tokens")
    print(f"ultra zh:   {len(zh_rows)} -> {len(zh_kept)} after <={args.max_tokens} tokens")

    rows.extend(tool_kept)
    rows.extend(zh_kept)
    out = pd.DataFrame(rows).sample(frac=1.0, random_state=1337).reset_index(drop=True)
    out.to_csv(args.output, index=False)
    print(f"WROTE {args.output}: {len(out)} rows, source counts:\n{out.source.value_counts()}")


if __name__ == "__main__":
    main()
