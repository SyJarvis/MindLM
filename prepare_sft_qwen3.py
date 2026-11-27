"""Convert heterogeneous SFT sources into a unified Qwen3-chat-template messages CSV.

Sources:
  1. Legacy history/q/a CSVs (single & multi turn)  -> messages column
  2. perfectblend_sft_zh.jsonl                      -> messages column
  3. open-perfectblend (mlabonne) parquet            -> English chat/QA
  4. UltraData-SFT-Agent-2609 Tool_Use subset       -> tool-call trajectories
  5. UltraData Chinese samples across all subsets   -> short zh trajectories
  6. toolcall_pretrain_raw_v1.jsonl (optional)      -> legacy tool-call rows

Output: data/sft_qwen3_combined.csv with columns: messages, tools, source, domain
(messages is JSON string of OpenAI-style message list, rendered by qwen3 chat template).

Every kept row passes: canonical dedup (messages+tools), full chat template render,
and token length <= max_tokens.  Rendering errors count as invalid.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import tempfile
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    returned as the deduplicated called-only list (longer than the cap)
    rather than corrupted; callers must treat such rows as invalid.
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


def _is_empty(val) -> bool:
    """Check if a value is NaN, None, or blank string."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def _extract_content(content) -> str:
    """Safely extract a text string from content that may be str, list, or dict."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content",
                                 json.dumps(item, ensure_ascii=False)))))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def parse_messages(value) -> list | None:
    """Parse messages from a list or JSON string. Returns None on failure.

    Validates that at least one assistant message exists.
    """
    if isinstance(value, list):
        messages = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            messages = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(messages, list):
            return None
    else:
        return None
    if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
        return None
    return messages


def _parse_list_field(value) -> list | None:
    """Parse a field that is either already a list or a JSON string of one.

    Returns None for missing/blank/unparseable values and for JSON that is
    not a list.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def canonical_key(messages: list, tools: list | None = None) -> str:
    """Canonical dedup key: sort_keys JSON of messages, optionally suffixed with tools.

    Same messages with different tool schemas are treated as distinct entries.
    """
    key = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    if tools:
        key += "\t" + json.dumps(tools, sort_keys=True, separators=(",", ":"))
    return key


# ── Source processors ────────────────────────────────────────────────────────

def legacy_rows(df: pd.DataFrame) -> list:
    """Convert legacy history/q/a rows into unified message dicts.

    Returns a list of dicts with keys ``messages`` (JSON string) and ``source``.
    Rows with empty q/a are silently skipped (counted as missing by caller).
    """
    rows = []
    for sample in df.itertuples(index=False):
        if _is_empty(sample.q) or _is_empty(sample.a):
            continue
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
        rows.append({"messages": json.dumps(messages, ensure_ascii=False),
                      "source": "legacy"})
    return rows


def perfectblend_en_rows(df: pd.DataFrame) -> list:
    rows = []
    for sample in df.itertuples(index=False):
        conv = sample.conversations
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        messages = []
        valid = True
        for turn in conv:
            frm = turn.get("from", "")
            val = turn.get("value", "")
            if not val or (isinstance(val, float) and (val != val)):
                valid = False
                break
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(frm)
            if role is None:
                valid = False
                break
            messages.append({"role": role, "content": str(val)})
        if not valid or not any(m["role"] == "assistant" for m in messages):
            continue
        rows.append({"messages": json.dumps(messages, ensure_ascii=False),
                      "source": "perfectblend_en"})
    return rows


def _tool_name(tool: dict) -> str:
    """Extract the function name from a tool schema dict, or empty string."""
    return (tool.get("function") or {}).get("name") or ""


def _normalize_messages(raw_messages: list) -> list:
    """Map raw messages onto OpenAI message conventions.

    Assistant ``tool_calls`` are normalized (dict ``arguments`` become JSON
    strings); structured (dict/list) content is flattened via
    ``_extract_content``.  Non-dict entries are skipped defensively so a
    malformed line cannot crash the whole run.
    """
    messages = []
    for m in raw_messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                if not isinstance(fn, dict):
                    fn = {}
                args = fn.get("arguments")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                calls.append({
                    "type": "function",
                    "function": {"name": fn.get("name"), "arguments": args or "{}"},
                })
            content = _extract_content(m.get("content"))
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        elif role in ("user", "system", "tool", "assistant") and m.get("content"):
            content = _extract_content(m.get("content"))
            messages.append({"role": role, "content": content})
    return messages


def validate_and_normalize_ultra(record: dict, max_tools: int = 0,
                                 stats: dict | None = None) -> dict | None:
    """Validate and map an UltraData record onto OpenAI message conventions.

    * Tool-call rows must have a valid non-empty ``tools`` schema.
    * All called function names must exist in the schema.
    * After ``cap_tools``, still validate called names are in the capped set
      and that the distinct called set fits within ``max_tools`` (rows whose
      called set alone exceeds the cap are invalid and dropped).
    * Handles structured (dict/list) user content via ``_extract_content``.
    * Output preserves ``source`` and ``domain`` fields from the record.

    Returns a row dict with ``messages``, ``source``, optionally ``tools`` and
    ``domain``, or ``None`` when the record fails validation.
    """
    messages = _normalize_messages(record.get("messages", []))

    if not any(m["role"] == "assistant" for m in messages):
        if stats is not None:
            stats["invalid"] = stats.get("invalid", 0) + 1
        return None

    called = called_function_names({"messages": messages})
    tools = record.get("tools")

    # Tool-call rows must have a valid non-empty tools schema
    if called:
        if not isinstance(tools, list) or not tools:
            if stats is not None:
                stats["missing"] = stats.get("missing", 0) + 1
            return None
        # All called names must exist in schema
        schema_names = {_tool_name(t) for t in tools if _tool_name(t)}
        unknown = called - schema_names
        if unknown:
            if stats is not None:
                stats["invalid"] = stats.get("invalid", 0) + 1
            return None
        # Cap tools
        tools = cap_tools(tools, called, max_tools)
        # Still validate after cap: called names must survive the cap, and the
        # distinct called set itself must fit within max_tools
        if max_tools > 0:
            capped_names = {_tool_name(t) for t in tools if _tool_name(t)}
            still_unknown = called - capped_names
            if still_unknown or len(called) > max_tools:
                if stats is not None:
                    stats["invalid"] = stats.get("invalid", 0) + 1
                return None
    else:
        tools = None

    row = {"messages": json.dumps(messages, ensure_ascii=False), "source": "ultra"}
    # Preserve domain from record
    domain = record.get("domain") or record.get("source")
    if domain:
        row["domain"] = str(domain)
    if tools:
        row["tools"] = json.dumps(tools, ensure_ascii=False)
    return row


def validate_toolcall_record(record: dict, max_tools: int = 0,
                             stats: dict | None = None) -> dict | None:
    """Validate a legacy toolcall_pretrain_raw_v1.jsonl record.

    Line format: ``{"source", "messages", "tools"}`` where ``messages`` and
    ``tools`` may each be a list or a JSON string of a list.

    * Must contain assistant ``tool_calls``.
    * ``tools`` must be a valid non-empty list of tool schema dicts; lines
      missing ``tools`` are skipped (counted as ``missing``) and never
      degraded to plain chat rows.
    * All called function names must exist in the schema, and are
      re-validated after ``cap_tools`` (called-first pruning); rows whose
      distinct called set exceeds ``max_tools`` are invalid and dropped.
    * ``source`` is preserved from the record; ``domain`` is set to the
      record's ``source``, or ``toolcall_legacy`` when absent.

    Returns a row dict (``messages``/``tools``/``source``/``domain``) or
    ``None`` on failure; updates ``stats`` in-place when given.
    """

    def bump(counter: str) -> None:
        if stats is not None:
            stats[counter] = stats.get(counter, 0) + 1

    if not isinstance(record, dict):
        bump("invalid")
        return None

    raw_messages = _parse_list_field(record.get("messages"))
    if raw_messages is None:
        bump("invalid")
        return None

    messages = _normalize_messages(raw_messages)
    called = called_function_names({"messages": messages})
    if not called:
        bump("invalid")
        return None

    tools = _parse_list_field(record.get("tools"))
    if not tools:
        bump("missing")
        return None
    if not all(isinstance(t, dict) for t in tools):
        bump("invalid")
        return None

    schema_names = {_tool_name(t) for t in tools if _tool_name(t)}
    if called - schema_names:
        bump("invalid")
        return None

    tools = cap_tools(tools, called, max_tools)
    # Re-validate after cap: called names must survive the cap, and the
    # distinct called set itself must fit within max_tools
    if max_tools > 0:
        capped_names = {_tool_name(t) for t in tools if _tool_name(t)}
        if called - capped_names or len(called) > max_tools:
            bump("invalid")
            return None

    source = record.get("source")
    if isinstance(source, str) and source.strip():
        source = source.strip()
    else:
        source = "toolcall_legacy"
    return {
        "messages": json.dumps(messages, ensure_ascii=False),
        "tools": json.dumps(tools, ensure_ascii=False),
        "source": source,
        "domain": source,
    }


def deterministic_sample(rows: list, limit: int, seed: int = 42) -> list:
    """Sample deterministically with a fixed seed, *not* prefix truncation.

    Returns a new list; original list is not modified.
    """
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def check_render_length(row: dict, tok, max_tokens: int,
                          stats: dict | None = None) -> bool:
    """Render a row with the chat template and check its token length.

    Uses ``enable_thinking=False`` to avoid extraneous thinking tokens.
    Render exceptions count as ``invalid``; rows longer than ``max_tokens``
    count as ``over_long`` (only when ``stats`` is given).  Returns True if
    the row passes.  Does not write anything or touch dedup state.
    """
    messages = json.loads(row["messages"])
    tools = json.loads(row["tools"]) if row.get("tools") else None
    try:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools,
            enable_thinking=False,
        )
    except Exception:
        if stats is not None:
            stats["invalid"] = stats.get("invalid", 0) + 1
        return False
    if len(tok(rendered, add_special_tokens=False).input_ids) > max_tokens:
        if stats is not None:
            stats["over_long"] = stats.get("over_long", 0) + 1
        return False
    return True


def keep_short(row: dict, tok, max_tokens: int) -> bool:
    """Check if a rendered row fits within ``max_tokens``.

    Thin wrapper over ``check_render_length`` without stats.
    Exposed at module level so tests can call it without re-creating logic.
    """
    return check_render_length(row, tok, max_tokens)


def process_and_write(row_dict: dict, tok, max_tokens: int,
                      seen: set, writer, stats: dict,
                      already_validated: bool = False) -> bool:
    """Dedup, render, length-check a single row; write to spool if kept.

    When ``already_validated`` is True the row already passed the real
    render + token-length check during pre-filtering, so only canonical
    dedup and the spool write are performed: no re-rendering and no
    double-counting of ``invalid``/``over_long`` stats.

    Returns True if the row was written, False otherwise.
    Updates ``stats`` in-place.
    """
    messages = json.loads(row_dict["messages"])
    tools = json.loads(row_dict["tools"]) if row_dict.get("tools") else None
    source = row_dict.get("source", "unknown")
    domain = row_dict.get("domain", "")

    # Canonical dedup
    key = canonical_key(messages, tools)
    if key in seen:
        stats["duplicate"] += 1
        return False
    seen.add(key)

    # Render + length check (skipped when the caller pre-validated the row)
    if not already_validated and not check_render_length(
            row_dict, tok, max_tokens, stats):
        return False

    # Write to spool
    stats["kept"] += 1
    stats["source_counts"][source] = stats["source_counts"].get(source, 0) + 1
    if domain:
        stats["domain_counts"][domain] = stats["domain_counts"].get(domain, 0) + 1

    writer.writerow({
        "messages": row_dict["messages"],
        "tools": json.dumps(tools, ensure_ascii=False) if tools else "",
        "source": source,
        "domain": domain,
    })
    return True


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(REPO / "qwen3_tokenizer"))
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tool_limit", type=int, default=60000)
    parser.add_argument("--zh_limit", type=int, default=20000)
    parser.add_argument("--max_tools", type=int, default=8,
                        help="cap per-row tool schemas before rendering; "
                             "called functions always kept; 0 = no cap")
    parser.add_argument("--output", default=str(REPO / "data" / "sft_qwen3_combined.csv"))
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for deterministic sampling")
    parser.add_argument("--stats", type=str, default=None,
                        help="path to write stats JSON (optional)")
    parser.add_argument("--multilingual_zh_jsonl", type=str, default=None,
                        help="optional JSONL with multilingual Chinese samples")
    parser.add_argument("--toolcall_jsonl", type=str, default=None,
                        help="optional legacy toolcall_pretrain_raw_v1.jsonl; "
                             "each line has source/messages/tools (list or "
                             "JSON string); rows must contain assistant "
                             "tool_calls and a valid non-empty tools schema")
    parser.add_argument("--toolcall_limit", type=int, default=0,
                        help="max rows kept from --toolcall_jsonl after "
                             "render-length filtering; 0 = all available")
    # Explicit input paths
    parser.add_argument("--legacy_single", type=str, default=None,
                        help="path to legacy single-turn CSV")
    parser.add_argument("--legacy_multi", type=str, default=None,
                        help="path to legacy multi-turn CSV")
    parser.add_argument("--perfectblend_zh", type=str, default=None,
                        help="path to perfectblend_sft_zh.jsonl")
    parser.add_argument("--perfectblend_en", type=str, default=None,
                        help="path to open-perfectblend parquet directory")
    parser.add_argument("--ultra_root", type=str, default=None,
                        help="path to UltraData root directory")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Resolve input paths (use explicit arg if given, otherwise default)
    legacy_single = Path(args.legacy_single) if args.legacy_single \
        else REPO / "data" / "sft_data_single.csv"
    legacy_multi = Path(args.legacy_multi) if args.legacy_multi \
        else REPO / "data" / "sft_data_multi.csv"
    pb_path = Path(args.perfectblend_zh) if args.perfectblend_zh \
        else REPO / "data" / "perfectblend_sft_zh.jsonl"
    pb_en_path = Path(args.perfectblend_en) if args.perfectblend_en \
        else Path("/mnt/data2/reexen_datasets/llm_training/sft/open-perfectblend/data")
    ultra_root = Path(args.ultra_root) if args.ultra_root \
        else Path("/mnt/data2/reexen_datasets/llm_training/sft/UltraData-SFT-Agent-2609/data")

    stats = {
        "scanned": 0,
        "kept": 0,
        "missing": 0,
        "invalid": 0,
        "duplicate": 0,
        "over_long": 0,
        "source_counts": {},
        "domain_counts": {},
    }
    seen = set()

    # ── Output path: resolve & create the parent directory BEFORE
    # building, so the spool can live on the same filesystem as the final
    # output (an os.replace across filesystems would fail with EXDEV) ────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Temporary spool CSV (same dir as output → atomic os.replace) ────
    spool_fd, spool_path = tempfile.mkstemp(
        dir=out_path.parent, prefix="." + out_path.name + ".", suffix=".tmp")
    os.close(spool_fd)
    spool_file = open(spool_path, "w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(spool_file,
                                fieldnames=["messages", "tools", "source", "domain"])
        writer.writeheader()

        # ── 1) Legacy CSVs (chunked, filtered, written to spool immediately) ─
        # Deduplicate by resolved path: the same actual file passed as both
        # single & multi is scanned only once.  Nonexistent paths keep the
        # existing per-path SKIP handling.
        legacy_scanned = set()
        for csv_path in [legacy_single, legacy_multi]:
            if not csv_path.exists():
                print(f"SKIP {csv_path}: not found")
                continue
            resolved = csv_path.resolve()
            if resolved in legacy_scanned:
                continue
            legacy_scanned.add(resolved)
            for chunk in pd.read_csv(csv_path, usecols=["history", "q", "a"], chunksize=100000):
                total = len(chunk)
                raw = legacy_rows(chunk)
                stats["scanned"] += total
                stats["missing"] += total - len(raw)
                for r in raw:
                    process_and_write(r, tok, args.max_tokens, seen, writer, stats)
            print(f"legacy {csv_path}: scanned={stats['scanned']} kept={stats['kept']}")

        # ── 2) perfectblend_sft_zh.jsonl ────────────────────────────────────
        if pb_path.exists():
            with open(pb_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    stats["scanned"] += 1
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        stats["invalid"] += 1
                        continue
                    q = sample.get("q", "")
                    a = sample.get("a", "")
                    if _is_empty(q) or _is_empty(a):
                        stats["missing"] += 1
                        continue
                    messages = [
                        {"role": "user", "content": str(q)},
                        {"role": "assistant", "content": str(a)},
                    ]
                    row = {"messages": json.dumps(messages, ensure_ascii=False),
                           "source": "pb_zh"}
                    process_and_write(row, tok, args.max_tokens, seen, writer, stats)
            print(f"pb_zh: kept={stats['kept']}")
        else:
            print(f"SKIP {pb_path}: not found")

        # ── 2b) open-perfectblend English (parquet) ──────────────────────────
        if pb_en_path.exists():
            import glob as _glob
            parquet_files = sorted(_glob.glob(f"{pb_en_path}/*.parquet"))
            if not parquet_files:
                print(f"SKIP {pb_en_path}: no parquet files found")
            else:
                total_en = 0
                for pf in parquet_files:
                    chunk = pd.read_parquet(pf)
                    raw = perfectblend_en_rows(chunk)
                    total_en += len(chunk)
                    stats["scanned"] += len(chunk)
                    for r in raw:
                        process_and_write(r, tok, args.max_tokens, seen, writer, stats)
                print("perfectblend_en: scanned=" + str(total_en) + " kept=" + str(stats["kept"]) + "")
        else:
            print(f"SKIP {pb_en_path}: not found")

        # ── 3) Multilingual zh JSONL (optional) ─────────────────────────────
        if args.multilingual_zh_jsonl:
            m_path = Path(args.multilingual_zh_jsonl)
            if m_path.exists():
                with open(m_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        stats["scanned"] += 1
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            stats["invalid"] += 1
                            continue
                        msg_value = record.get("messages", record)
                        messages = parse_messages(msg_value)
                        if messages is None:
                            stats["invalid"] += 1
                            continue
                        row = {"messages": json.dumps(messages, ensure_ascii=False),
                               "source": "multilingual_zh"}
                        if "domain" in record:
                            row["domain"] = str(record["domain"])
                        process_and_write(row, tok, args.max_tokens, seen, writer, stats)
                print(f"multilingual_zh: kept={stats['kept']}")

        # ── 4) Legacy toolcall JSONL (optional) ─────────────────────────────
        toolcall_collected = []
        if args.toolcall_jsonl:
            tc_path = Path(args.toolcall_jsonl)
            if tc_path.exists():
                with open(tc_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        stats["scanned"] += 1
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            stats["invalid"] += 1
                            continue
                        row = validate_toolcall_record(record, args.max_tools, stats)
                        if row:
                            toolcall_collected.append(row)
                print(f"toolcall {tc_path}: collected={len(toolcall_collected)}")
            else:
                print(f"SKIP {tc_path}: not found")

        # ── 5) UltraData: collect candidates in memory (~500k rows) ─────────
        tool_collected, zh_collected = [], []
        ultra_pattern = f"{ultra_root}/*/*.jsonl"
        ultra_dirs = sorted(glob.glob(ultra_pattern))
        if not ultra_dirs:
            # Fallback to recursive glob
            ultra_dirs = sorted(glob.glob(f"{ultra_root}/**/*.jsonl", recursive=True))
        for path_str in ultra_dirs:
            path = Path(path_str)
            if not path.is_file():
                continue
            is_tool_file = "Tool_Use" in path_str
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    stats["scanned"] += 1

                    # Extract user text safely (handle dict/list multimodal content)
                    user_text_parts = []
                    for m in record.get("messages", []):
                        if m.get("role") == "user" and m.get("content"):
                            user_text_parts.append(_extract_content(m.get("content")))
                    user_text = " ".join(user_text_parts)
                    is_zh = zh_ratio(user_text[:300]) > 0.3

                    if is_zh:
                        # zh candidate: any subset
                        row = validate_and_normalize_ultra(record, args.max_tools, stats)
                        if row:
                            zh_collected.append(row)
                    if is_tool_file and any(
                            m.get("role") == "assistant" and m.get("tool_calls")
                            for m in record.get("messages", [])):
                        if not is_zh:
                            # Only process for tool if not already processed as zh
                            row = validate_and_normalize_ultra(record, args.max_tools, stats)
                            if row:
                                tool_collected.append(row)
            print(f"scanned {path_str}: tool_collected={len(tool_collected)} "
                  f"zh_collected={len(zh_collected)}")

        # ── 6) Filter candidates, then sample ──────────────────────────────
        # Run the real render + tokenize check on every candidate FIRST, so the
        # sampling quota is not consumed by rows that would be dropped as
        # invalid/over_long at write time.
        toolcall_filtered = [r for r in toolcall_collected
                             if check_render_length(r, tok, args.max_tokens, stats)]
        if toolcall_collected:
            print(f"toolcall filter: {len(toolcall_collected)} -> "
                  f"{len(toolcall_filtered)}")
        tool_filtered = [r for r in tool_collected
                         if check_render_length(r, tok, args.max_tokens, stats)]
        zh_filtered = [r for r in zh_collected
                       if check_render_length(r, tok, args.max_tokens, stats)]
        print(f"ultra filter: tool {len(tool_collected)} -> {len(tool_filtered)}, "
              f"zh {len(zh_collected)} -> {len(zh_filtered)}")

        tool_kept = deterministic_sample(tool_filtered, args.tool_limit, args.seed)
        zh_kept = deterministic_sample(zh_filtered, args.zh_limit, args.seed)
        # Seed derived from the global seed: stable across runs and a distinct
        # stream from the ultra_tool/ultra_zh samples.
        toolcall_kept = deterministic_sample(toolcall_filtered, args.toolcall_limit,
                                             args.seed + 9173)

        # Write already-validated rows: dedup-only pass, no re-render/tokenize.
        for row_list, src_label in [(tool_kept, "ultra_tool"), (zh_kept, "ultra_zh"),
                                    (toolcall_kept, "toolcall")]:
            kept_in_source = 0
            for r in row_list:
                if process_and_write(r, tok, args.max_tokens, seen, writer, stats,
                                     already_validated=True):
                    kept_in_source += 1
            print(f"{src_label}: {len(row_list)} sampled -> {kept_in_source} kept")

        spool_file.flush()
        spool_file.close()

        # ── 7) Output: atomic replace to final path (same filesystem) ──
        os.replace(spool_path, out_path)
    finally:
        # Failure path (any exception, including BaseException such as
        # KeyboardInterrupt): close the spool and delete it when it still
        # exists, i.e. when the atomic replace never happened.  The final
        # output is therefore only ever replaced by a complete, closed
        # spool.
        try:
            spool_file.close()  # no-op when already closed
        except OSError:
            pass
        if os.path.exists(spool_path):
            os.unlink(spool_path)
    print(f"WROTE {out_path}: {stats['kept']} rows")

    # ── 8) Stats ─────────────────────────────────────────────────────────
    for src, cnt in sorted(stats["source_counts"].items()):
        print(f"  {src}: {cnt}")
    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via a unique temp file in the same directory as the
        # final stats path; the temp is removed when anything fails.
        stats_fd, stats_tmp = tempfile.mkstemp(
            dir=stats_path.parent, prefix="." + stats_path.name + ".",
            suffix=".tmp")
        try:
            with os.fdopen(stats_fd, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            os.replace(stats_tmp, stats_path)
        except BaseException:
            if os.path.exists(stats_tmp):
                os.unlink(stats_tmp)
            raise
        print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
