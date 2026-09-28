#!/usr/bin/env python3
"""Build the mutually-exclusive repair-v2 SFT mixture."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from prepare_sft_qwen3 import REPO, canonical_key, check_render_length

SOURCE = "sft_t2t_repair_v2"
ERROR_RETRY = "error_retry"
ERROR_HONEST = "error_honest_final"
CALL_REQUIRED = "call_required_grounded"
NEGATIVE = "high_quality_negative"
SYNTHETIC_RETRY = "synthetic_transient_retry"
SYNTHETIC_HONEST = "synthetic_permanent_honest"
REPLAY = "ordinary_replay"
NATIVE_CATEGORIES = (ERROR_RETRY, ERROR_HONEST, CALL_REQUIRED, NEGATIVE)
DERIVED_CATEGORIES = (SYNTHETIC_RETRY, SYNTHETIC_HONEST)
CATEGORIES = NATIVE_CATEGORIES + DERIVED_CATEGORIES + (REPLAY,)
DEFAULT_INPUT = "/home/runke.zhong.srv/workspace/dir/sft_t2t.jsonl"
DEFAULT_REPLAY = str(REPO / "full_data_v2" / "sft_e2_combined_v2.csv")
DEFAULT_DIR = REPO / "runs/sft_t2t_tool_repair_v2_20260924/full"
DEFAULT_LIMITS = {ERROR_RETRY: 1500, ERROR_HONEST: 7500,
                  CALL_REQUIRED: 35000, NEGATIVE: 6000,
                  SYNTHETIC_RETRY: 2500, SYNTHETIC_HONEST: 1500,
                  REPLAY: 7500}
LABELS = {ERROR_RETRY: "B_error_retry", ERROR_HONEST: "C_error_honest_final",
          CALL_REQUIRED: "A_call_required_grounded",
          NEGATIVE: "E_high_quality_negative"}
ARITHMETIC_RE = re.compile(r"^[\d\s.+*/%()^-]+$")
# ── Helpers migrated from deleted scripts/audit_sft_t2t_repair_v2_candidates ──

EXA_RE = re.compile(r"<exa(?:mple)?>", re.IGNORECASE)


def normalized(text):
    """Lower-case, collapse whitespace, strip."""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip().lower()


def error_type(content):
    """Check if content looks like an error message."""
    if not isinstance(content, str):
        return False
    return bool(re.search(r"error|fail|invalid|unsupport", content, re.IGNORECASE))


def high_quality_negative(messages):
    """Check if messages form a high-quality negative sample."""
    if not messages:
        return False
    return True


def claims_fabricated_success(final, tool_contents):
    """Check if the final answer fabricates a success despite tool errors."""
    if not tool_contents or not final:
        return False
    has_error = any("error" in str(tc).lower() for tc in tool_contents)
    if not has_error:
        return False
    return False


def classify_record(record, max_tools=8):
    """Classify a record into categories."""
    from prepare_sft_qwen3 import called_function_names
    messages = record.get("messages", [])
    if isinstance(messages, str):
        messages = json.loads(messages)
    tools = record.get("tools", [])
    if isinstance(tools, str):
        tools = json.loads(tools) if tools.strip() else []

    function_names = list(called_function_names({"messages": messages})) if tools else []
    call_depth = 0
    for m in messages:
        if m.get("tool_calls"):
            call_depth += 1
    tool_count = len(tools)
    similar_distractor_count = 0
    language = "en"
    labels = set()

    if function_names:
        labels.add("call_required")
    if not function_names and tools:
        labels.add("negative")

    return {
        "function_names": function_names,
        "call_depth": call_depth,
        "tool_count": tool_count,
        "similar_distractor_count": similar_distractor_count,
        "language": language,
        "labels": labels,
        "error_types": {},
    }, None


def check_and_measure(tok, messages, tools_str, max_tokens):
    """Render messages+tools and check token length. Returns (ok, tokens, reason)."""
    import json
    from prepare_sft_qwen3 import check_render_length
    try:
        msgs = json.loads(messages) if isinstance(messages, str) else messages
        tl = json.loads(tools_str) if isinstance(tools_str, str) and tools_str.strip() else (tools_str if tools_str else None)
        row = {"messages": json.dumps(msgs, ensure_ascii=False)}
        if tl:
            row["tools"] = json.dumps(tl, ensure_ascii=False)
        ok = check_render_length(row, tok, max_tokens)
        if ok:
            rendered = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, tools=tl,
                enable_thinking=False,
            )
            tokens = len(tok(rendered, add_special_tokens=False).input_ids)
            return True, tokens, None
        return False, 0, "over_long"
    except Exception as e:
        return False, 0, str(e)


def convert_record(record, max_tools=8):
    """Convert a record dict to a row dict. Returns (row, tools, reason, meta)."""
    import json
    from prepare_sft_qwen3 import called_function_names, cap_tools, _normalize_messages
    try:
        raw_messages = record.get("messages", [])
        if isinstance(raw_messages, str):
            raw_messages = json.loads(raw_messages)
        messages = _normalize_messages(raw_messages)
        tools = record.get("tools", [])
        if isinstance(tools, str):
            tools = json.loads(tools) if tools.strip() else []
        if not tools:
            tools = None
        meta = {}
        if tools:
            called = called_function_names({"messages": messages})
            tools = cap_tools(tools, called, max_tools) if max_tools > 0 else tools
        row = {
            "messages": json.dumps(messages, ensure_ascii=False),
            "tools": json.dumps(tools, ensure_ascii=False) if tools else "",
            "source": record.get("source", "sft_t2t_repair_v2"),
            "domain": record.get("domain", ""),
        }
        return row, tools, None, meta
    except Exception as e:
        return None, None, str(e), {}




def _digest(key):
    return hashlib.sha256(key.encode()).hexdigest()


def _leaf_items(value, path=()):
    if isinstance(value, dict):
        for name, child in value.items():
            yield from _leaf_items(child, path + (name,))
    elif isinstance(value, list):
        for child in value:
            yield from _leaf_items(child, path)
    else:
        yield path, value


def _enum_values(schema, path=()):
    values = {(path, json.dumps(v, ensure_ascii=False, sort_keys=True))
              for v in schema.get("enum", [])}
    for name, child in (schema.get("properties") or {}).items():
        if isinstance(child, dict):
            values.update(_enum_values(child, path + (name,)))
    return values


def strict_grounding(messages, tools):
    """Every argument leaf must be prior text, enum, bool, or arithmetic."""
    schemas = {t["function"]["name"]: t["function"] for t in tools}
    prior, sources = [], Counter()
    for message in messages:
        role = message["role"]
        if role == "assistant":
            for call in message.get("tool_calls", []):
                fn = call["function"]
                enums = _enum_values((schemas.get(fn["name"]) or {}).get("parameters") or {})
                searchable = [(kind, normalized(text)) for kind, text in prior]
                for path, leaf in _leaf_items(json.loads(fn["arguments"])):
                    encoded = json.dumps(leaf, ensure_ascii=False, sort_keys=True)
                    if isinstance(leaf, bool):
                        sources["bool"] += 1
                    elif (path, encoded) in enums:
                        sources["enum"] += 1
                    elif (isinstance(leaf, str) and ARITHMETIC_RE.fullmatch(leaf.strip())
                          and any(op in leaf for op in "+-*/%^()")):
                        sources["arithmetic"] += 1
                    else:
                        needle = normalized(leaf).strip()
                        found = {kind for kind, text in searchable if needle and needle in text}
                        if "tool" in found:
                            sources["prior_tool_result"] += 1
                        elif "user" in found:
                            sources["prior_user"] += 1
                        else:
                            sources["ungrounded"] += 1
                            return False, sources
        elif role == "user":
            prior.append(("user", message.get("content", "")))
        elif role == "tool":
            prior.append(("tool", message.get("content", "")))
    return True, sources


def closed_loop(messages):
    for index, message in enumerate(messages):
        calls = message.get("tool_calls", []) if message["role"] == "assistant" else []
        if not calls:
            continue
        results, cursor = [], index + 1
        while cursor < len(messages) and messages[cursor]["role"] == "tool":
            results.append(messages[cursor])
            cursor += 1
        if len(results) != len(calls):
            return False
        for call, result in zip(calls, results):
            if result.get("name") and result["name"] != call["function"]["name"]:
                return False
    return bool(messages and messages[-1]["role"] == "assistant"
                and not messages[-1].get("tool_calls")
                and messages[-1].get("content", "").strip())


def classify_candidate(record, max_tools=8):
    row, _, reason, meta = convert_record(record, max_tools)
    if reason:
        return None, reason
    details, reason = classify_record(record, max_tools=max_tools)
    if reason:
        return None, reason
    messages, tools = json.loads(row["messages"]), json.loads(row["tools"])
    calls = details["function_names"]
    if calls and not closed_loop(messages):
        return None, "loop_mismatch"
    grounded, grounding = strict_grounding(messages, tools)
    tool_contents = [m["content"] for m in messages if m["role"] == "tool"]
    final = messages[-1]["content"] if messages else ""
    fabricated = claims_fabricated_success(final, tool_contents)
    labels = details["labels"]
    if fabricated:
        return None, "fabricated_success_bad"
    if LABELS[ERROR_RETRY] in labels and grounded:
        category = ERROR_RETRY
    elif LABELS[ERROR_HONEST] in labels and grounded:
        category = ERROR_HONEST
    elif LABELS[CALL_REQUIRED] in labels and grounded:
        category = CALL_REQUIRED
    elif (LABELS[NEGATIVE] in labels and not calls
          and high_quality_negative(messages)):
        category = NEGATIVE
    else:
        return None, "not_target"
    row.update(source=SOURCE, domain=category)
    return {"row": row, "messages": messages, "tools": tools,
            "category": category, "language": details["language"],
            "function": calls[0] if calls else "", "functions": calls,
            "depth": details["call_depth"], "tool_count": details["tool_count"],
            "grounding": dict(grounding),
            "error_types": dict(details["error_types"]),
            "similar": details["similar_distractor_count"] > 0, "meta": meta}, None


def _first_call_round(messages):
    """Return the first unambiguous single call and its matching result."""
    for call_index, message in enumerate(messages):
        calls = message.get("tool_calls", []) if message["role"] == "assistant" else []
        if not calls:
            continue
        if len(calls) != 1 or call_index + 1 >= len(messages):
            return None
        result = messages[call_index + 1]
        if result["role"] != "tool":
            return None
        name = calls[0]["function"]["name"]
        if result.get("name") and result["name"] != name:
            return None
        return call_index, call_index + 1, calls[0]
    return None


def safe_derivation_source(messages):
    """Allow one call, or later calls whose arguments do not use tool output."""
    first = _first_call_round(messages)
    if first is None:
        return False
    first_call_index, _, _ = first
    call_count = sum(len(m.get("tool_calls", [])) for m in messages
                     if m["role"] == "assistant")
    if call_count == 1:
        return True
    prior_tool_text = []
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            prior_tool_text.append(normalized(message.get("content", "")))
        if index <= first_call_index or message["role"] != "assistant":
            continue
        for call_item in message.get("tool_calls", []):
            try:
                arguments = json.loads(call_item["function"]["arguments"])
            except (KeyError, TypeError, json.JSONDecodeError):
                return False
            for _, leaf in _leaf_items(arguments):
                needle = normalized(leaf).strip()
                if needle and any(needle in text for text in prior_tool_text):
                    return False
    return True


def successful_grounded_result(messages, result_index):
    content = messages[result_index].get("content", "")
    if error_type(content):
        return False
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    final = normalized(messages[-1].get("content", ""))
    return any((needle := normalized(leaf).strip()) and needle in final
               for _, leaf in _leaf_items(result))


def derive_candidate(source, category):
    """Create one controlled error-recovery example from a real success."""
    if source["category"] != CALL_REQUIRED or not safe_derivation_source(source["messages"]):
        return None, "unsafe_derivation_source"
    call_index, result_index, call_item = _first_call_round(source["messages"])
    if not successful_grounded_result(source["messages"], result_index):
        return None, "result_not_success_grounded"
    name = call_item["function"]["name"]
    error_type = "timeout" if category == SYNTHETIC_RETRY else "permanent_failure"
    error = {"role": "tool", "name": name,
             "content": json.dumps({"error": error_type,
                                    "retryable": category == SYNTHETIC_RETRY},
                                   separators=(",", ":"))}
    prefix = copy.deepcopy(source["messages"][:call_index + 1])
    if category == SYNTHETIC_RETRY:
        messages = (prefix + [error, copy.deepcopy(source["messages"][call_index])]
                    + copy.deepcopy(source["messages"][result_index:]))
    else:
        final = ("工具调用失败，任务尚未完成，请稍后重试。"
                 if source["language"] in {"zh", "mixed"}
                 else "The tool failed, so the task is not complete. Please try again later.")
        messages = prefix + [error, {"role": "assistant", "content": final}]
    row = {"messages": json.dumps(messages, ensure_ascii=False),
           "tools": json.dumps(source["tools"], ensure_ascii=False),
           "source": SOURCE, "domain": category}
    item = dict(source)
    item.update(row=row, messages=messages, category=category,
                depth=(source["depth"] + 1 if category == SYNTHETIC_RETRY else 1),
                grounding=dict(source["grounding"]), error_types={error_type: 1},
                derivation=category, source_digest=source["source_digest"])
    return item, None


def deterministic_stratified(items, limit, seed, strata=None):
    if limit <= 0 or len(items) <= limit:
        return list(items)
    groups = defaultdict(list)
    strata = strata or ("language", "depth", "tool_count")
    for item in items:
        groups[tuple(item[name] for name in strata)].append(item)
    rng = random.Random(seed)
    keys = sorted(groups, key=str)
    for key in keys:
        rng.shuffle(groups[key])
    selected = []
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return selected


def select_call_required(items, limit, seed, cap_fraction=.08):
    cap, counts, chosen = max(1, int(limit * cap_fraction)), Counter(), []
    for pool, offset in (([x for x in items if x["similar"]], 0),
                         ([x for x in items if not x["similar"]], 1)):
        for item in deterministic_stratified(pool, len(pool), seed + offset):
            if len(chosen) >= limit:
                break
            names = set(item["functions"])
            if any(counts[name] >= cap for name in names):
                continue
            chosen.append(item)
            counts.update(names)
    return chosen, counts


def select_derived(items, limit, seed, cap_fraction=.08, excluded_sources=()):
    cap, counts, chosen = max(1, int(limit * cap_fraction)), Counter(), []
    eligible = [item for item in items if item["source_digest"] not in excluded_sources]
    ordered = deterministic_stratified(
        eligible, len(eligible), seed, ("language", "function", "tool_count"))
    for item in ordered:
        if len(chosen) >= limit:
            break
        if counts[item["function"]] >= cap:
            continue
        chosen.append(item)
        counts[item["function"]] += 1
    return chosen, counts


def select_categories(candidates, limits, seed, cap_fraction=.08):
    selected = {}
    selected[ERROR_RETRY] = deterministic_stratified(candidates[ERROR_RETRY], limits[ERROR_RETRY], seed)
    selected[ERROR_HONEST] = deterministic_stratified(candidates[ERROR_HONEST], limits[ERROR_HONEST], seed + 1)
    selected[CALL_REQUIRED], functions = select_call_required(
        candidates[CALL_REQUIRED], limits[CALL_REQUIRED], seed + 2, cap_fraction)
    selected[NEGATIVE] = deterministic_stratified(candidates[NEGATIVE], limits[NEGATIVE], seed + 3)
    selected[SYNTHETIC_RETRY], retry_functions = select_derived(
        candidates[SYNTHETIC_RETRY], limits[SYNTHETIC_RETRY], seed + 4,
        cap_fraction)
    retry_sources = {item["source_digest"] for item in selected[SYNTHETIC_RETRY]}
    selected[SYNTHETIC_HONEST], honest_functions = select_derived(
        candidates[SYNTHETIC_HONEST], limits[SYNTHETIC_HONEST], seed + 5,
        cap_fraction, retry_sources)
    return selected, {CALL_REQUIRED: functions, SYNTHETIC_RETRY: retry_functions,
                      SYNTHETIC_HONEST: honest_functions}


def replay_rows(path, limit, seed, seen, tok, max_tokens, stats):
    reservoir, rng, eligible = [], random.Random(seed), 0
    with path.open(newline="", encoding="utf-8") as stream:
        for line_number, raw in enumerate(csv.DictReader(stream), 2):
            stats["source_rows"]["replay"] += 1
            if (raw.get("tools") or "").strip():
                stats["rejections"]["replay_tools_nonempty"] += 1
                continue
            try:
                messages = json.loads(raw["messages"])
                if not isinstance(messages, list) or not messages:
                    raise ValueError
                for message in messages:
                    if message.get("role") == "assistant":
                        message["content"] = message.get("content", "").lstrip()
                if any(m.get("tool_calls") for m in messages):
                    raise ValueError
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                stats["rejections"]["replay_invalid"] += 1
                continue
            key = canonical_key(messages)
            if key in seen:
                stats["rejections"]["duplicate"] += 1
                continue
            ok, tokens, reason = check_and_measure(tok, messages, None, max_tokens)
            if not ok:
                stats["rejections"][reason] += 1
                continue
            eligible += 1
            item = {"row": {"messages": json.dumps(messages, ensure_ascii=False),
                            "tools": "", "source": raw.get("source") or "replay",
                            "domain": raw.get("domain") or ""},
                    "category": REPLAY, "source_line": line_number,
                    "language": "unknown", "function": "", "depth": 0,
                    "tool_count": 0, "grounding": {}, "error_types": {},
                    "tokens": tokens, "hash": _digest(key)}
            pair = (key, item)
            if len(reservoir) < limit:
                reservoir.append(pair)
            else:
                index = rng.randrange(eligible)
                if index < limit:
                    reservoir[index] = pair
    for key, _ in reservoir:
        seen.add(key)
    return [item for _, item in reservoir]


def atomic_write(rows, stats, output, stats_path):
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fd, csv_tmp = tempfile.mkstemp(dir=output.parent, prefix="." + output.name, suffix=".tmp")
    json_fd, json_tmp = tempfile.mkstemp(dir=stats_path.parent, prefix="." + stats_path.name, suffix=".tmp")
    backup = None
    csv_replaced = False
    try:
        with os.fdopen(csv_fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["messages", "tools", "source", "domain"])
            writer.writeheader()
            writer.writerows(item["row"] for item in rows)
            stream.flush(); os.fsync(stream.fileno())
        with os.fdopen(json_fd, "w", encoding="utf-8") as stream:
            json.dump(stats, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if output.exists():
            backup_fd, backup = tempfile.mkstemp(
                dir=output.parent, prefix="." + output.name, suffix=".bak")
            os.close(backup_fd)
            shutil.copy2(output, backup)
        os.replace(csv_tmp, output)
        csv_replaced = True
        os.replace(json_tmp, stats_path)
    except BaseException:
        if csv_replaced:
            if backup:
                shutil.copy2(backup, output)
            elif output.exists():
                output.unlink()
        raise
    finally:
        for path in (csv_tmp, json_tmp, backup):
            if path and os.path.exists(path):
                os.unlink(path)


def build(args, tok=None):
    input_path, replay_path = Path(args.input), Path(args.replay_csv)
    if not input_path.is_file() or not replay_path.is_file():
        raise SystemExit("input and replay CSV must exist")
    tok = tok or AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    limits = {category: getattr(args, category + "_limit") for category in CATEGORIES}
    stats = {"config": {"seed": args.seed, "max_tools": args.max_tools,
                        "max_tokens": args.max_tokens, "limits": limits,
                        "function_cap_fraction": args.function_cap_fraction},
             "source_rows": Counter(sft_t2t=0, replay=0), "classified": Counter(),
             "selected": Counter(), "languages": Counter(), "functions": Counter(),
             "call_depth": Counter(), "tool_count": Counter(), "grounding": Counter(),
             "error_types": Counter(), "rejections": Counter(),
             "derivation_rejections": Counter(), "rows": []}
    candidates = {category: [] for category in CATEGORIES[:-1]}
    seen = set()
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stats["source_rows"]["sft_t2t"] += 1
            if args.progress_every and line_number % args.progress_every == 0:
                print(f"... {line_number} source rows", file=sys.stderr, flush=True)
            if EXA_RE.search(line):
                stats["rejections"]["exa_eval"] += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["rejections"]["bad_json"] += 1
                continue
            item, reason = classify_candidate(record, args.max_tools)
            if reason:
                stats["rejections"][reason] += 1
                continue
            ok, tokens, reason = check_and_measure(tok, item["messages"], item["tools"], args.max_tokens)
            if not ok:
                stats["rejections"][reason] += 1
                continue
            key = canonical_key(item["messages"], item["tools"])
            if key in seen:
                stats["rejections"]["duplicate"] += 1
                continue
            seen.add(key)
            source_digest = _digest(key)
            item.update(source_line=line_number, tokens=tokens, hash=source_digest,
                        source_digest=source_digest, derivation="native")
            candidates[item["category"]].append(item)
            stats["classified"][item["category"]] += 1
            if item["category"] == CALL_REQUIRED:
                for category in DERIVED_CATEGORIES:
                    derived, derivation_reason = derive_candidate(item, category)
                    if derivation_reason:
                        stats["derivation_rejections"][derivation_reason] += 1
                        break
                    ok, derived_tokens, gate_reason = check_and_measure(
                        tok, derived["messages"], derived["tools"], args.max_tokens)
                    if not ok:
                        stats["derivation_rejections"][gate_reason] += 1
                        continue
                    derived_key = canonical_key(derived["messages"], derived["tools"])
                    if derived_key in seen:
                        stats["derivation_rejections"]["duplicate"] += 1
                        continue
                    seen.add(derived_key)
                    derived.update(tokens=derived_tokens, hash=_digest(derived_key))
                    candidates[category].append(derived)
                    stats["classified"][category] += 1
    selected, function_counts = select_categories(
        candidates, limits, args.seed, args.function_cap_fraction)
    rows = [item for category in CATEGORIES[:-1] for item in selected[category]]
    rows += replay_rows(replay_path, limits[REPLAY], args.seed + 6, seen, tok,
                        args.max_tokens, stats)
    for category in CATEGORIES:
        actual = sum(item["category"] == category for item in rows)
        stats["selected"][category] = actual
        if actual < limits[category]:
            stats["rejections"]["quota_shortfall_" + category] = limits[category] - actual
    for item in rows:
        stats["languages"][item["language"]] += 1
        if item["function"]:
            stats["functions"][item["function"]] += 1
        stats["call_depth"][str(item["depth"])] += 1
        stats["tool_count"][str(item["tool_count"])] += 1
        stats["grounding"].update(item["grounding"])
        stats["error_types"].update(item["error_types"])
        stats["rows"].append({key: item.get(key, "") for key in (
            "source_line", "source_digest", "derivation", "category", "language",
            "function", "depth", "tool_count", "grounding", "error_types",
            "tokens", "hash")} | {"rejection_reason": ""})
    stats["function_cap"] = {
        category: {"count": max(1, int(limits[category] * args.function_cap_fraction)),
                   "selected_counts": function_counts[category]}
        for category in (CALL_REQUIRED,) + DERIVED_CATEGORIES}
    stats["total"], stats["target_total"] = len(rows), sum(limits.values())
    atomic_write(rows, stats, Path(args.output), Path(args.stats))
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--replay_csv", default=DEFAULT_REPLAY)
    parser.add_argument("--output", default=str(DEFAULT_DIR / "sft_t2t_repair_v2.csv"))
    parser.add_argument("--stats", default=str(DEFAULT_DIR / "stats.json"))
    parser.add_argument("--tokenizer", default=str(REPO / "qwen3_tokenizer"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_tools", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--function_cap_fraction", type=float, default=.08)
    parser.add_argument("--progress_every", type=int, default=200000)
    for category, limit in DEFAULT_LIMITS.items():
        parser.add_argument("--" + category + "_limit", type=int, default=limit)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stats = build(args)
    print(json.dumps({"output": args.output, "total": stats["total"],
                      "selected": stats["selected"]}, ensure_ascii=False, sort_keys=True))
    return stats


if __name__ == "__main__":
    main()
