"""Audit UltraData tool schemas: per-row tool counts, function frequencies, pool coverage.

Answers the two questions needed to size the E2/E3 tool limits:

  A. How many schemas does a typical row carry?  -> ``--max_tools_per_row``
  B. How many distinct functions do we need to cover most rows?
     -> ``--tool_pool`` whitelist size K

Runs against the UltraData jsonl tree on the training server:

  python3 scripts/audit_tool_families.py \
      --ultra /mnt/data2/reexen_datasets/llm_training/sft/UltraData-SFT-Agent-2609/data \
      [--limit 50000] [--dump_pool 50 --pool_out /tmp/tool_pool.json]

Also verifies that every function called by an assistant message exists in the
row's own ``tools`` list (a prerequisite for called-first capping) and that the
eval-probe function names actually occur in the corpus.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

EVAL_PROBE_NAMES = [
    # eval/eval_sft_tool.py RETAIL_TOOLS
    "find_user_id_by_email", "get_order_details", "get_product_details",
    "cancel_pending_order", "modify_pending_order_items",
    # eval/eval_sft_tool.py GENERIC_TOOLS
    "calculate", "get_reservation_details",
]


def called_functions(record):
    names = set()
    for m in record.get("messages", []):
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name"):
                names.add(fn["name"])
    return names


def tool_names(record):
    names = []
    for tool in record.get("tools") or []:
        fn = tool.get("function") or {}
        if fn.get("name"):
            names.append(fn["name"])
    return names


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    return sorted_values[min(int(len(sorted_values) * q), len(sorted_values) - 1)]


def analyse(paths, limit, verbose):
    rows = 0
    rows_with_tools = 0
    rows_with_calls = 0
    called_missing_from_tools = 0
    counts = []
    name_freq = Counter()
    domain_freq = Counter()
    subset_freq = Counter()
    # For every row, the set of called names; later reused for K-coverage.
    row_calls = []

    for path in paths:
        subset = Path(path).parent.name
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if limit and rows >= limit:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows += 1
                subset_freq[subset] += 1
                domain_freq[record.get("domain", "?")] += 1
                names = tool_names(record)
                if names:
                    rows_with_tools += 1
                    counts.append(len(names))
                    name_freq.update(set(names))
                calls = called_functions(record)
                if calls:
                    rows_with_calls += 1
                    row_calls.append((calls, set(names)))
                    if not calls <= set(names):
                        called_missing_from_tools += 1
                if verbose and rows % 100000 == 0:
                    print(f"  ... {rows} rows", flush=True)
        if limit and rows >= limit:
            break

    counts.sort()
    stats = {
        "rows": rows,
        "rows_with_tools": rows_with_tools,
        "rows_with_tool_calls": rows_with_calls,
        "called_missing_from_tools_rows": called_missing_from_tools,
        "tools_per_row": {
            "min": counts[0] if counts else 0,
            "p50": percentile(counts, 0.50),
            "p90": percentile(counts, 0.90),
            "p99": percentile(counts, 0.99),
            "max": counts[-1] if counts else 0,
            "mean": round(sum(counts) / len(counts), 2) if counts else 0,
        },
        "distinct_functions": len(name_freq),
        "top_functions": name_freq.most_common(50),
        "top_domains": domain_freq.most_common(20),
        "rows_per_subset": subset_freq.most_common(),
    }

    # K-coverage: share of tool-calling rows fully covered by the top-K names.
    coverage = {}
    for k in (10, 25, 50, 100, 200, 500):
        top = {name for name, _ in name_freq.most_common(k)}
        covered = sum(1 for calls, _ in row_calls if calls <= top)
        coverage[f"top_{k}"] = round(covered / len(row_calls), 4) if row_calls else None
    stats["pool_coverage_of_calling_rows"] = coverage
    stats["eval_probe_occurrences"] = {name: name_freq.get(name, 0) for name in EVAL_PROBE_NAMES}
    return stats, name_freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ultra", required=True, help="UltraData data/ root")
    parser.add_argument("--subsets", default="Tool_Use", help="comma list of subset dirs; 'all' scans everything")
    parser.add_argument("--limit", type=int, default=0, help="stop after N rows (0 = full scan)")
    parser.add_argument("--dump_pool", type=int, default=0, help="also write the top-K function names")
    parser.add_argument("--pool_out", default="tool_pool.json")
    parser.add_argument("--json_out", default="tool_audit.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.ultra)
    if args.subsets == "all":
        paths = sorted(root.glob("*/*.jsonl"))
    else:
        paths = []
        for subset in args.subsets.split(","):
            paths.extend(sorted((root / subset.strip()).glob("*.jsonl")))
    if not paths:
        raise SystemExit(f"no jsonl found under {root} (subsets={args.subsets})")

    print(f"scanning {len(paths)} files from {root}")
    stats, name_freq = analyse(paths, args.limit, args.verbose)

    Path(args.json_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.json_out}")

    if args.dump_pool:
        pool = [name for name, _ in name_freq.most_common(args.dump_pool)]
        Path(args.pool_out).write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
        covered = stats["pool_coverage_of_calling_rows"].get(f"top_{args.dump_pool}")
        print(f"wrote {args.pool_out} ({len(pool)} names, calling-row coverage ≈ {covered})")


if __name__ == "__main__":
    main()
