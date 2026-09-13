"""Generate a compact Markdown report from tool-call evaluation JSON.

The historical root-level ``gen_tool_call_report.py`` remains as a command-line
compatibility shim.  This module contains the reusable implementation so report
generation belongs with the other evaluation utilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(data: dict) -> str:
    """Render the common result shape emitted by ``eval_sft_tool.py``."""
    results = data.get("results", [])
    passed = sum(bool(row.get("pass")) for row in results)
    lines = ["# Tool-call evaluation", "", f"Passed: {passed}/{len(results)}", "", "| Case | Pass | Calls | Name |", "|---|---:|---:|---|"]
    for row in results:
        calls = row.get("n_calls", 0)
        name = row.get("expect_name") or ""
        lines.append(f"| {row.get('id', '')} | {'yes' if row.get('pass') else 'no'} | {calls} | {name} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a Markdown tool-call report")
    parser.add_argument("input", nargs="?", help="evaluation JSON file")
    parser.add_argument("--output", help="Markdown output path")
    args = parser.parse_args(argv)
    source = Path(args.input) if args.input else Path(__file__).resolve().parent.parent / "runs/sft_v3_20260913_eval/prompts_and_raw.json"
    output = Path(args.output) if args.output else (Path(__file__).resolve().parent.parent / "docs/tool_call_eval_step15200_20260913.md" if not args.input else source.with_suffix(".md"))
    output.write_text(build_report(json.loads(source.read_text(encoding="utf-8"))), encoding="utf-8")
    print("wrote", output)


if __name__ == "__main__":
    main()
