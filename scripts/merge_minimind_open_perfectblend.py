#!/usr/bin/env python3
"""Merge MinIMind and Open-PerfectBlend text into one CSV."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEFAULT_MINIMIND_INPUT = BASE / "data" / "pretrain_data.csv"
DEFAULT_OPEN_PERFECTBLEND_INPUT = BASE / "data" / "perfectblend_mathcode.csv"
DEFAULT_OUTPUT = BASE / "data" / "pretrain_minimind_open_perfectblend.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two text CSV sources into a text,source CSV."
    )
    parser.add_argument("--minimind-input", type=Path, default=DEFAULT_MINIMIND_INPUT)
    parser.add_argument(
        "--open-perfectblend-input",
        type=Path,
        default=DEFAULT_OPEN_PERFECTBLEND_INPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument(
        "--limit-per-source",
        type=int,
        help="Read at most this many input documents from each source.",
    )
    parser.add_argument(
        "--no-dedup-minimind",
        action="store_true",
        help="Keep all non-empty MinIMind documents instead of prefix deduplication.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit_per_source is not None and args.limit_per_source < 1:
        parser.error("--limit-per-source must be positive")
    return args


def iter_text(path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"CSV is missing required column 'text': {path}")
        for row in reader:
            yield row.get("text") or ""


def prefix_key(text, prefix_length=150):
    return hashlib.md5(text[:prefix_length].encode("utf-8", "ignore")).digest()


def make_source_stats(source, input_path):
    return {
        "source": source,
        "input": str(input_path.resolve()),
        "rows_read": 0,
        "rows_written": 0,
        "empty_rows": 0,
        "deduplicated_rows": 0,
        "characters_written": 0,
    }


def write_source_rows(writer, path, source, limit, deduplicate):
    stats = make_source_stats(source, path)
    seen_prefixes = set()
    for text in iter_text(path):
        if limit is not None and stats["rows_read"] >= limit:
            break
        stats["rows_read"] += 1
        text = str(text)
        if not text.strip():
            stats["empty_rows"] += 1
            continue
        if deduplicate:
            key = prefix_key(text)
            if key in seen_prefixes:
                stats["deduplicated_rows"] += 1
                continue
            seen_prefixes.add(key)
        writer.writerow([text, source])
        stats["rows_written"] += 1
        stats["characters_written"] += len(text)
        if stats["rows_read"] % 1_000_000 == 0:
            print(
                f"[{source}] read {stats['rows_read']:,}, "
                f"written {stats['rows_written']:,}",
                flush=True,
            )
    return stats


def main():
    args = parse_args()
    input_paths = [args.minimind_input.resolve(), args.open_perfectblend_input.resolve()]
    output_path = args.output.resolve()
    stats_path = (args.stats_output or output_path.with_suffix(".json")).resolve()
    if output_path in input_paths:
        raise ValueError("Output path must differ from both input paths")
    if stats_path in input_paths or stats_path == output_path:
        raise ValueError("Stats path must differ from both input paths and the output path")
    if not all(path.is_file() for path in input_paths):
        missing = [str(path) for path in input_paths if not path.is_file()]
        raise FileNotFoundError(", ".join(missing))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    if stats_path.exists() and not args.overwrite:
        raise FileExistsError(f"{stats_path} already exists; pass --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    source_stats = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(
                temporary_file,
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            writer.writerow(["text", "source"])
            source_stats.append(
                write_source_rows(
                    writer,
                    input_paths[0],
                    "minimind",
                    args.limit_per_source,
                    not args.no_dedup_minimind,
                )
            )
            source_stats.append(
                write_source_rows(
                    writer,
                    input_paths[1],
                    "open-perfectblend",
                    args.limit_per_source,
                    False,
                )
            )
        os.replace(temporary_path, output_path)
        temporary_path = None

        manifest = {
            "format": "mindlm_text_source_v1",
            "columns": ["text", "source"],
            "deduplication": {
                "minimind": "md5 of first 150 characters"
                if not args.no_dedup_minimind
                else None,
            },
            "limit_per_source": args.limit_per_source,
            "sources": source_stats,
            "total_rows_written": sum(item["rows_written"] for item in source_stats),
            "total_characters_written": sum(
                item["characters_written"] for item in source_stats
            ),
        }
        with stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(manifest, stats_file, ensure_ascii=False, indent=2)
            stats_file.write("\n")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"saved -> {output_path}")
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
