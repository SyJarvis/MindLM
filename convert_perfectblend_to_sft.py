"""Convert the flattened perfectblend pretrain CSV into the SFT `history,q,a` schema.

Reads the ChatML-rendered conversations produced by convert_perfectblend_to_csv.py
and emits one row per final assistant turn:

- single-turn conversations (1 user + 1 assistant) -> ``history=[]``, ``q``/``a``
- multi-turn conversations -> earlier turns go to ``history`` as ``[[q1,a1],...]``,
  the last user/assistant pair becomes ``q``/``a``

Rows are dropped when they cannot form a supervised sample under SFTDataset:
- the last role is not ``assistant`` (truncated/noisy conversations)
- consecutive same-role turns (malformed alternation, only a handful exist)

Only ``user``/``assistant`` roles are kept; ``system`` turns (6 rows) are dropped
because SFTDataset builds messages from user/assistant pairs only.
"""

import argparse
import csv
from pathlib import Path

MARKER = "<|im_start|>"
ROLE_MAP = {"user": "user", "assistant": "assistant"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert perfectblend ChatML CSV to SFT history,q,a CSV")
    parser.add_argument("--input_csv", default="data/perfectblend_pretrain_clean.csv")
    parser.add_argument("--output_csv", default="data/perfectblend_sft.csv")
    parser.add_argument("--max_history_pairs", type=int, default=0,
                        help="0 = keep all history pairs; otherwise truncate to the last N")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_turns(text):
    """Split a ChatML-rendered conversation into (role, content) turns."""
    turns = []
    pos = 0
    while True:
        start = text.find(MARKER, pos)
        if start < 0:
            break
        body_start = start + len(MARKER)
        end = text.find("<|im_end|>", body_start)
        if end < 0:
            # No closing marker: the rest belongs to this (truncated) turn.
            body = text[body_start:]
            pos = len(text)
        else:
            body = text[body_start:end]
            pos = end + len("<|im_end|>")
        for role in ROLE_MAP:
            if body.startswith(role):
                content = body[len(role):].strip("\n")
                turns.append((role, content.strip()))
                break
        # Roles other than user/assistant (system) are skipped entirely.
    return turns


def main():
    args = parse_args()
    csv.field_size_limit(10 ** 9)
    output = Path(args.output_csv)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace")

    total = kept = 0
    drop_last_not_assistant = drop_bad_alternation = drop_too_few_turns = 0
    with open(args.input_csv, newline="", encoding="utf-8") as in_f, \
            output.open("w", encoding="utf-8", newline="") as out_f:
        reader = csv.reader(in_f)
        next(reader)  # header: text
        writer = csv.writer(out_f)
        writer.writerow(["history", "q", "a"])
        for row in reader:
            total += 1
            turns = parse_turns(row[0])
            if not turns or turns[-1][0] != "assistant":
                drop_last_not_assistant += 1
                continue
            if any(turns[i][0] == turns[i + 1][0] for i in range(len(turns) - 1)):
                drop_bad_alternation += 1
                continue
            if len(turns) < 2:
                drop_too_few_turns += 1
                continue
            history = [[turn[1], next_turn[1]] for turn, next_turn in zip(turns[:-2:2], turns[1:-1:2])]
            if len(turns) % 2 != 0:
                # Non-alternating residue (e.g. system-only rows): skip rather
                # than supervise a shifted pair.
                drop_bad_alternation += 1
                continue
            if args.max_history_pairs > 0:
                history = history[-args.max_history_pairs:]
            writer.writerow([repr(history), turns[-2][1], turns[-1][1]])
            kept += 1

    print(f"total={total} kept={kept} "
          f"dropped: last_not_assistant={drop_last_not_assistant} "
          f"bad_alternation={drop_bad_alternation} "
          f"too_few_turns={drop_too_few_turns}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
