#!/usr/bin/env python3
"""Unified MindLM data preparation entry point."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("pretrain", "sft"), required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.type == "pretrain":
        from data_process.pretrain import prepare, parse_args
        prepare(parse_args())
    else:
        from data_process.sft import main as prepare_sft
        prepare_sft()


if __name__ == "__main__":
    main()
