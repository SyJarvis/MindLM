"""Break held-out NLL down by data source and by position type (document EOS vs body).

Plain held-out PPL is a single number; when it plateaus it cannot say *which* part
of the mixture is holding it up. The V3 corpus mixes Chinese web text
(``minimind``) with English math/code (``open-perfectblend``), which have very
different entropy, over layers of packed 4097-token records.

This script recomputes per-token NLL over the whole held-out bin, then attributes
every token to its document using the split's ``documents.jsonl`` index
(``stream_offset`` + ``source`` + ``source_tokens``). The bin is the split stream
minus the discarded tail, in order, so flat bin position maps 1:1 onto stream
position.

Reports per source: NLL on document bodies, NLL on the appended EOS token (how
well the model predicts document boundaries), and token share.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from eval_common import default_tokenizer_path, inference_context, load_checkpoint_with_tokenizer_check, supported_configs
from modeling_mindlm import MindLM
from training_utils import build_model_config


class HeldoutPackedDataset(Dataset):
    def __init__(self, prefix, max_length, num_records):
        self.tokens = np.memmap(Path(prefix).with_suffix(".bin"), dtype=np.uint32, mode="r",
                                shape=(num_records, max_length + 1))
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        record = self.tokens[index]
        input_ids = torch.from_numpy(np.asarray(record[:-1], dtype=np.int64).copy())
        targets = torch.from_numpy(np.asarray(record[1:], dtype=np.int64).copy())
        return input_ids, targets


def load_document_spans(documents_path, split="heldout"):
    """Return [(start, end, source)] over the flat stream, plus token totals."""
    spans = []
    with open(documents_path, encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            if doc.get("split") != split:
                continue
            start = doc["stream_offset"]
            length = doc["source_tokens"] + (1 if doc.get("eos_appended") else 0)
            spans.append((start, start + length, doc["source"]))
    spans.sort()
    return spans


def main():
    parser = argparse.ArgumentParser(description="Per-source held-out NLL breakdown")
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.2b_gdn")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--packed_prefix", required=True)
    parser.add_argument("--documents", required=True, help="documents.jsonl from the same data build")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--nll_out", default=None, help="optional .npy path for the raw per-token NLL")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(Path(args.packed_prefix).with_suffix(".json"), encoding="utf-8") as handle:
        bin_meta = json.load(handle)
    max_length = bin_meta["sequence_length"]
    num_records = bin_meta["num_sequences"]

    tokenizer = AutoTokenizer.from_pretrained(default_tokenizer_path(args.tokenizer_path), trust_remote_code=True)
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    metadata = load_checkpoint_with_tokenizer_check(model, args.checkpoint, args.device, tokenizer)
    model.eval()
    print(f"config={args.config} step={metadata.get('step') if isinstance(metadata, dict) else None}")

    dataset = HeldoutPackedDataset(args.packed_prefix, max_length, num_records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # nll[p] is the loss of predicting the token at flat stream position p.
    # Record r covers positions [r*(max_length+1), r*(max_length+1)+max_length]; its
    # first token is only ever an input, so it keeps no NLL entry.
    total_positions = num_records * (max_length + 1)
    nll = np.full(total_positions, np.nan, dtype=np.float64)

    cursor = 0
    with inference_context(args.device):
        for input_ids, targets in loader:
            input_ids = input_ids.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            logits = model(input_ids=input_ids).logits
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            batch_nll = torch.empty(flat_targets.numel(), dtype=torch.float64, device=args.device)
            for start in range(0, flat_targets.numel(), 8192):
                stop = min(start + 8192, flat_targets.numel())
                chunk = flat_logits[start:stop].float()
                log_probs = torch.log_softmax(chunk, dim=-1)
                batch_nll[start:stop] = -log_probs.gather(
                    1, flat_targets[start:stop, None]
                ).squeeze(1)
                del chunk, log_probs
            values = batch_nll.cpu().numpy()
            rows = input_ids.size(0)
            for row in range(rows):
                base = (cursor + row) * (max_length + 1) + 1
                nll[base:base + max_length] = values[row * max_length:(row + 1) * max_length]
            cursor += rows

    if args.nll_out:
        np.save(args.nll_out, nll)

    spans = load_document_spans(args.documents)
    print(f"held-out documents: {len(spans)}, stream tokens: {spans[-1][1]}")

    stats = defaultdict(lambda: {"body_sum": 0.0, "body_n": 0, "eos_sum": 0.0, "eos_n": 0, "docs": 0})
    uncovered = 0
    for start, end, source in spans:
        end = min(end, total_positions)
        entry = stats[source]
        entry["docs"] += 1
        body = nll[start:end - 1]
        eos = nll[end - 1] if end - 1 < total_positions else np.nan
        valid = ~np.isnan(body)
        uncovered += int(np.isnan(body).sum())
        entry["body_sum"] += float(body[valid].sum())
        entry["body_n"] += int(valid.sum())
        if not math.isnan(eos):
            entry["eos_sum"] += float(eos)
            entry["eos_n"] += 1

    all_body_sum = sum(e["body_sum"] for e in stats.values())
    all_body_n = sum(e["body_n"] for e in stats.values())
    print(f"\noverall body NLL {all_body_sum / all_body_n:.4f}  ppl {math.exp(all_body_sum / all_body_n):.2f}"
          f"  ({all_body_n:,} tokens)")
    print(f"{'source':<20}{'docs':>8}{'tok_share':>11}{'body_NLL':>10}{'body_ppl':>10}{'EOS_NLL':>9}")
    for source, entry in sorted(stats.items(), key=lambda kv: -kv[1]["body_n"]):
        body_nll = entry["body_sum"] / entry["body_n"]
        eos_nll = entry["eos_sum"] / entry["eos_n"] if entry["eos_n"] else float("nan")
        print(f"{source:<20}{entry['docs']:>8}{entry['body_n'] / all_body_n:>10.1%}"
              f"{body_nll:>10.4f}{math.exp(body_nll):>10.2f}{eos_nll:>9.4f}")

    # Blended NLL if every source were as predictable as the best one.
    best = min(e["body_sum"] / e["body_n"] for e in stats.values())
    print(f"\nif all sources matched the most predictable one ({best:.4f}), overall body NLL would be {best:.4f}")
    print(f"total body NLL contribution gap vs best-per-source: "
          f"{all_body_sum / all_body_n - best:+.4f}")
    print(f"positions without NLL (record-first tokens / truncated tail): {uncovered:,}")


if __name__ == "__main__":
    main()
