"""Compute held-out perplexity for a MindLM pretraining checkpoint."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import default_tokenizer_path, inference_context, load_checkpoint_with_tokenizer_check, supported_configs
from modeling_mindlm import MindLM
from training_utils import build_model_config


class HeldoutPackedDataset(Dataset):
    """Memory-map the leading N records of a packed .bin as held-out data."""

    def __init__(self, prefix, max_length, num_records):
        prefix = Path(prefix)
        self.tokens = np.memmap(
            prefix.with_suffix(".bin"),
            dtype=np.uint32,
            mode="r",
            shape=(num_records[1], max_length + 1),
        )[: num_records[0]]
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        record = self.tokens[index]
        input_ids = torch.from_numpy(np.asarray(record[:-1], dtype=np.int64).copy())
        targets = torch.from_numpy(np.asarray(record[1:], dtype=np.int64).copy())
        loss_mask = torch.ones(self.max_length, dtype=torch.int64)
        return input_ids, targets, loss_mask


def main():
    parser = argparse.ArgumentParser(description="MindLM pretraining held-out PPL")
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.2b_gdn")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--packed_prefix", required=True, help="Packed held-out <prefix>.bin/.json")
    parser.add_argument("--num_records", type=int, default=256, help="Number of leading packed records to evaluate (0 = all)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json_out", default=None, help="write the result summary as JSON to this path")
    args = parser.parse_args()

    with open(Path(args.packed_prefix).with_suffix(".json"), encoding="utf-8") as file:
        bin_metadata = json.load(file)
    total_records = bin_metadata["num_sequences"]
    num_records = args.num_records if args.num_records > 0 else total_records
    if num_records > total_records:
        raise SystemExit(f"--num_records={num_records} exceeds the {total_records} records in the held-out bin")

    dataset = HeldoutPackedDataset(args.packed_prefix, bin_metadata["sequence_length"], (num_records, total_records))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    tokenizer = AutoTokenizer.from_pretrained(
        default_tokenizer_path(args.tokenizer_path), trust_remote_code=True
    )
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    checkpoint_metadata = load_checkpoint_with_tokenizer_check(model, args.checkpoint, args.device, tokenizer)
    model.eval()

    total_nll, total_tokens = 0.0, 0
    with inference_context(args.device):
        for input_ids, targets, loss_mask in loader:
            input_ids = input_ids.to(args.device)
            targets = targets.to(args.device)
            loss_mask = loss_mask.to(args.device)
            logits = model(input_ids=input_ids).logits
            log_probs = torch.log_softmax(logits.reshape(-1, logits.size(-1)).float(), dim=-1)
            token_nll = -log_probs.gather(1, targets.reshape(-1, 1)).squeeze(1)
            mask = loss_mask.reshape(-1).to(token_nll.dtype)
            total_nll += (token_nll * mask).sum().item()
            total_tokens += mask.sum().item()

    mean_nll = total_nll / total_tokens
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "step": checkpoint_metadata.get("step") if isinstance(checkpoint_metadata, dict) else None,
        "heldout_prefix": args.packed_prefix,
        "records": num_records,
        "tokens": int(total_tokens),
        "mean_nll": round(mean_nll, 6),
        "ppl": round(math.exp(mean_nll), 4),
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
