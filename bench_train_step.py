"""Measure repeatable MindLM training-step throughput before long runs."""

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modeling_mindlm import MindLM
from training_utils import build_model_config, masked_language_model_loss


def parse_args():
    parser = argparse.ArgumentParser(description="MindLM synthetic train-step benchmark")
    parser.add_argument("--model_config", choices=("mindlm_0.1b", "mindlm_0.1b_moe", "mindlm_0.7b"), default="mindlm_0.7b")
    parser.add_argument("--tokenizer_path", default="qwen3_tokenizer")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Write a Chrome trace for one measured step")
    parser.add_argument("--trace_dir", default="out/profile")
    args = parser.parse_args()
    if args.batch_size < 1 or args.seq_len < 2 or args.warmup < 0 or args.steps < 1:
        raise ValueError("batch_size, seq_len, and steps must be positive; warmup cannot be negative")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("bench_train_step.py requires a CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    config = build_model_config(args.model_config, tokenizer)
    if args.seq_len > config.max_seq_len:
        raise ValueError(f"seq_len={args.seq_len} exceeds configured max_seq_len={config.max_seq_len}")

    device = "cuda"
    model = MindLM(config).to(device).train()
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    autocast_context = (
        nullcontext()
        if args.dtype == "float32"
        else torch.autocast(device_type="cuda", dtype=getattr(torch, args.dtype))
    )
    input_ids = torch.randint(0, len(tokenizer), (args.batch_size, args.seq_len), device=device)
    targets = torch.randint(0, len(tokenizer), (args.batch_size, args.seq_len), device=device)
    loss_mask = torch.ones_like(targets)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_context:
            outputs = model(input_ids=input_ids)
            loss = masked_language_model_loss(outputs.logits, targets, loss_mask, outputs.aux_loss)
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(args.warmup):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    if args.profile:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            train_step()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(f"{args.trace_dir}/mindlm_{args.model_config}_trace.json")

    start = time.perf_counter()
    for _ in range(args.steps):
        loss = train_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = args.steps * args.batch_size * args.seq_len
    print(f"loss={loss.item():.4f}")
    print(f"step_ms={elapsed / args.steps * 1000:.2f}")
    print(f"tokens_per_second={tokens / elapsed:.0f}")
    print(f"peak_memory_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}")


if __name__ == "__main__":
    main()
