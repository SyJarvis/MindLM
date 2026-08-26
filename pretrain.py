"""Pretrain one of the supported MindLM configurations."""

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from dataset import PackedPretrainDataset, PretrainDataset
from modeling_mindlm import MindLM
from training_utils import (
    build_model_config,
    EpochRandomSampler,
    load_model_checkpoint,
    masked_language_model_loss,
    save_training_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent


def is_primary_process():
    return not ddp or dist.get_rank() == 0


def log(message):
    if is_primary_process():
        print(message)


def get_lr(step, total_steps):
    if args.warmup_iters > 0 and step < args.warmup_iters:
        return args.learning_rate * step / args.warmup_iters

    decay_steps = max(total_steps - args.warmup_iters, 1)
    progress = min(max((step - args.warmup_iters) / decay_steps, 0.0), 1.0)
    min_lr = args.learning_rate / 10
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (args.learning_rate - min_lr)


def optimizer_step(gradient_scale=1.0):
    scaler.unscale_(optimizer)
    if gradient_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def checkpoint_path(epoch=None):
    suffix = "latest" if epoch is None else f"epoch{epoch}"
    return Path(args.out_dir) / f"mindlm_pretrain_{args.model_config}_{suffix}.pt"


def save_checkpoint(epoch, step, epoch_complete):
    model.eval()
    path = checkpoint_path(None if not epoch_complete else epoch)
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        config,
        epoch,
        step,
        epoch_complete,
        training_stage="pretrain",
    )
    log(f"Saved checkpoint: {path}")
    model.train()


def train_epoch(epoch, wandb, skip_steps=0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = None
    processed_batches = 0
    processed_tokens = 0
    pending_steps = 0

    for step, (input_ids, targets, loss_mask) in enumerate(train_loader):
        if step < skip_steps:
            continue
        if start_time is None:
            start_time = time.time()
        input_ids = input_ids.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)

        global_step = epoch * iter_per_epoch + step
        lr = get_lr(global_step, args.epochs * iter_per_epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        is_update_step = (
            pending_steps + 1 == args.accumulation_steps
            or step + 1 == iter_per_epoch
        )
        sync_context = (
            model.no_sync()
            if ddp and not is_update_step
            else nullcontext()
        )
        with sync_context:
            with ctx:
                outputs = model(input_ids=input_ids)
                unscaled_loss = masked_language_model_loss(
                    outputs.logits,
                    targets,
                    loss_mask,
                    outputs.aux_loss,
                )
                loss = unscaled_loss / args.accumulation_steps
            scaler.scale(loss).backward()
        pending_steps += 1

        is_update_step = pending_steps == args.accumulation_steps
        is_last_step = step + 1 == iter_per_epoch
        if is_update_step or is_last_step:
            # The final partial accumulation window has fewer contributions.
            optimizer_step(args.accumulation_steps / pending_steps)
            pending_steps = 0

        processed_batches += 1
        processed_tokens += input_ids.numel() * (dist.get_world_size() if ddp else 1)
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            remaining_minutes = elapsed / processed_batches * (iter_per_epoch - step - 1) / 60
            tokens_per_second = processed_tokens / elapsed
            log(
                f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{iter_per_epoch} "
                f"loss={unscaled_loss.item():.4f} lr={lr:.7f} "
                f"tok/s={tokens_per_second:.0f} eta={remaining_minutes:.1f}m"
            )
            if wandb is not None and is_primary_process():
                wandb.log({
                    "loss": unscaled_loss.item(),
                    "lr": lr,
                    "tokens_per_second": tokens_per_second,
                    "epoch": epoch,
                    "step": global_step,
                })

        if (
            args.save_interval > 0
            and (step + 1) % args.save_interval == 0
            and pending_steps == 0
            and is_primary_process()
        ):
            save_checkpoint(epoch, step, epoch_complete=False)


def init_distributed_mode():
    if not ddp:
        return args.device

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training currently requires CUDA/NCCL.")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="MindLM pretraining")
    parser.add_argument("--out_dir", default="out", help="Checkpoint directory")
    parser.add_argument("--data_path", default="data/pretrain_data.csv", help="CSV containing a text column")
    parser.add_argument(
        "--packed_data_prefix",
        default=None,
        help="Prefix of <prefix>.bin/.json from prepare_pretrain_data.py; overrides --data_path",
    )
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--model_config", choices=("mindlm_0.1b", "mindlm_0.1b_moe", "mindlm_0.7b"), default="mindlm_0.1b")
    parser.add_argument("--resume_from", default=None, help="Legacy state dict or MindLM checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ddp", action="store_true", help="Expected when launched through torchrun")
    parser.add_argument("--ddp_bucket_cap_mb", type=int, default=100)
    parser.add_argument("--ddp_static_graph", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MindLM-Pretrain")
    parser.add_argument("--wandb_run_name", default=None)
    parsed = parser.parse_args()
    if parsed.tokenizer_path is None:
        parsed.tokenizer_path = str(
            REPOSITORY_ROOT / ("qwen3_tokenizer" if parsed.model_config == "mindlm_0.7b" else "mindlm_tokenizer")
        )
    if parsed.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if parsed.ddp_bucket_cap_mb < 1:
        raise ValueError("ddp_bucket_cap_mb must be positive")
    return parsed


if __name__ == "__main__":
    args = parse_args()
    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")

    ddp = "RANK" in os.environ
    if args.ddp and not ddp:
        raise ValueError("--ddp requires launching with torchrun")
    args.device = init_distributed_mode()
    device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    ctx = nullcontext() if device_type == "cpu" or args.dtype == "float32" else torch.autocast(device_type=device_type, dtype=getattr(torch, args.dtype))

    torch.manual_seed(1337)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    config = build_model_config(args.model_config, tokenizer)
    model = MindLM(config).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler(device_type, enabled=args.dtype == "float16" and device_type == "cuda")

    resume_metadata = {}
    if args.resume_from:
        resume_metadata = load_model_checkpoint(model, args.resume_from, args.device)
        if resume_metadata.get("training_stage") not in (None, "pretrain"):
            raise ValueError("Pretraining can only resume a pretraining checkpoint.")
        if resume_metadata.get("optimizer"):
            optimizer.load_state_dict(resume_metadata["optimizer"])
        if resume_metadata.get("scaler"):
            scaler.load_state_dict(resume_metadata["scaler"])
        log(f"Loaded checkpoint: {args.resume_from}")

    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            # Routed MoE experts can be unused on an individual rank for a
            # batch. Dense configurations keep the faster default path.
            find_unused_parameters=config.use_moe,
            static_graph=args.ddp_static_graph and not config.use_moe,
        )

    if args.packed_data_prefix:
        train_dataset = PackedPretrainDataset(
            args.packed_data_prefix,
            max_length=config.max_seq_len,
            tokenizer_vocab_size=len(tokenizer),
        )
        log(f"Using packed pretraining data: {args.packed_data_prefix} ({len(train_dataset)} sequences)")
    else:
        dataframe = pd.read_csv(args.data_path)
        if "text" not in dataframe.columns:
            raise ValueError("Pretraining CSV must contain a 'text' column")
        train_dataset = PretrainDataset(dataframe, tokenizer, max_length=config.max_seq_len)
    train_sampler = DistributedSampler(train_dataset, seed=1337) if ddp else EpochRandomSampler(train_dataset)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "sampler": train_sampler,
        "num_workers": args.num_workers,
        "pin_memory": device_type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )
    iter_per_epoch = len(train_loader)
    if iter_per_epoch == 0:
        raise ValueError("Pretraining dataset is empty")

    wandb = None
    if args.use_wandb and is_primary_process():
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    start_epoch = 0
    resume_step = 0
    if resume_metadata:
        if resume_metadata.get("epoch_complete"):
            start_epoch = resume_metadata["epoch"] + 1
        else:
            start_epoch = resume_metadata.get("epoch", 0)
            resume_step = resume_metadata.get("step", -1) + 1
            log(f"Resuming epoch {start_epoch + 1} from batch {resume_step + 1}.")

    global_batch_size = args.batch_size * args.accumulation_steps * (dist.get_world_size() if ddp else 1)
    log(
        f"Training {args.model_config}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
        f"global_batch={global_batch_size}"
    )
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb, skip_steps=resume_step if epoch == start_epoch else 0)
        resume_step = 0
        if is_primary_process():
            save_checkpoint(epoch, iter_per_epoch - 1, epoch_complete=True)

    if wandb is not None:
        wandb.finish()
    if ddp:
        dist.destroy_process_group()
