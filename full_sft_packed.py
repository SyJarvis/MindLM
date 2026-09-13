"""Grouped-batch SFT training for MindLM.

Replaces the fixed-length padded SFT loop with token-bin inputs produced by
``prepare_sft_bins.py``. Sequences are grouped by similar length so batches
stay tight; per-batch padding is bounded by the longest sequence in the group
instead of a global max. Loss masking (assistant-only) is precomputed during
tokenization and stored in the bin metadata, so this loop only needs to slice
tokens and masks — no per-step re-rendering.

Run:
  python3 full_sft_packed.py \
    --bin data/sft_tokens_4096.bin --meta data/sft_tokens_4096.rows.jsonl \
    --model_config mindlm_0.1b \
    --resume_from out_pretrain_minimind_pb/mindlm_pretrain_mindlm_0.1b_epoch2.pt \
    --resume_weights_only \
    --token_budget 8192 --accumulation_steps 4 --epochs 2 --learning_rate 5e-5 \
    --use_wandb --wandb_run_name mindlm_0.1b_sft_grouped
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from modeling_mindlm import MindLM
from training_utils import build_model_config, load_model_checkpoint, masked_lm_head_loss, save_training_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parent


def is_primary_process():
    return True


def log(message):
    print(message, flush=True)


def get_lr(step, total_steps, warmup, base_lr):
    if warmup > 0 and step < warmup:
        return base_lr * step / warmup
    decay_steps = max(total_steps - warmup, 1)
    progress = min(max((step - warmup) / decay_steps, 0.0), 1.0)
    min_lr = base_lr / 10
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (base_lr - min_lr)


class GroupedSFTDataset(Dataset):
    """Random access over variable-length records stored in one token bin."""

    def __init__(self, bin_path, meta_path):
        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.records = []
        digest = hashlib.sha256()
        with open(meta_path, encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                digest.update(line.encode("utf-8"))
                record = json.loads(line)
                if not all(type(record.get(key)) is int for key in ("off", "n", "ans")):
                    raise ValueError(f"metadata line {line_number}: off, n and ans must be integers")
                off, n, ans = record["off"], record["n"], record["ans"]
                if off < 0 or n < 2 or off + n > len(self.tokens) or not 1 <= ans < n:
                    raise ValueError(f"metadata line {line_number}: invalid token bounds or answer length: {record}")
                self.records.append(record)
        if not self.records:
            raise ValueError("empty metadata file")
        stat = Path(bin_path).stat()
        self.fingerprint = {"metadata_sha256": digest.hexdigest(), "bin_size": stat.st_size, "bin_mtime_ns": stat.st_mtime_ns}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        seq = torch.from_numpy(self.tokens[rec["off"]:rec["off"] + rec["n"]].astype(np.int64))
        X = seq[:-1]
        Y = seq[1:]
        mask = torch.zeros_like(Y)
        # Y drops the first prompt token, so every answer token remains a target.
        mask[-rec["ans"]:] = 1
        return X, Y, mask


class LengthGroupedBatchSampler:
    """Yield lists of indices: length-sorted inside mega-batches, shuffled across batches."""

    def __init__(self, lengths, token_budget, mega_batch, max_count=1024, seed=1337):
        if token_budget < 1 or mega_batch < 1 or max_count < 1:
            raise ValueError("require positive token_budget, mega_batch and max_count")
        if any(n < 2 or n - 1 > token_budget for n in lengths):
            raise ValueError("every record must have 2 <= n <= token_budget + 1")
        self.lengths = lengths
        self.token_budget = token_budget
        self.max_count = max_count
        self.mega_batch = mega_batch
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0
        self.batches = self._build(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.start_batch = 0
        self.batches = self._build(epoch)

    def _build(self, epoch):
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        batches = []
        for start in range(0, len(order), self.mega_batch):
            chunk = sorted(order[start:start + self.mega_batch], key=lambda i: self.lengths[i])
            current, longest = [], 0
            for idx in chunk:
                candidate_longest = max(longest, self.lengths[idx] - 1)
                if current and ((len(current) + 1) * candidate_longest > self.token_budget or len(current) >= self.max_count):
                    batches.append(current)
                    current, longest = [], 0
                current.append(idx)
                longest = max(longest, self.lengths[idx] - 1)
            if current:
                batches.append(current)
        perm = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[i] for i in perm]

    def __iter__(self):
        yield from self.batches[self.start_batch:]

    def __len__(self):
        return len(self.batches) - self.start_batch


def collate_variable(batch):
    longest = max(x.size(0) for x, _, _ in batch)
    batch_size = len(batch)
    X = torch.zeros(batch_size, longest, dtype=torch.long)
    Y = torch.zeros(batch_size, longest, dtype=torch.long)
    M = torch.zeros(batch_size, longest, dtype=torch.long)
    for r, (x, y, m) in enumerate(batch):
        n = x.size(0)
        X[r, :n] = x
        Y[r, :n] = y
        M[r, :n] = m
    return X, Y, M


def parse_args():
    parser = argparse.ArgumentParser(description="Grouped-batch MindLM SFT")
    parser.add_argument("--bin", default="data/sft_tokens_4096.bin")
    parser.add_argument("--meta", default="data/sft_tokens_4096.rows.jsonl")
    parser.add_argument("--tokenizer_path", default="qwen3_tokenizer")
    parser.add_argument("--model_config", default="mindlm_0.1b")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--resume_from", required=True)
    parser.add_argument("--resume_weights_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None, help="deprecated and ignored; use --max_batch_count")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--mega_batch", type=int, default=512, help="sequences grouped per length-sorting pool")
    parser.add_argument("--token_budget", type=int, default=8192, help="max padded input tokens per micro-batch: B * (max(n) - 1)")
    parser.add_argument("--max_batch_count", type=int, default=1024, help="max sequences per micro-batch")
    parser.add_argument("--loss_chunk_tokens", "--loss_chunk", dest="loss_chunk_tokens", type=int, default=256, help="supervised tokens per checkpointed LM-head CE chunk")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gradient_checkpointing", choices=("off", "linear_attn", "all"), default=None, help="override the config's layer checkpointing policy")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000, help="optimizer steps between checkpoints; 0 disables")
    parser.add_argument("--limit_steps", type=int, default=0, help="stop at absolute optimizer update N without changing the LR schedule (0 = full epochs)")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MindLM-SFT")
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()
    for name in ("token_budget", "mega_batch", "max_batch_count", "accumulation_steps", "loss_chunk_tokens", "epochs", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if min(args.num_workers, args.warmup_iters, args.save_interval, args.limit_steps) < 0:
        parser.error("worker count, warmup, save interval and step limit cannot be negative")
    if args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        parser.error("learning rate and grad clip must be positive; weight decay cannot be negative")
    if args.batch_size is not None:
        log("Warning: --batch_size is deprecated and ignored; use --max_batch_count to cap sequences.")
    return args


def grouped_forward_loss(model, input_ids, targets, loss_mask, chunk_tokens=256, reduction="sum"):
    """Dense MindLM backbone followed by the checkpointed, supervised-only head."""
    if model.config.use_moe:
        raise ValueError("grouped SFT currently supports dense models only; MoE auxiliary loss weighting is not defined")
    if input_ids.size(1) > model.config.max_seq_len:
        raise ValueError("input length exceeds model max_seq_len")
    h = model.dropout(model.tok_embeddings(input_ids))
    pos_cis = model.pos_cis[:input_ids.size(1)] if model.pos_cis is not None else None
    for layer in model.layers:
        h, _ = layer(h, pos_cis, False)
    h = model.norm(h)
    return masked_lm_head_loss(model.output, h, targets, loss_mask, chunk_tokens, reduction=reduction)


def main():
    args = parse_args()
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if not is_cuda and dtype == torch.float16:
        raise ValueError("CPU training requires --dtype bfloat16")
    torch.manual_seed(args.seed)
    if is_cuda:
        torch.cuda.manual_seed_all(args.seed)
    ctx = torch.autocast(device_type=device.type, dtype=dtype)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    config = build_model_config(args.model_config, tokenizer)
    if args.gradient_checkpointing is not None:
        config.gradient_checkpointing = args.gradient_checkpointing
    if config.use_moe:
        raise ValueError("grouped SFT currently supports dense models only; MoE auxiliary loss weighting is not defined")
    config_fields = (
        "dim", "n_layers", "n_heads", "n_kv_heads", "linear_attn_heads", "vocab_size",
        "max_seq_len", "dropout", "norm_eps", "hidden_dim", "multiple_of", "use_moe",
        "layer_types", "conv_kernel_size", "linear_attn_chunk_size", "linear_attn_impl",
    )
    config_signature = {key: getattr(config, key) for key in config_fields}
    # Load on CPU: weights-only starts must not leave checkpoint optimizer tensors on GPU.
    model = MindLM(config)
    metadata = load_model_checkpoint(model, args.resume_from, "cpu")
    stage = metadata.get("training_stage")
    if stage not in (None, "pretrain", "sft"):
        raise ValueError(f"cannot start SFT from a {stage} checkpoint")
    saved_config = metadata.get("config", {})
    differences = {key: (saved_config[key], value) for key, value in config_signature.items()
                   if key in saved_config and saved_config[key] != value}
    if differences:
        raise ValueError(f"checkpoint model config mismatch: {differences}")
    metadata.pop("model", None)
    resume = None if args.resume_weights_only else metadata.get("extra_state", {}).get("grouped_sft")
    if not args.resume_weights_only and (stage != "sft" or resume is None):
        raise ValueError("full resume requires a grouped SFT checkpoint; use --resume_weights_only to start a new schedule")

    dataset = GroupedSFTDataset(args.bin, args.meta)
    lengths = [rec["n"] for rec in dataset.records]
    if max(lengths) - 1 > config.max_seq_len:
        raise ValueError("dataset input length exceeds model max_seq_len")
    sampler = LengthGroupedBatchSampler(lengths, args.token_budget, args.mega_batch,
                                        max_count=args.max_batch_count, seed=args.seed)
    # Pool composition varies by epoch, so count each deterministic epoch exactly.
    updates_by_epoch = []
    for epoch_index in range(args.epochs):
        sampler.set_epoch(epoch_index)
        updates_by_epoch.append(math.ceil(len(sampler) / args.accumulation_steps))
    total_updates = sum(updates_by_epoch)
    training_args = {key: getattr(args, key) for key in (
        "accumulation_steps", "mega_batch", "token_budget", "max_batch_count", "loss_chunk_tokens",
        "epochs", "learning_rate", "warmup_iters", "grad_clip", "weight_decay", "dtype", "seed",
    )}
    training_args.update(model_config=config_signature, gradient_checkpointing=config.gradient_checkpointing,
                         dataset=dataset.fingerprint, total_updates=total_updates, device_type=device.type)
    start_epoch, next_micro, global_update = 0, 0, 0
    wandb_run_id = None
    if resume is not None:
        if resume.get("version") != 1 or resume.get("training_args") != training_args:
            saved_args = resume.get("training_args", {})
            mismatch = {key: (saved_args.get(key), value) for key, value in training_args.items()
                        if saved_args.get(key) != value}
            raise ValueError(f"grouped SFT resume configuration mismatch: {mismatch}")
        start_epoch, next_micro, global_update = resume["epoch"], resume["next_micro"], resume["global_update"]
        if not 0 <= start_epoch < args.epochs:
            raise ValueError("invalid resume epoch")
        sampler.set_epoch(start_epoch)
        if not 0 <= next_micro <= len(sampler) or (next_micro != len(sampler) and next_micro % args.accumulation_steps):
            raise ValueError("resume cursor is not at an optimizer-update boundary")
        expected_update = sum(updates_by_epoch[:start_epoch]) + math.ceil(next_micro / args.accumulation_steps)
        if global_update != expected_update:
            raise ValueError("resume global_update disagrees with the epoch/micro-batch cursor")
        wandb_run_id = metadata.get("wandb_run_id")
    if args.limit_steps and args.limit_steps <= global_update:
        raise ValueError("--limit_steps must exceed the checkpoint's global_update")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=is_cuda)
    scaler = torch.amp.GradScaler(device.type, enabled=is_cuda and dtype == torch.float16)
    if resume is not None:
        optimizer.load_state_dict(metadata["optimizer"])
        scaler.load_state_dict(metadata["scaler"])
    del metadata
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_variable,
        num_workers=args.num_workers, pin_memory=is_cuda, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        # Worker/iterator seeds must not consume the model's dropout RNG on resume.
        generator=torch.Generator().manual_seed(args.seed),
    )
    wandb = None
    if args.use_wandb:
        import wandb as wandb_module
        continuing_wandb = resume is not None and wandb_run_id is not None
        wandb = wandb_module.init(project=args.wandb_project, name=args.wandb_run_name,
                                 id=wandb_run_id, resume="must" if continuing_wandb else "never", config=vars(args))
        wandb_run_id = wandb.id
    if resume is not None:
        torch.set_rng_state(resume["torch_rng_state"])
        if is_cuda:
            if len(resume["cuda_rng_state"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count differs from resume checkpoint")
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
    del resume
    log(f"Grouped SFT: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
        f"{len(dataset)} sequences, updates/epoch={updates_by_epoch}, total_updates={total_updates}, "
        f"start_update={global_update}, token_budget={args.token_budget}, loss_chunk={args.loss_chunk_tokens}, "
        f"gradient_checkpointing={config.gradient_checkpointing}, linear_attn_impl={config.linear_attn_impl}")

    checkpoint_path = Path(args.out_dir) / "mindlm_sft_grouped_latest.pt"
    epoch = start_epoch
    epoch_micro_batches = len(sampler.batches)

    def save_progress(destination=checkpoint_path):
        state = {"version": 1, "epoch": epoch, "next_micro": next_micro,
                 "global_update": global_update, "training_args": training_args,
                 "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if is_cuda else []}
        save_training_checkpoint(destination, model, optimizer, scaler, config,
                                 epoch, global_update, next_micro == epoch_micro_batches,
                                 training_stage="sft", wandb_run_id=wandb_run_id,
                                 extra_state={"grouped_sft": state})
        log(f"Saved checkpoint: {destination} (epoch={epoch}, next_micro={next_micro}, update={global_update})")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    initial_update = global_update
    stop = False
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_micro_batches = len(sampler.batches)
        sampler.start_batch = next_micro if epoch == start_epoch else 0
        next_micro = sampler.start_batch
        pending = window_tokens = window_padded = 0
        window_loss = 0.0
        window_start = time.monotonic()
        for micro, (input_ids, targets, loss_mask) in enumerate(loader, start=sampler.start_batch + 1):
            supervised_tokens = int(loss_mask.sum().item())
            if supervised_tokens == 0:
                raise ValueError("batch has no supervised tokens")
            padded_tokens = input_ids.numel()
            if padded_tokens > args.token_budget:
                raise RuntimeError("sampler exceeded padded input token budget")
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            with ctx:
                loss_sum = grouped_forward_loss(model, input_ids, targets, loss_mask, args.loss_chunk_tokens)
            if not torch.isfinite(loss_sum.detach()).item():
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, micro={micro}")
            scaler.scale(loss_sum).backward()
            window_loss += loss_sum.detach().item()
            window_tokens += supervised_tokens
            window_padded += padded_tokens
            pending += 1
            del loss_sum, input_ids, targets, loss_mask
            if pending != args.accumulation_steps and micro != epoch_micro_batches:
                continue

            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_tokens)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            lr = get_lr(global_update, total_updates, args.warmup_iters, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            next_micro = micro
            peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30 if is_cuda else 0.0
            peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30 if is_cuda else 0.0
            if global_update % args.log_interval == 0:
                elapsed = time.monotonic() - start
                eta_min = elapsed / max(global_update - initial_update, 1) * (total_updates - global_update) / 60
                metrics = {"loss": window_loss / window_tokens, "lr": lr, "epoch": epoch + 1,
                           "update": global_update, "supervised_tokens": window_tokens, "padded_tokens": window_padded,
                           "grad_norm": float(grad_norm), "peak_allocated_gib": peak_allocated,
                           "peak_reserved_gib": peak_reserved,
                           "supervised_tokens_per_second": window_tokens / max(time.monotonic() - window_start, 1e-9)}
                log(f"epoch={epoch + 1}/{args.epochs} update={global_update}/{total_updates} "
                    f"loss={metrics['loss']:.6f} lr={lr:.8f} supervised_tokens={window_tokens} "
                    f"padded_tokens={window_padded} grad_norm={float(grad_norm):.6f} "
                    f"peak_allocated_gib={peak_allocated:.3f} peak_reserved_gib={peak_reserved:.3f} "
                    f"supervised_tok_s={metrics['supervised_tokens_per_second']:.1f} eta={eta_min:.1f}m")
                if wandb is not None:
                    wandb.log(metrics, step=global_update)
                if is_cuda:
                    torch.cuda.reset_peak_memory_stats(device)
            epoch_complete = micro == epoch_micro_batches
            if epoch_complete or (args.save_interval and global_update % args.save_interval == 0):
                save_progress()
            if epoch_complete:
                save_progress(Path(args.out_dir) / f"mindlm_sft_grouped_epoch{epoch}.pt")
            pending = window_tokens = window_padded = 0
            window_loss = 0.0
            window_start = time.monotonic()
            if args.limit_steps and global_update >= args.limit_steps:
                stop = True
                break
        if stop:
            break

    save_progress()
    status = "Paused at step limit" if stop and global_update < total_updates else "Training complete"
    log(f"{status}: {global_update} optimizer updates in {(time.monotonic() - start) / 60:.1f} min")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
