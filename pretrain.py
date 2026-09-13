"""Pretrain one of the supported MindLM configurations."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import numpy as np
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
    cosine_learning_rate,
    EpochRandomSampler,
    load_model_checkpoint,
    extract_model_state,
    masked_lm_head_loss,
    masked_language_model_loss,
    save_training_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent

# Holds the primary process's wandb.Run so checkpoints can embed the run id
# and a resumed process can continue the same wandb run.
wandb_state = {"handle": None}


def is_primary_process():
    return not ddp or dist.get_rank() == 0


def log(message):
    if is_primary_process():
        print(message)


def get_lr(step, total_steps):
    return cosine_learning_rate(step, total_steps, args.warmup_iters, args.learning_rate)


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
    run_id = None
    if isinstance(wandb_state["handle"], object) and wandb_state["handle"] is not None:
        run_id = getattr(wandb_state["handle"], "id", None)
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
        wandb_run_id=run_id,
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
    parser.add_argument("--model_config", choices=("mindlm_0.1b", "mindlm_0.1b_moe", "mindlm_0.7b", "mindlm_0.2b_gdn"), default="mindlm_0.1b")
    parser.add_argument("--resume_from", default=None, help="Legacy state dict or MindLM checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=None, help="Legacy micro-batch warmup; V3 uses --warmup_updates/--warmup_ratio")
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
    parser.add_argument("--wandb_run_id", default=None, help="Force a wandb run id to continue an existing run")
    parser.add_argument("--val_packed_data_prefix", default=None, help="V3 heldout packed dataset; required for V3")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="V3 heldout batch size (defaults to training batch size)")
    parser.add_argument("--eval_interval", type=int, default=1000, help="V3 optimizer updates between full heldout evaluations; 0 = stop/epoch only")
    parser.add_argument("--loss_chunk_tokens", type=int, default=256, help="V3 supervised tokens per checkpointed LM-head chunk")
    parser.add_argument("--limit_updates", type=int, default=0, help="V3 absolute stopping update; does not change the full schedule")
    parser.add_argument("--warmup_updates", type=int, default=None, help="V3 explicit warmup in optimizer updates")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="V3 warmup fraction of the complete optimizer-update plan")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="V3 decay for matrix parameters")
    parser.add_argument("--seed", type=int, default=1337, help="V3 model and sampler seed")
    parsed = parser.parse_args()
    if parsed.epochs is None:
        parsed.epochs = 3 if parsed.model_config == "mindlm_0.2b_gdn" else 5
    if parsed.model_config == "mindlm_0.2b_gdn" and parsed.warmup_iters is not None:
        parser.error("V3 uses --warmup_updates or --warmup_ratio, not legacy --warmup_iters")
    if parsed.warmup_iters is None:
        parsed.warmup_iters = 100
    if parsed.tokenizer_path is None:
        parsed.tokenizer_path = str(
            REPOSITORY_ROOT / ("qwen3_tokenizer" if parsed.model_config in ("mindlm_0.7b", "mindlm_0.2b_gdn") else "mindlm_tokenizer")
        )
    if parsed.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if parsed.ddp_bucket_cap_mb < 1:
        raise ValueError("ddp_bucket_cap_mb must be positive")
    return parsed


def run_legacy(parsed_args):
    global args, ddp, ctx, config, model, optimizer, scaler, train_loader, iter_per_epoch
    args = parsed_args
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
        resume_run_id = (resume_metadata or {}).get("wandb_run_id") or args.wandb_run_id
        wandb_state["handle"] = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=resume_run_id,
            resume="must" if resume_run_id else None,
            config=vars(args),
        )

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


class V3BatchSampler:
    """Deterministic batches with a cursor that skips already consumed records."""

    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch = self.start_batch = 0

    def set_epoch(self, epoch):
        self.epoch, self.start_batch = epoch, 0

    def __len__(self):
        return math.ceil(self.size / self.batch_size) - self.start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for start in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield order[start:start + self.batch_size]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v3_data_fingerprint(dataset):
    return {"prefix": str(dataset.tokens_path.with_suffix("").resolve()),
            "bin_sha256": file_sha256(dataset.tokens_path),
            "metadata_sha256": file_sha256(dataset.tokens_path.with_suffix(".json")),
            "num_sequences": len(dataset), "sequence_length": dataset.max_length}


def v3_tokenizer_fingerprint(tokenizer, path):
    vocabulary = json.dumps(tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"):
        source = Path(path) / name
        if source.is_file():
            files[name] = file_sha256(source)
    return {"vocab_sha256": hashlib.sha256(vocabulary.encode()).hexdigest(), "files": files,
            "special_tokens": json.loads(json.dumps(tokenizer.special_tokens_map, default=str, sort_keys=True))}


def v3_runtime_fingerprint(device):
    packages = {}
    for name in ("transformers", "tokenizers", "numpy", "triton", "pytorch-triton", "flash-attn-4",
                 "flash-linear-attention", "tilelang", "apache-tvm-ffi", "nvidia-cutlass-dsl", "quack-kernels"):
        try:
            distribution = importlib.metadata.distribution(name)
            packages[name] = {"version": distribution.version,
                              "record_sha256": hashlib.sha256((distribution.read_text("RECORD") or "").encode()).hexdigest()}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    state = {"python": platform.python_version(), "torch": str(torch.__version__),
             "torch_path": torch.__file__, "torch_init_sha256": file_sha256(torch.__file__),
             "cuda": torch.version.cuda, "packages": packages,
             "device_type": device.type, "matmul_precision": torch.get_float32_matmul_precision(),
             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "source_sha256": {name: file_sha256(REPOSITORY_ROOT / name)
                               for name in ("pretrain.py", "training_utils.py", "modeling_mindlm.py", "config.py", "dataset.py")},
             "kernel_environment": {key: os.environ.get(key) for key in (
                 "FLA_DISABLE_BACKEND_DISPATCH", "FLA_TILELANG", "FLA_FLASH_QLA", "FLA_USE_TMA",
                 "TRITON_F32_DEFAULT", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                 "CUBLAS_WORKSPACE_CONFIG")}}
    if device.type == "cuda":
        state.update(gpu_name=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)),
                     cuda_device_count=torch.cuda.device_count(), cudnn_version=torch.backends.cudnn.version(),
                     cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    return state


def v3_rng_state(device):
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])}


def restore_v3_rng(state, device):
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA device count differs")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def v3_optimizer(model, learning_rate, weight_decay, is_cuda):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32:
            raise ValueError("V3 optimizer requires FP32 model parameters")
        exempt = parameter.ndim < 2 or name.endswith(("bias", "A_log", "dt_bias")) or getattr(parameter, "_no_weight_decay", False)
        (no_decay if exempt else decay).append(parameter)
    return optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                        {"params": no_decay, "weight_decay": 0.0}], lr=learning_rate, fused=is_cuda)


def v3_learning_rate(update, total_updates, warmup_updates, base_lr):
    if warmup_updates and update < warmup_updates:
        return base_lr * (update + 1) / warmup_updates
    progress = min(max((update - warmup_updates) / max(total_updates - warmup_updates - 1, 1), 0), 1)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def v3_autocast(device, dtype):
    return nullcontext() if dtype == "float32" else torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))


def v3_forward_loss(model, input_ids, targets, loss_mask, chunk_tokens):
    output = model(input_ids=input_ids, return_logits=False)
    return masked_lm_head_loss(model.output, output.last_hidden_state, targets, loss_mask,
                               chunk_tokens=chunk_tokens, reduction="sum")


def evaluate_v3(model, loader, device, dtype, chunk_tokens):
    """Full heldout token mean without gradients or changes to training RNG/state."""
    was_training = model.training
    rng = v3_rng_state(device)
    loss_total, tokens_total = 0.0, 0
    try:
        model.eval()
        with torch.inference_mode():
            for input_ids, targets, loss_mask in loader:
                count = int(loss_mask.sum().item())
                with v3_autocast(device, dtype):
                    loss = v3_forward_loss(model, input_ids.to(device), targets.to(device), loss_mask.to(device), chunk_tokens)
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError("non-finite heldout loss")
                loss_total += value
                tokens_total += count
        if tokens_total == 0:
            raise ValueError("heldout dataset contains no supervised tokens")
        return {"val_loss": loss_total / tokens_total, "val_supervised_tokens": tokens_total}
    finally:
        model.train(was_training)
        restore_v3_rng(rng, device)


def run_v3(parsed):
    if parsed.ddp or "RANK" in os.environ or parsed.compile:
        raise ValueError("V3 currently requires a single process without --ddp/--compile")
    if not parsed.packed_data_prefix or not parsed.val_packed_data_prefix:
        raise ValueError("V3 requires separate --packed_data_prefix and --val_packed_data_prefix")
    if Path(parsed.packed_data_prefix).resolve() == Path(parsed.val_packed_data_prefix).resolve():
        raise ValueError("training and heldout prefixes must differ")
    for name in ("batch_size", "accumulation_steps", "epochs", "loss_chunk_tokens", "log_interval"):
        if getattr(parsed, name) < 1:
            raise ValueError(f"{name} must be positive")
    if min(parsed.num_workers, parsed.eval_interval, parsed.save_interval, parsed.limit_updates) < 0:
        raise ValueError("workers and update intervals cannot be negative")
    if (not all(math.isfinite(getattr(parsed, name)) for name in ("warmup_ratio", "learning_rate", "weight_decay", "grad_clip"))
            or not 0 <= parsed.warmup_ratio < 1 or parsed.learning_rate <= 0 or parsed.weight_decay < 0 or parsed.grad_clip <= 0):
        raise ValueError("invalid warmup, learning rate, weight decay or gradient clipping")
    if parsed.warmup_updates is not None and parsed.warmup_updates < 0:
        raise ValueError("warmup_updates cannot be negative")
    eval_batch_size = parsed.batch_size if parsed.eval_batch_size is None else parsed.eval_batch_size
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    out_dir = Path(parsed.out_dir)
    if not parsed.resume_from and any(out_dir.glob(f"mindlm_pretrain_{parsed.model_config}_*.pt")):
        raise ValueError("random initialization requires an output directory without existing V3 checkpoints")
    device = torch.device(parsed.device)
    if parsed.dtype == "float16" or (device.type == "cuda" and parsed.dtype != "bfloat16"):
        raise ValueError("V3 uses BF16 on CUDA; CPU also supports FP32 references")
    random.seed(parsed.seed)
    np.random.seed(parsed.seed)
    torch.manual_seed(parsed.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(parsed.seed)
    tokenizer = AutoTokenizer.from_pretrained(parsed.tokenizer_path, trust_remote_code=True)
    model_config = build_model_config(parsed.model_config, tokenizer)
    if model_config.use_moe or model_config.linear_attn_impl != "gated_delta_rule" or model_config.initialization_scheme != "gdn_v3":
        raise ValueError("V3 requires dense gated_delta_rule with initialization_scheme=gdn_v3")
    if device.type == "cuda" and (model_config.attention_backend != "flash_attn_4" or model_config.linear_attn_backend != "fla"):
        raise ValueError("CUDA V3 requires flash_attn_4 and fla backends")
    model = MindLM(model_config)
    train_dataset = PackedPretrainDataset(parsed.packed_data_prefix, model_config.max_seq_len, len(tokenizer))
    val_dataset = PackedPretrainDataset(parsed.val_packed_data_prefix, model_config.max_seq_len, len(tokenizer))
    sampler = V3BatchSampler(len(train_dataset), parsed.batch_size, parsed.seed)
    batches_per_epoch = len(sampler)
    updates_per_epoch = math.ceil(batches_per_epoch / parsed.accumulation_steps)
    total_updates = parsed.epochs * updates_per_epoch
    warmup_updates = parsed.warmup_updates if parsed.warmup_updates is not None else math.ceil(total_updates * parsed.warmup_ratio)
    if warmup_updates >= total_updates:
        raise ValueError("warmup_updates must be smaller than the complete update plan")
    contract = {name: getattr(parsed, name) for name in (
        "model_config", "batch_size", "accumulation_steps", "epochs", "learning_rate", "weight_decay",
        "grad_clip", "dtype", "seed", "loss_chunk_tokens", "num_workers",
    )}
    contract.update(config=model_config.to_dict(), train=v3_data_fingerprint(train_dataset),
                    val=v3_data_fingerprint(val_dataset), tokenizer=v3_tokenizer_fingerprint(tokenizer, parsed.tokenizer_path),
                    runtime=v3_runtime_fingerprint(device), total_updates=total_updates,
                    updates_per_epoch=updates_per_epoch, warmup_updates=warmup_updates, eval_batch_size=eval_batch_size)
    if contract["train"]["bin_sha256"] == contract["val"]["bin_sha256"]:
        raise ValueError("training and heldout token files are identical")
    metadata, resume = None, None
    start_epoch = next_batch = global_update = 0
    wandb_run_id = None
    if parsed.resume_from:
        metadata = torch.load(parsed.resume_from, map_location="cpu", weights_only=True)
        resume = metadata.get("extra_state", {}).get("pretrain_v3")
        if metadata.get("training_stage") != "pretrain" or resume is None or resume.get("version") != 1:
            raise ValueError("V3 resume requires a complete V3 checkpoint; simple/legacy checkpoints cannot be resumed")
        if metadata.get("config") != resume.get("contract", {}).get("config"):
            raise ValueError("V3 checkpoint config and resume contract disagree")
        if resume.get("contract") != contract:
            changed = [key for key in contract if resume.get("contract", {}).get(key) != contract[key]]
            raise ValueError(f"V3 resume contract mismatch: {changed}; math/config/data/tokenizer/backend/runtime must agree")
        start_epoch, next_batch, global_update = resume["epoch"], resume["next_batch"], resume["global_update"]
        if not 0 <= start_epoch < parsed.epochs or not 0 <= next_batch <= batches_per_epoch:
            raise ValueError("invalid V3 resume cursor")
        if next_batch != batches_per_epoch and next_batch % parsed.accumulation_steps:
            raise ValueError("V3 cursor must be at an optimizer-update boundary")
        if global_update != start_epoch * updates_per_epoch + math.ceil(next_batch / parsed.accumulation_steps):
            raise ValueError("V3 global_update and cursor disagree")
        if (metadata.get("epoch") != start_epoch or metadata.get("step") != global_update
                or metadata.get("epoch_complete") != (next_batch == batches_per_epoch)):
            raise ValueError("V3 checkpoint metadata and resume cursor disagree")
        model.load_state_dict(extract_model_state(metadata), strict=True)
        metadata.pop("model", None)
        wandb_run_id = metadata.get("wandb_run_id")
        if parsed.wandb_run_id and parsed.wandb_run_id != wandb_run_id:
            raise ValueError("cannot replace the checkpoint's W&B run ID during full resume")
    elif parsed.wandb_run_id:
        raise ValueError("random initialization starts a new W&B run; --wandb_run_id is only accepted for matching resume")
    if parsed.limit_updates and parsed.limit_updates <= global_update:
        raise ValueError("limit_updates must exceed the checkpoint's global_update")
    model.to(device)
    optimizer_v3 = v3_optimizer(model, parsed.learning_rate, parsed.weight_decay, device.type == "cuda")
    scaler_v3 = torch.amp.GradScaler(device.type, enabled=False)
    if resume is not None:
        optimizer_v3.load_state_dict(metadata["optimizer"])
        scaler_v3.load_state_dict(metadata["scaler"])
    del metadata
    loader_kwargs = {"num_workers": parsed.num_workers, "pin_memory": device.type == "cuda"}
    if parsed.num_workers:
        loader_kwargs.update(persistent_workers=not parsed.no_persistent_workers, prefetch_factor=parsed.prefetch_factor)
    train = DataLoader(train_dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(parsed.seed), **loader_kwargs)
    val = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False,
                     generator=torch.Generator().manual_seed(parsed.seed + 1), **loader_kwargs)
    wandb = None
    if parsed.use_wandb:
        import wandb as wandb_module
        wandb = wandb_module.init(project=parsed.wandb_project, name=parsed.wandb_run_name, id=wandb_run_id,
                                 resume="must" if wandb_run_id else "never", config=vars(parsed))
        wandb_run_id = wandb.id
    if resume is not None:
        restore_v3_rng(resume["rng"], device)
    del resume
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch = start_epoch

    def save_progress(destination=None):
        path = destination or out_dir / f"mindlm_pretrain_{parsed.model_config}_latest.pt"
        state = {"version": 1, "epoch": epoch, "next_batch": next_batch, "global_update": global_update,
                 "contract": contract, "rng": v3_rng_state(device)}
        save_training_checkpoint(path, model, optimizer_v3, scaler_v3, model_config, epoch, global_update,
                                 next_batch == batches_per_epoch, training_stage="pretrain", wandb_run_id=wandb_run_id,
                                 extra_state={"pretrain_v3": state})
        print(f"Saved checkpoint: {path} epoch={epoch} next_batch={next_batch} update={global_update}", flush=True)

    print(f"V3 config={parsed.model_config} params={sum(p.numel() for p in model.parameters())} "
          f"train_sequences={len(train_dataset)} val_sequences={len(val_dataset)} batches_per_epoch={batches_per_epoch} "
          f"updates_per_epoch={updates_per_epoch} total_updates={total_updates} warmup_updates={warmup_updates} "
          f"start_update={global_update} linear={model_config.linear_attn_impl}/{model_config.linear_attn_backend} "
          f"attention={model_config.attention_backend} init={model_config.initialization_scheme}", flush=True)
    print("V3_CONTRACT " + json.dumps(contract, sort_keys=True), flush=True)
    model.train()
    optimizer_v3.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stopped = False
    for epoch in range(start_epoch, parsed.epochs):
        sampler.set_epoch(epoch)
        sampler.start_batch = next_batch if epoch == start_epoch else 0
        next_batch = sampler.start_batch
        pending = tokens = padded_tokens = 0
        window_loss = 0.0
        window_start = time.monotonic()
        for batch, (input_ids, targets, mask) in enumerate(train, sampler.start_batch + 1):
            count = int(mask.sum().item())
            if count == 0:
                raise ValueError("training batch has no supervised tokens")
            padded_tokens += input_ids.numel()
            with v3_autocast(device, parsed.dtype):
                loss_sum = v3_forward_loss(model, input_ids.to(device), targets.to(device), mask.to(device), parsed.loss_chunk_tokens)
            value = loss_sum.detach().item()
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite loss epoch={epoch} batch={batch}")
            scaler_v3.scale(loss_sum).backward()
            window_loss += value
            tokens += count
            pending += 1
            del loss_sum, input_ids, targets, mask
            if pending != parsed.accumulation_steps and batch != batches_per_epoch:
                continue
            scaler_v3.unscale_(optimizer_v3)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(tokens)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), parsed.grad_clip, error_if_nonfinite=True)
            lr = v3_learning_rate(global_update, total_updates, warmup_updates, parsed.learning_rate)
            for group in optimizer_v3.param_groups:
                group["lr"] = lr
            scaler_v3.step(optimizer_v3)
            scaler_v3.update()
            optimizer_v3.zero_grad(set_to_none=True)
            global_update += 1
            next_batch = batch
            epoch_complete = batch == batches_per_epoch
            stopped = bool(parsed.limit_updates and global_update >= parsed.limit_updates)
            metrics = {"loss": window_loss / tokens, "grad_norm": float(grad_norm), "lr": lr,
                       "update": global_update, "epoch": epoch + 1, "supervised_tokens": tokens,
                       "padded_tokens": padded_tokens, "tokens_per_second": tokens / max(time.monotonic() - window_start, 1e-9)}
            if device.type == "cuda":
                metrics.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                               peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
            eval_due = epoch_complete or stopped or (parsed.eval_interval and global_update % parsed.eval_interval == 0)
            if eval_due:
                metrics.update(evaluate_v3(model, val, device, parsed.dtype, parsed.loss_chunk_tokens))
            if global_update % parsed.log_interval == 0 or eval_due:
                print("V3_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
                if wandb is not None:
                    wandb.log(metrics, step=global_update)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            if epoch_complete or (parsed.save_interval and global_update % parsed.save_interval == 0):
                save_progress()
            if epoch_complete:
                save_progress(out_dir / f"mindlm_pretrain_{parsed.model_config}_epoch{epoch}.pt")
            pending = tokens = padded_tokens = 0
            window_loss = 0.0
            window_start = time.monotonic()
            if stopped:
                break
        if stopped:
            break
    save_progress()
    print(f"V3 {'paused' if stopped and global_update < total_updates else 'complete'} update={global_update}/{total_updates}", flush=True)
    if wandb is not None:
        wandb.finish()


def main():
    parsed = parse_args()
    if parsed.model_config == "mindlm_0.2b_gdn":
        run_v3(parsed)
    else:
        run_legacy(parsed)


if __name__ == "__main__":
    main()
