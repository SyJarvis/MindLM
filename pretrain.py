"""Train MindLM on fixed-length, pre-tokenized causal-LM data.

The trainer consumes the packed dataset described in
``docs/MindLM_Pretrain_Data_Format_v1.md``. A dataset prefix points to a
``.bin`` file containing ``uint32`` token records and a matching ``.json``
manifest. Each record contains ``sequence_length + 1`` tokens; the first
``sequence_length`` tokens are inputs and the remaining tokens are targets.

Runtime fingerprints, strict checkpoint state, and the resumable sampler are
kept in named helpers so that the training loop only deals with optimization.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import PackedPretrainDataset
from modeling_mindlm import MindLM
from training_utils import (
    build_model_config,
    extract_model_state,
    masked_lm_head_loss,
    save_training_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
CHECKPOINT_STATE_KEY = "pretrain"
CHECKPOINT_STATE_VERSION = 2
CHAT_EOS_TOKEN = "<|im_end|>"
HASH_BLOCK_BYTES = 8 * 1024 * 1024
MUTABLE_RESUME_KEYS = frozenset({
    "batch_size",
    "gradient_accumulation_steps",
    "loss_chunk_tokens",
    "num_workers",
    "total_updates",
    "updates_per_epoch",
    "warmup_updates",
    "eval_batch_size",
})
MUTABLE_CONFIG_KEYS = frozenset({"gradient_checkpointing"})
SOURCE_FILES = (
    "pretrain.py",
    "training_utils.py",
    "modeling_mindlm.py",
    "config.py",
    "dataset.py",
)
RUNTIME_PACKAGES = (
    "transformers",
    "tokenizers",
    "numpy",
    "triton",
    "pytorch-triton",
    "flash-attn-4",
    "flash-linear-attention",
    "tilelang",
    "apache-tvm-ffi",
    "nvidia-cutlass-dsl",
    "quack-kernels",
)
KERNEL_ENVIRONMENT_KEYS = (
    "FLA_DISABLE_BACKEND_DISPATCH",
    "FLA_TILELANG",
    "FLA_FLASH_QLA",
    "FLA_USE_TMA",
    "TRITON_F32_DEFAULT",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG",
)


@dataclass(frozen=True)
class TrainingPlan:
    """The optimizer-update schedule derived from the dataset and CLI."""

    batches_per_epoch: int
    updates_per_epoch: int
    total_updates: int
    warmup_updates: int


@dataclass
class TrainingCursor:
    """Position persisted in a checkpoint and used to resume exactly."""

    epoch: int = 0
    next_record: int = 0
    next_batch: int = 0
    global_update: int = 0


def parse_args(argv=None):
    """Parse CLI options without touching files or initializing CUDA."""

    parser = argparse.ArgumentParser(
        description="Train MindLM on packed pre-tokenized causal-LM data"
    )
    parser.add_argument(
        "--output_dir",
        default="out",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--train_data_prefix",
        required=True,
        help="Training prefix containing <prefix>.bin and <prefix>.json",
    )
    parser.add_argument(
        "--validation_data_prefix",
        required=True,
        help="Held-out prefix containing <prefix>.bin and <prefix>.json",
    )
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument(
        "--model_config",
        choices=("mindlm_0.2b_gdn",),
        default="mindlm_0.2b_gdn",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        choices=("off", "linear_attn", "all"),
        default=None,
        help="Override the config checkpointing policy; safe to change on resume",
    )
    parser.add_argument(
        "--resume_from", default=None, help="Complete MindLM pretraining checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MindLM-Pretrain")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_run_id",
        default=None,
        help="Force a W&B run id when resuming that exact run",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Held-out batch size (defaults to --batch_size)",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="Optimizer updates between held-out evaluations; 0 disables periodic evaluation",
    )
    parser.add_argument(
        "--loss_chunk_tokens",
        type=int,
        default=256,
        help="Supervised tokens per checkpointed LM-head chunk",
    )
    parser.add_argument(
        "--limit_updates",
        type=int,
        default=0,
        help="Absolute optimizer-update limit; 0 trains the complete schedule",
    )
    parser.add_argument(
        "--warmup_updates",
        type=int,
        default=None,
        help="Explicit warmup length in optimizer updates",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.02,
        help="Warmup fraction of the complete optimizer-update plan",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="AdamW decay applied to matrix parameters",
    )
    parser.add_argument("--seed", type=int, default=1337)

    parsed = parser.parse_args(argv)
    if parsed.epochs is None:
        parsed.epochs = 3
    if parsed.tokenizer_path is None:
        parsed.tokenizer_path = str(REPOSITORY_ROOT / "qwen3_tokenizer")
    return parsed


def validate_args(args):
    """Validate arguments whose constraints depend on the training contract."""

    if not getattr(args, "train_data_prefix", None) or not getattr(
        args, "validation_data_prefix", None
    ):
        raise ValueError(
            "pretraining requires separate --train_data_prefix and --validation_data_prefix"
        )
    train_prefix = Path(args.train_data_prefix).resolve()
    validation_prefix = Path(args.validation_data_prefix).resolve()
    if train_prefix == validation_prefix:
        raise ValueError("training and validation prefixes must differ")

    for name in (
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "loss_chunk_tokens",
        "log_interval",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if min(args.num_workers, args.eval_interval, args.save_interval, args.limit_updates) < 0:
        raise ValueError("workers and update intervals cannot be negative")
    if (
        not all(
            math.isfinite(getattr(args, name))
            for name in ("warmup_ratio", "learning_rate", "weight_decay", "grad_clip")
        )
        or not 0 <= args.warmup_ratio < 1
        or args.learning_rate <= 0
        or args.weight_decay < 0
        or args.grad_clip <= 0
    ):
        raise ValueError("invalid warmup, learning rate, weight decay or gradient clipping")
    if args.warmup_updates is not None and args.warmup_updates < 0:
        raise ValueError("warmup_updates cannot be negative")
    eval_batch_size = args.batch_size if args.eval_batch_size is None else args.eval_batch_size
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    if args.num_workers and getattr(args, "prefetch_factor", 4) < 1:
        raise ValueError("prefetch_factor must be positive when workers are enabled")
    return train_prefix, validation_prefix, eval_batch_size


class ResumableBatchSampler:
    """Deterministically shuffle records and resume from a record cursor."""

    def __init__(self, dataset_size, batch_size, seed):
        if dataset_size < 1 or batch_size < 1:
            raise ValueError("dataset_size and batch_size must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.start_record = 0
        self.start_batch = 0

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch
        self.start_record = 0
        self.start_batch = 0

    def set_start_record(self, record):
        if not 0 <= record <= self.dataset_size:
            raise ValueError("start record must be within the dataset")
        self.start_record = record
        self.start_batch = math.ceil(record / self.batch_size)

    def __len__(self):
        return math.ceil(max(self.dataset_size - self.start_record, 0) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        first = self.start_record
        for start in range(first, self.dataset_size, self.batch_size):
            yield order[start : start + self.batch_size]


def sha256_file(path):
    """Return a streaming SHA256 digest for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_data(dataset):
    """Fingerprint the exact packed token file consumed by the trainer."""

    metadata_path = dataset.tokens_path.with_suffix(".json")
    return {
        "prefix": str(dataset.tokens_path.with_suffix("").resolve()),
        "bin_sha256": sha256_file(dataset.tokens_path),
        "metadata_sha256": sha256_file(metadata_path),
        "num_sequences": len(dataset),
        "sequence_length": dataset.max_length,
    }


def chat_eos_id(tokenizer):
    """Resolve the explicit Qwen3 chat boundary token."""

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ValueError("tokenizer must expose convert_tokens_to_ids")
    vocabulary = tokenizer.get_vocab()
    if CHAT_EOS_TOKEN not in vocabulary:
        raise ValueError(f"tokenizer does not contain {CHAT_EOS_TOKEN}")
    token_id = convert(CHAT_EOS_TOKEN)
    if token_id is None or isinstance(token_id, (list, tuple)):
        raise ValueError(f"tokenizer does not contain {CHAT_EOS_TOKEN}")
    token_id = int(token_id)
    if not 0 <= token_id < len(tokenizer):
        raise ValueError(f"invalid {CHAT_EOS_TOKEN} id: {token_id}")
    return token_id


def validate_dataset_boundary(dataset, expected_token_id):
    """Reject a packed manifest built with a different chat boundary."""

    declared = dataset.metadata.get("boundary_token_id")
    if declared is not None and int(declared) != expected_token_id:
        raise ValueError(
            "packed dataset chat boundary does not match the selected tokenizer: "
            f"{declared} != {expected_token_id}"
        )


def fingerprint_tokenizer(tokenizer, path):
    """Fingerprint tokenizer vocabulary, special tokens, and serialized files."""

    vocabulary = json.dumps(
        tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ):
        source = Path(path) / name
        if source.is_file():
            files[name] = sha256_file(source)
    return {
        "vocab_sha256": hashlib.sha256(vocabulary.encode()).hexdigest(),
        "files": files,
        "special_tokens": json.loads(
            json.dumps(tokenizer.special_tokens_map, default=str, sort_keys=True)
        ),
    }


def fingerprint_runtime(device):
    """Capture software, hardware, and kernel settings affecting reproducibility."""

    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            distribution = importlib.metadata.distribution(name)
            record = distribution.read_text("RECORD") or ""
            packages[name] = {
                "version": distribution.version,
                "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
            }
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    state = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_path": torch.__file__,
        "torch_init_sha256": sha256_file(torch.__file__),
        "cuda": torch.version.cuda,
        "packages": packages,
        "device_type": device.type,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "source_sha256": {
            name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES
        },
        "kernel_environment": {
            key: os.environ.get(key) for key in KERNEL_ENVIRONMENT_KEYS
        },
    }
    if device.type == "cuda":
        state.update(
            gpu_name=torch.cuda.get_device_name(device),
            capability=list(torch.cuda.get_device_capability(device)),
            cuda_device_count=torch.cuda.device_count(),
            cudnn_version=torch.backends.cudnn.version(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        )
    return state


def capture_rng_state(device):
    """Capture every RNG used by the trainer, including all CUDA devices."""

    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }


def restore_rng_state(state, device):
    """Restore a state produced by :func:`capture_rng_state`."""

    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA device count differs")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )


def seed_everything(seed, device):
    """Seed Python, NumPy, PyTorch, and CUDA before model construction."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, learning_rate, weight_decay, use_fused):
    """Create AdamW groups with standard no-decay exceptions."""

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32:
            raise ValueError("pretraining optimizer requires FP32 model parameters")
        exempt = (
            parameter.ndim < 2
            or name.endswith(("bias", "A_log", "dt_bias"))
            or getattr(parameter, "_no_weight_decay", False)
        )
        (no_decay if exempt else decay).append(parameter)
    return optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        fused=use_fused,
    )


def learning_rate_at(update, total_updates, warmup_updates, base_lr):
    """Warm up linearly, then cosine-decay to ten percent of ``base_lr``."""

    if warmup_updates and update < warmup_updates:
        return base_lr * (update + 1) / warmup_updates
    progress = min(
        max(
            (update - warmup_updates) / max(total_updates - warmup_updates - 1, 1),
            0,
        ),
        1,
    )
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def autocast_context(device, dtype):
    """Return the configured autocast context, or a no-op for FP32."""

    return (
        nullcontext()
        if dtype == "float32"
        else torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))
    )


def forward_loss(model, input_ids, targets, loss_mask, chunk_tokens):
    """Run the model and compute chunked masked CE without materializing logits."""

    output = model(input_ids=input_ids, return_logits=False)
    return masked_lm_head_loss(
        model.output,
        output.last_hidden_state,
        targets,
        loss_mask,
        chunk_tokens=chunk_tokens,
        reduction="sum",
    )


def evaluate(model, loader, device, dtype, chunk_tokens):
    """Compute held-out token mean while preserving train mode, gradients, and RNG."""

    was_training = model.training
    rng_state = capture_rng_state(device)
    loss_total, token_total = 0.0, 0
    try:
        model.eval()
        with torch.inference_mode():
            for input_ids, targets, loss_mask in loader:
                supervised_tokens = int(loss_mask.sum().item())
                with autocast_context(device, dtype):
                    loss = forward_loss(
                        model,
                        input_ids.to(device, non_blocking=device.type == "cuda"),
                        targets.to(device, non_blocking=device.type == "cuda"),
                        loss_mask.to(device, non_blocking=device.type == "cuda"),
                        chunk_tokens,
                    )
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError("non-finite heldout loss")
                loss_total += value
                token_total += supervised_tokens
        if token_total == 0:
            raise ValueError("heldout dataset contains no supervised tokens")
        return {
            "val_loss": loss_total / token_total,
            "val_supervised_tokens": token_total,
        }
    finally:
        model.train(was_training)
        restore_rng_state(rng_state, device)


def build_training_plan(args, batches_per_epoch):
    """Derive optimizer and warmup counts from the full, unskipped schedule."""

    updates_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_updates = args.epochs * updates_per_epoch
    warmup_updates = (
        args.warmup_updates
        if args.warmup_updates is not None
        else math.ceil(total_updates * args.warmup_ratio)
    )
    if warmup_updates >= total_updates:
        raise ValueError("warmup_updates must be smaller than the complete update plan")
    return TrainingPlan(
        batches_per_epoch=batches_per_epoch,
        updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
    )


def build_training_contract(
    args,
    model_config,
    train_dataset,
    validation_dataset,
    tokenizer,
    device,
    plan,
    eval_batch_size,
):
    """Build the contract used to validate data and math on resume."""

    contract = {
        name: getattr(args, name)
        for name in (
            "model_config",
            "batch_size",
            "gradient_accumulation_steps",
            "epochs",
            "learning_rate",
            "weight_decay",
            "grad_clip",
            "dtype",
            "seed",
            "loss_chunk_tokens",
            "num_workers",
        )
    }
    contract.update(
        data_format={
            "format": PackedPretrainDataset.FORMAT,
            "boundary_token": CHAT_EOS_TOKEN,
            "boundary_token_id": chat_eos_id(tokenizer),
            "token_dtype": "uint32",
            "loss_mask_dtype": "uint8",
        },
        config=model_config.to_dict(),
        train=fingerprint_data(train_dataset),
        val=fingerprint_data(validation_dataset),
        tokenizer=fingerprint_tokenizer(tokenizer, args.tokenizer_path),
        runtime=fingerprint_runtime(device),
        total_updates=plan.total_updates,
        updates_per_epoch=plan.updates_per_epoch,
        warmup_updates=plan.warmup_updates,
        eval_batch_size=eval_batch_size,
    )
    if contract["train"]["bin_sha256"] == contract["val"]["bin_sha256"]:
        raise ValueError("training and validation token files are identical")
    return contract


def _resume_contract_signature(contract):
    """Return the immutable part of a contract for elastic resume checks."""

    signature = dict(contract)
    for key in MUTABLE_RESUME_KEYS:
        signature.pop(key, None)
    signature["config"] = _resume_config_signature(signature.get("config", {}))
    return signature


def _resume_config_signature(config):
    """Return the immutable model configuration for checkpoint comparison."""

    signature = dict(config)
    for key in MUTABLE_CONFIG_KEYS:
        signature.pop(key, None)
    return signature


def _seeded_loaders(
    args,
    device,
    train_dataset,
    validation_dataset,
    batch_sampler,
    eval_batch_size,
):
    """Create train and validation loaders with bounded worker settings."""

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        loader_kwargs.update(
            persistent_workers=not getattr(args, "no_persistent_workers", False),
            prefetch_factor=getattr(args, "prefetch_factor", 4),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **loader_kwargs,
    )
    return train_loader, validation_loader


def _load_resume_state(path):
    """Load and validate the persisted pretraining state envelope."""

    metadata = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(metadata, dict):
        raise ValueError("pretraining checkpoint must be a dictionary")
    extra_state = metadata.get("extra_state", {})
    state = extra_state.get(CHECKPOINT_STATE_KEY)
    if (
        metadata.get("training_stage") != "pretrain"
        or not isinstance(state, dict)
        or state.get("version") not in (1, CHECKPOINT_STATE_VERSION)
    ):
        raise ValueError("pretraining resume requires a complete pretraining checkpoint")
    required = {"epoch", "next_batch", "global_update", "contract", "rng"}
    if not required <= state.keys():
        missing = sorted(required - state.keys())
        raise ValueError(f"pretraining checkpoint is missing state fields: {missing}")
    if state["version"] == 1:
        batch_size = state["contract"].get("batch_size")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("legacy pretraining checkpoint has invalid batch_size")
        state = dict(state)
        state["version"] = CHECKPOINT_STATE_VERSION
        train_sequences = state["contract"].get("train", {}).get("num_sequences")
        if not isinstance(train_sequences, int) or train_sequences < 1:
            raise ValueError("legacy pretraining checkpoint has invalid train size")
        state["next_record"] = min(state["next_batch"] * batch_size, train_sequences)
        state["schedule"] = {
            "total_updates": state["contract"].get("total_updates"),
            "warmup_updates": state["contract"].get("warmup_updates"),
            "base_learning_rate": state["contract"].get("learning_rate"),
        }
        optimizer_groups = metadata.get("optimizer", {}).get("param_groups", [])
        state["optimizer_step"] = {
            "learning_rate": optimizer_groups[0].get("lr") if optimizer_groups else None,
        }
        state["last_metrics"] = None
        state["legacy_contract"] = True
        return metadata, state
    required_v2 = {"next_record", "schedule", "optimizer_step", "last_metrics"}
    if not required_v2 <= state.keys():
        missing = sorted(required_v2 - state.keys())
        raise ValueError(f"pretraining checkpoint is missing state fields: {missing}")
    return metadata, state


def _validate_resume_cursor(metadata, state, train_size, args):
    """Validate checkpoint cursor and public metadata before loading weights."""

    saved_batch_size = state["contract"].get("batch_size")
    if not isinstance(saved_batch_size, int) or saved_batch_size < 1:
        raise ValueError("pretraining checkpoint has invalid batch_size")
    cursor = TrainingCursor(
        epoch=state["epoch"],
        next_record=state["next_record"],
        next_batch=state["next_batch"],
        global_update=state["global_update"],
    )
    if not 0 <= cursor.epoch < args.epochs or not 0 <= cursor.next_record <= train_size:
        raise ValueError("invalid pretraining resume cursor")
    saved_expected_batch = math.ceil(cursor.next_record / saved_batch_size)
    if cursor.next_batch != saved_expected_batch:
        raise ValueError("pretraining cursor record and batch positions disagree")
    if cursor.global_update < 0:
        raise ValueError("pretraining global_update cannot be negative")
    if (
        metadata.get("epoch") != cursor.epoch
        or metadata.get("step") != cursor.global_update
        or metadata.get("epoch_complete") != (cursor.next_record == train_size)
    ):
        raise ValueError("pretraining checkpoint metadata and resume cursor disagree")
    cursor.next_batch = math.ceil(cursor.next_record / args.batch_size)
    return cursor


class PackedPretrainTrainer:
    """Own model, data, optimizer, checkpoint, and logging state for one run."""

    def __init__(self, args):
        self.args = args
        self.train_prefix, self.validation_prefix, self.eval_batch_size = validate_args(args)
        self.output_dir = Path(args.output_dir)
        if not args.resume_from and any(
            self.output_dir.glob(f"mindlm_pretrain_{args.model_config}_*.pt")
        ):
            raise ValueError(
                "random initialization requires an output directory without existing pretraining checkpoints"
            )

        self.device = torch.device(args.device)
        if self.device.type == "cuda" and args.dtype != "bfloat16":
            raise ValueError("CUDA pretraining requires bfloat16; use float32 for CPU reference runs")
        seed_everything(args.seed, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
        self.model_config = build_model_config(args.model_config, self.tokenizer)
        if args.gradient_checkpointing is not None:
            self.model_config.gradient_checkpointing = args.gradient_checkpointing
        if self.device.type == "cuda" and (
            self.model_config.attention_backend != "flash_attn_4"
            or self.model_config.linear_attn_backend != "fla"
        ):
            raise ValueError("CUDA pretraining requires flash_attn_4 and fla backends")

        self.model = MindLM(self.model_config)
        self.train_dataset = PackedPretrainDataset(
            self.train_prefix, self.model_config.max_seq_len, len(self.tokenizer)
        )
        self.validation_dataset = PackedPretrainDataset(
            self.validation_prefix, self.model_config.max_seq_len, len(self.tokenizer)
        )
        expected_chat_eos_id = chat_eos_id(self.tokenizer)
        validate_dataset_boundary(self.train_dataset, expected_chat_eos_id)
        validate_dataset_boundary(self.validation_dataset, expected_chat_eos_id)
        self.batch_sampler = ResumableBatchSampler(
            len(self.train_dataset), args.batch_size, args.seed
        )
        self.plan = build_training_plan(args, len(self.batch_sampler))
        self.schedule = {
            "total_updates": self.plan.total_updates,
            "warmup_updates": self.plan.warmup_updates,
            "base_learning_rate": args.learning_rate,
        }
        self.contract = build_training_contract(
            args,
            self.model_config,
            self.train_dataset,
            self.validation_dataset,
            self.tokenizer,
            self.device,
            self.plan,
            self.eval_batch_size,
        )

        self.cursor = TrainingCursor()
        self.resume_state = None
        self.last_metrics = None
        self.current_learning_rate = args.learning_rate
        self.wandb_run_id = None
        checkpoint_metadata = None
        if args.resume_from:
            checkpoint_metadata, self.resume_state = _load_resume_state(args.resume_from)
            if _resume_config_signature(checkpoint_metadata.get("config", {})) != _resume_config_signature(
                self.resume_state.get("contract", {}).get("config", {})
            ):
                raise ValueError("pretraining checkpoint config and resume contract disagree")
            if _resume_contract_signature(self.resume_state.get("contract", {})) != _resume_contract_signature(self.contract):
                changed = [
                    key
                    for key in self.contract
                    if key not in MUTABLE_RESUME_KEYS
                    and _resume_contract_signature(self.resume_state.get("contract", {})).get(key)
                    != _resume_contract_signature(self.contract).get(key)
                ]
                raise ValueError(
                    f"pretraining resume contract mismatch: {changed}; math/config/data/tokenizer/backend/runtime must agree"
                )
            self.cursor = _validate_resume_cursor(
                checkpoint_metadata, self.resume_state, len(self.train_dataset), args
            )
            self.schedule = dict(self.resume_state["schedule"])
            if self.schedule["base_learning_rate"] != self.contract["learning_rate"]:
                raise ValueError("pretraining checkpoint schedule and learning_rate disagree")
            self.last_metrics = self.resume_state["last_metrics"]
            self.model.load_state_dict(extract_model_state(checkpoint_metadata), strict=True)
            self.wandb_run_id = checkpoint_metadata.get("wandb_run_id")
            if args.wandb_run_id and args.wandb_run_id != self.wandb_run_id:
                raise ValueError("cannot replace the checkpoint's W&B run ID during full resume")
        elif args.wandb_run_id:
            raise ValueError(
                "random initialization starts a new W&B run; --wandb_run_id is only accepted for matching resume"
            )
        if args.limit_updates and args.limit_updates <= self.cursor.global_update:
            raise ValueError("limit_updates must exceed the checkpoint's global_update")

        self.model.to(self.device)
        self.optimizer = build_optimizer(
            self.model,
            args.learning_rate,
            args.weight_decay,
            self.device.type == "cuda",
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
        if checkpoint_metadata is not None:
            self.optimizer.load_state_dict(checkpoint_metadata["optimizer"])
            self.scaler.load_state_dict(checkpoint_metadata["scaler"])
            self.current_learning_rate = float(
                self.resume_state["optimizer_step"]["learning_rate"]
            )
            for group in self.optimizer.param_groups:
                group["lr"] = self.current_learning_rate

        self.train_loader, self.validation_loader = _seeded_loaders(
            args,
            self.device,
            self.train_dataset,
            self.validation_dataset,
            self.batch_sampler,
            self.eval_batch_size,
        )
        self.wandb = self._init_wandb()
        if self.resume_state is not None:
            restore_rng_state(self.resume_state["rng"], self.device)

    def _init_wandb(self):
        if not self.args.use_wandb:
            return None
        import wandb as wandb_module

        run = wandb_module.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            id=self.wandb_run_id,
            resume="must" if self.wandb_run_id else "never",
            config=vars(self.args),
        )
        self.wandb_run_id = run.id
        return run

    def save_progress(self, destination=None):
        """Atomically save the exact cursor and runtime state."""

        path = destination or self.output_dir / f"mindlm_pretrain_{self.args.model_config}_latest.pt"
        checkpoint_metrics = None
        if self.last_metrics is not None:
            checkpoint_metrics = {
                key: value
                for key, value in self.last_metrics.items()
                if key not in {"tokens_per_second", "peak_allocated_gib", "peak_reserved_gib"}
            }
        state = {
            "version": CHECKPOINT_STATE_VERSION,
            "epoch": self.cursor.epoch,
            "next_record": self.cursor.next_record,
            "next_batch": self.cursor.next_batch,
            "global_update": self.cursor.global_update,
            "contract": self.contract,
            "schedule": dict(self.schedule),
            "optimizer_step": {
                "learning_rate": self.current_learning_rate,
            },
            "last_metrics": checkpoint_metrics,
            "rng": capture_rng_state(self.device),
        }
        save_training_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scaler,
            self.model_config,
            self.cursor.epoch,
            self.cursor.global_update,
            self.cursor.next_record == len(self.train_dataset),
            training_stage="pretrain",
            wandb_run_id=self.wandb_run_id,
            extra_state={CHECKPOINT_STATE_KEY: state},
        )
        print(
            f"Saved checkpoint: {path} epoch={self.cursor.epoch} "
            f"next_batch={self.cursor.next_batch} update={self.cursor.global_update}",
            flush=True,
        )

    def _print_run_summary(self):
        print(
            f"Pretrain config={self.args.model_config} "
            f"params={sum(p.numel() for p in self.model.parameters())} "
            f"train_sequences={len(self.train_dataset)} "
            f"val_sequences={len(self.validation_dataset)} "
            f"batches_per_epoch={self.plan.batches_per_epoch} "
            f"updates_per_epoch={self.plan.updates_per_epoch} "
            f"total_updates={self.plan.total_updates} "
            f"warmup_updates={self.plan.warmup_updates} "
            f"start_update={self.cursor.global_update} "
            f"linear={self.model_config.linear_attn_backend} "
            f"attention={self.model_config.attention_backend}",
            flush=True,
        )
        print("PRETRAIN_CONTRACT " + json.dumps(self.contract, sort_keys=True), flush=True)

    def _optimizer_update(self, token_count):
        self.scaler.unscale_(self.optimizer)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(token_count)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.grad_clip, error_if_nonfinite=True
        )
        lr = learning_rate_at(
            self.cursor.global_update,
            self.schedule["total_updates"],
            self.schedule["warmup_updates"],
            self.schedule["base_learning_rate"],
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.current_learning_rate = lr
        self.cursor.global_update += 1
        return float(grad_norm), lr

    def _metrics(self, loss_sum, token_count, padded_token_count, grad_norm, lr, epoch, elapsed):
        metrics = {
            "loss": loss_sum / token_count,
            "grad_norm": grad_norm,
            "lr": lr,
            "update": self.cursor.global_update,
            "epoch": epoch + 1,
            "supervised_tokens": token_count,
            "padded_tokens": padded_token_count,
            "tokens_per_second": token_count / max(elapsed, 1e-9),
        }
        if self.device.type == "cuda":
            metrics.update(
                peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(self.device) / 2**30,
            )
        return metrics

    def train(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._print_run_summary()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        stopped = False
        for epoch in range(self.cursor.epoch, self.args.epochs):
            self.batch_sampler.set_epoch(epoch)
            start_record = self.cursor.next_record if epoch == self.cursor.epoch else 0
            self.batch_sampler.set_start_record(start_record)
            self.cursor.epoch = epoch
            self.cursor.next_record = start_record
            self.cursor.next_batch = math.ceil(start_record / self.args.batch_size)
            records_consumed = start_record
            pending_batches = supervised_tokens = padded_tokens = 0
            window_loss = 0.0
            window_start = time.monotonic()

            for local_batch_number, batch in enumerate(self.train_loader, 1):
                input_ids, targets, loss_mask = batch
                records_consumed += input_ids.size(0)
                batch_number = math.ceil(records_consumed / self.args.batch_size)
                batch_tokens = int(loss_mask.sum().item())
                if batch_tokens == 0:
                    raise ValueError("training batch has no supervised tokens")
                padded_tokens += input_ids.numel()
                with autocast_context(self.device, self.args.dtype):
                    loss_sum = forward_loss(
                        self.model,
                        input_ids.to(
                            self.device, non_blocking=self.device.type == "cuda"
                        ),
                        targets.to(
                            self.device, non_blocking=self.device.type == "cuda"
                        ),
                        loss_mask.to(
                            self.device, non_blocking=self.device.type == "cuda"
                        ),
                        self.args.loss_chunk_tokens,
                    )
                value = loss_sum.detach().item()
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite loss epoch={epoch} batch={batch_number}"
                    )
                self.scaler.scale(loss_sum).backward()
                window_loss += value
                supervised_tokens += batch_tokens
                pending_batches += 1
                del loss_sum, input_ids, targets, loss_mask

                is_epoch_end = records_consumed == len(self.train_dataset)
                if (
                    pending_batches != self.args.gradient_accumulation_steps
                    and not is_epoch_end
                ):
                    continue

                grad_norm, lr = self._optimizer_update(supervised_tokens)
                self.cursor.next_record = records_consumed
                self.cursor.next_batch = math.ceil(records_consumed / self.args.batch_size)
                epoch_complete = is_epoch_end
                stopped = bool(
                    self.args.limit_updates
                    and self.cursor.global_update >= self.args.limit_updates
                )
                metrics = self._metrics(
                    window_loss,
                    supervised_tokens,
                    padded_tokens,
                    grad_norm,
                    lr,
                    epoch,
                    time.monotonic() - window_start,
                )
                eval_due = epoch_complete or stopped or (
                    self.args.eval_interval
                    and self.cursor.global_update % self.args.eval_interval == 0
                )
                if eval_due:
                    metrics.update(
                        evaluate(
                            self.model,
                            self.validation_loader,
                            self.device,
                            self.args.dtype,
                            self.args.loss_chunk_tokens,
                        )
                    )
                self.last_metrics = metrics
                if self.cursor.global_update % self.args.log_interval == 0 or eval_due:
                    print(
                        "PRETRAIN_METRICS " + json.dumps(metrics, sort_keys=True),
                        flush=True,
                    )
                    if self.wandb is not None:
                        self.wandb.log(metrics, step=self.cursor.global_update)
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                if epoch_complete or (
                    self.args.save_interval
                    and self.cursor.global_update % self.args.save_interval == 0
                ):
                    self.save_progress()
                if epoch_complete:
                    self.save_progress(
                        self.output_dir
                        / f"mindlm_pretrain_{self.args.model_config}_epoch{epoch}.pt"
                    )

                pending_batches = supervised_tokens = padded_tokens = 0
                window_loss = 0.0
                window_start = time.monotonic()
                if stopped:
                    break
            if stopped:
                break

        self.save_progress()
        status = (
            "paused"
            if stopped and self.cursor.global_update < self.schedule["total_updates"]
            else "complete"
        )
        print(
            f"Pretrain {status} update={self.cursor.global_update}/{self.schedule['total_updates']}",
            flush=True,
        )
        if self.wandb is not None:
            self.wandb.finish()


def run_training(args):
    """Build a trainer and execute one complete or bounded run."""

    PackedPretrainTrainer(args).train()


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
