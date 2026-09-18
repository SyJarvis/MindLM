"""Shared configuration, loss, and checkpoint helpers for MindLM scripts."""

import math
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.utils.checkpoint import checkpoint


def cosine_learning_rate(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.1):
    """Warmup followed by cosine decay, shared by all training entry points."""
    if warmup_steps < 0 or total_steps < 1 or base_lr <= 0:
        raise ValueError("total_steps must be positive, warmup_steps non-negative, and base_lr positive")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (base_lr - min_lr)

try:
    from .config import load_config
    from .modeling_mindlm import MindLMConfig
except ImportError:
    from config import load_config
    from modeling_mindlm import MindLMConfig


def build_model_config(name: str, tokenizer=None) -> MindLMConfig:
    """Build a config directly from JSON and align token ids with a tokenizer."""
    config = MindLMConfig(**load_config(name))
    if tokenizer is not None:
        config.vocab_size = len(tokenizer)
        if tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.bos_token_id is not None:
            config.bos_token_id = tokenizer.bos_token_id
        elif "<|endoftext|>" in getattr(tokenizer, "all_special_tokens", []):
            # Qwen3's tokenizer config leaves bos_token unset; its model config
            # uses end-of-text as the beginning token when one is needed.
            config.bos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if tokenizer.eos_token_id is not None:
            config.eos_token_id = tokenizer.eos_token_id
    return config


class EpochRandomSampler(Sampler):
    """Deterministically shuffle a dataset once per epoch for resumable training."""

    def __init__(self, data_source, seed=1337):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self):
        return len(self.data_source)


def chunked_masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor, chunk_tokens: int = 4096):
    """Return weighted mean CE and fp32 mask count, recomputing CE in backward."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1).float()
    denom = flat_mask.sum()
    if denom.item() == 0:
        return flat_logits[:0].sum().float(), denom

    def chunk_loss(values, labels, weights, selection):
        per = F.cross_entropy(values[selection].float(), labels[selection], reduction="none")
        return (per * weights[selection]).sum()

    loss_sum = logits.new_zeros((), dtype=torch.float32)
    idx = torch.nonzero(flat_mask > 0, as_tuple=False).flatten()
    for start in range(0, idx.numel(), chunk_tokens):
        sel = idx[start:start + chunk_tokens]
        loss_sum = loss_sum + checkpoint(
            chunk_loss, flat_logits, flat_targets, flat_mask, sel, use_reentrant=False,
        )
    return loss_sum / denom, denom


def masked_language_model_loss(logits, targets, loss_mask, chunk_tokens=4096):
    """Return mean token loss over ``loss_mask``."""
    token_loss, _ = chunked_masked_ce_loss(logits, targets, loss_mask, chunk_tokens)
    return token_loss


def load_model_checkpoint(model, checkpoint_path, map_location, allow_partial_load=False):
    """Load a model and return checkpoint metadata when it is available."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state_dict = extract_model_state(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=not allow_partial_load)
    if allow_partial_load and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "Warning: partially loaded checkpoint; "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint


def extract_model_state(checkpoint):
    """Extract model weights from a standardized MindLM checkpoint."""
    if not isinstance(checkpoint, dict):
        raise TypeError("MindLM checkpoint must be a dictionary.")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("MindLM checkpoint must contain a 'model' state dictionary.")

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model state is not a state dictionary.")

    normalized = {}
    for key, value in state_dict.items():
        normalized[key] = value
    return normalized


def save_training_checkpoint(path, model, optimizer, scaler, config, epoch, step, epoch_complete, training_stage, wandb_run_id=None, extra_state=None):
    """Atomically save standard training state plus optional trainer-specific state."""
    unwrapped_model = model.module if hasattr(model, "module") else model
    unwrapped_model = unwrapped_model._orig_mod if hasattr(unwrapped_model, "_orig_mod") else unwrapped_model
    checkpoint = {
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "step": step,
        "epoch_complete": epoch_complete,
        "training_stage": training_stage,
        "wandb_run_id": wandb_run_id,
    }
    if extra_state is not None:
        checkpoint["extra_state"] = extra_state
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def masked_lm_head_loss(lm_head, hidden_states, targets, loss_mask, chunk_tokens: int = 256, reduction="mean"):
    """Masked CE computed by applying ``lm_head`` per chunk of hidden states.

    Select supervised hidden states before the head, and checkpoint the head
    and fp32 CE together so vocabulary-sized intermediates are not retained.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    flat_h = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    mask_sum = flat_mask.sum()
    if mask_sum.item() == 0:
        zero = flat_h[:0].sum().float()
        for parameter in lm_head.parameters():
            zero = zero + parameter.reshape(-1)[:0].sum().float()
        return zero

    def chunk_loss(hidden, labels, weights, selection):
        per = F.cross_entropy(lm_head(hidden[selection]).float(), labels[selection], reduction="none")
        return (per * weights[selection]).sum()

    idx = torch.nonzero(flat_mask > 0, as_tuple=False).flatten()
    loss_sum = flat_h.new_zeros((), dtype=torch.float32)
    for start in range(0, idx.numel(), chunk_tokens):
        sel = idx[start:start + chunk_tokens]
        loss_sum = loss_sum + checkpoint(
            chunk_loss, flat_h, flat_targets, flat_mask, sel, use_reentrant=False,
        )
    return loss_sum if reduction == "sum" else loss_sum / mask_sum
