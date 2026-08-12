"""Shared configuration, loss, and checkpoint helpers for MindLM scripts."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

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


def masked_language_model_loss(logits, targets, loss_mask, aux_loss=None):
    """Return mean token loss over ``loss_mask`` plus an optional MoE loss."""
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    flat_mask = loss_mask.reshape(-1).to(dtype=token_loss.dtype)
    mask_total = flat_mask.sum()
    if mask_total.item() == 0:
        raise ValueError("A batch has no supervised tokens after truncation.")

    loss = (token_loss * flat_mask).sum() / mask_total
    if aux_loss is not None:
        loss = loss + aux_loss
    return loss


def extract_model_state(checkpoint):
    """Accept legacy state dicts and the standardized MindLM checkpoint format."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dict or a MindLM checkpoint dictionary.")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model state is not a state dictionary.")

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.removeprefix("module.")
        if key.startswith("_orig_mod."):
            key = key.removeprefix("_orig_mod.")
        normalized[key] = value
    return normalized


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
    return checkpoint if "model" in checkpoint else {}


def save_training_checkpoint(path, model, optimizer, scaler, config, epoch, step, epoch_complete, training_stage):
    """Save all state required to resume an epoch-boundary training run."""
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
    }
    torch.save(checkpoint, Path(path))
