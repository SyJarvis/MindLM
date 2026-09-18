"""Shared helpers for MindLM evaluation scripts.

Deliberately kept out of ``training_utils``: a running job freezes
``source_sha256`` for the training modules (config.py, dataset.py,
modeling_mindlm.py, pretrain.py, training_utils.py) and refuses to resume when
they change, so evaluation-only helpers must live elsewhere.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Every current model config uses the Qwen3 tokenizer and the GDN V3 model
# contract.
DEFAULT_TOKENIZER_DIR = "qwen3_tokenizer"


def supported_configs() -> tuple:
    """Model config names accepted by the training and evaluation entry points.

    Derived from ``config.SUPPORTED_CONFIGS`` so a new config only has to be
    registered once instead of being added to every script's argparse whitelist.
    """
    from config import SUPPORTED_CONFIGS

    return tuple(SUPPORTED_CONFIGS)


def default_tokenizer_path(override: str = None) -> str:
    """Tokenizer directory to use, unless the caller passed an explicit one."""
    if override:
        return override
    return str(PROJECT_ROOT / DEFAULT_TOKENIZER_DIR)


def load_checkpoint_with_tokenizer_check(model, checkpoint_path: str, device, tokenizer):
    """Load a V3 checkpoint, turning a vocabulary mismatch into an actionable error.

    ``build_model_config`` adopts ``len(tokenizer)`` as the model vocabulary, so
    running a checkpoint with the wrong tokenizer otherwise surfaces as a raw
    ``RuntimeError: size mismatch for tok_embeddings.weight`` from
    ``load_state_dict``. Returns standardized checkpoint metadata.
    """
    from training_utils import load_model_checkpoint

    try:
        return load_model_checkpoint(model, checkpoint_path, device)
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        raise SystemExit(
            f"checkpoint/tokenizer mismatch while loading {checkpoint_path}:\n{exc}\n"
            f"The model was built for a tokenizer with {len(tokenizer)} tokens. "
            "Pass --tokenizer_path pointing at the tokenizer this checkpoint was trained with."
        ) from None


@contextlib.contextmanager
def inference_context(device) -> Iterator[None]:
    """``no_grad`` plus BF16 autocast on CUDA.

    The 0.2b_gdn config selects flash-attn backends that reject fp32 activations,
    so evaluation has to run under the same precision as training. On CPU the
    autocast is disabled and behaviour is unchanged.
    """
    import torch

    on_cuda = str(device).startswith("cuda")
    with torch.no_grad(), torch.autocast(
        device_type="cuda" if on_cuda else "cpu",
        dtype=torch.bfloat16 if on_cuda else torch.float32,
        enabled=on_cuda,
    ):
        yield
