"""Training entry points for MindLM."""

from .pretrain_legacy import run_legacy
from .pretrain_v3 import run_v3

__all__ = ["run_legacy", "run_v3"]
