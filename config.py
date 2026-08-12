"""Model configuration loading helpers."""

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "config"
SUPPORTED_CONFIGS = ("mindlm_0.1b", "mindlm_0.1b_moe", "mindlm_0.8b")


def load_config(name: str) -> dict:
    """Load one of the supported model configurations."""
    if name not in SUPPORTED_CONFIGS:
        supported = ", ".join(SUPPORTED_CONFIGS)
        raise ValueError(f"Unsupported model config '{name}'. Choose one of: {supported}")

    model_config_path = CONFIG_DIR / f"{name}.json"
    with model_config_path.open("r", encoding="utf-8") as file:
        return json.load(file)
