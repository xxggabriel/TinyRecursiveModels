from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "global_config.yaml"


def _get_config_path() -> Path:
    """Resolve the path to the global configuration file.

    TRM_GLOBAL_CONFIG env var allows overriding the default path for custom deployments.
    """
    return Path(os.environ.get("TRM_GLOBAL_CONFIG", DEFAULT_CONFIG_PATH))


@lru_cache()
def load_global_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the global configuration YAML once and cache it for reuse."""
    cfg_path = Path(config_path) if config_path else _get_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Global configuration file not found at {cfg_path}.")

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    return data


def apply_global_credentials(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Populate runtime environment variables from the credentials section."""
    cfg = config or load_global_config()
    credentials = cfg.get("credentials", {})

    wandb_api_key = credentials.get("wandb_api_key")
    if wandb_api_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = str(wandb_api_key)

    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(wandb_cfg["entity"]))

    huggingface_token = credentials.get("huggingface_token")
    if huggingface_token and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = str(huggingface_token)

    kaggle_cfg = credentials.get("kaggle", {})
    if kaggle_cfg.get("username"):
        os.environ.setdefault("KAGGLE_USERNAME", str(kaggle_cfg["username"]))
    if kaggle_cfg.get("api_key"):
        os.environ.setdefault("KAGGLE_KEY", str(kaggle_cfg["api_key"]))

    return cfg


def get_global_setting(*keys: str, default: Any = None) -> Any:
    """Access nested global configuration values using an ordered list of keys."""
    cfg = load_global_config()
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default
