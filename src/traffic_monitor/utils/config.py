"""Load YAML configuration files and resolve paths relative to the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file. Raises FileNotFoundError with a clear message."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level")
    return cfg


def resolve(path: str | Path) -> Path:
    """Return an absolute path; relative paths are taken from the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def require_keys(cfg: dict[str, Any], keys: list[str], name: str = "config") -> None:
    """Validate that required keys are present."""
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"{name} is missing required keys: {missing}")
