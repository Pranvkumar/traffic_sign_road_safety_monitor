"""Helpers shared by training scripts: cached GTSRB arrays and the experiment log."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from ..data.datasets import load_crop_arrays
from ..data.manifest import load_manifest
from ..utils.config import load_config, resolve
from ..utils.logging import get_logger
from ..utils.reproducibility import file_sha256

log = get_logger(__name__)

EXPERIMENT_LOG = "experiments/experiment_log.csv"
LOG_COLUMNS = ["run_id", "timestamp", "task", "model", "dataset", "manifest_sha256", "seed",
               "key_settings", "best_epoch", "val_metric", "val_value", "test_value", "checkpoint", "notes"]


def gtsrb_arrays(data_cfg_path: str = "configs/data.yaml", size: int = 32) -> dict[str, np.ndarray]:
    """Load (and cache as uint8) the GTSRB train/val/test crops defined by the split manifest."""
    cfg = load_config(data_cfg_path)["gtsrb"]
    manifest_path = resolve(cfg["manifest"])
    sha = file_sha256(manifest_path)
    cache = resolve(f"data/interim/gtsrb_{size}_{sha[:10]}.npz")
    if cache.exists():
        data = dict(np.load(cache))
        data["manifest_sha256"] = np.array(sha)
        return data
    log.info("Decoding GTSRB crops (first run only) ...")
    manifest = load_manifest(manifest_path)
    out = {}
    for split in ("train", "val", "test"):
        x, y, part = load_crop_arrays(manifest, resolve(cfg["root"]), split, size)
        out[f"X_{split}"] = (x * 255).round().astype(np.uint8)
        out[f"y_{split}"] = y
        out[f"group_{split}"] = np.array(part["physical_sign_id"].astype(str).tolist(), dtype=str)
        out[f"frame_{split}"] = part["frame_index"].astype(int).to_numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    out["manifest_sha256"] = np.array(sha)
    return out


def as_float(x_uint8: np.ndarray) -> np.ndarray:
    return x_uint8.astype(np.float32) / 255.0


def log_experiment(row: dict) -> None:
    path = resolve(EXPERIMENT_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row})


def class_names(taxonomy_path: str = "configs/label_taxonomy.yaml", section: str = "classification") -> dict[int, str]:
    return {int(k): v for k, v in load_config(taxonomy_path)[section].items()}
