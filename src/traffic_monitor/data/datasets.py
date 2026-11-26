"""Dataset adapters that turn manifests into NumPy arrays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .annotations import read_yolo_file, yolo_to_xyxy
from .transforms import preprocess_crop


def load_crop_arrays(manifest: pd.DataFrame, root: str | Path, split: str, size: int = 32,
                     limit_per_class: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load sign crops for one split as (N, size, size, 3) float32 and integer labels."""
    root = Path(root)
    part = manifest[manifest["split"] == split]
    if part.empty:
        raise ValueError(f"No samples in split '{split}'")
    if limit_per_class:
        part = (part.groupby("label_id", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), limit_per_class), random_state=seed)))
    images, labels = [], []
    for rel, label in zip(part["relative_path"], part["label_id"]):
        img = cv2.imread(str(root / rel), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Could not decode image {root / rel}")
        images.append(preprocess_crop(img, size))
        labels.append(int(label))
    return np.stack(images), np.array(labels, dtype=np.int64), part.reset_index(drop=True)


def load_detection_sample(root: str | Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (BGR image, (N,4) pixel xyxy boxes) for one manifest row."""
    root = Path(root)
    img = cv2.imread(str(root / row["relative_path"]), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Could not decode image {root / row['relative_path']}")
    if not row["annotation_path"]:
        raise FileNotFoundError(f"No annotation for {row['sample_id']}")
    yolo = read_yolo_file(root / row["annotation_path"])
    h, w = img.shape[:2]
    return img, yolo_to_xyxy(yolo[:, 1:], w, h)
