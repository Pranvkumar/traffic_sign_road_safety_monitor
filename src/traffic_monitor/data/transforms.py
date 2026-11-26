"""Shared preprocessing. The SAME functions are used for training and inference,
and their parameters are saved with each checkpoint."""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_CROP_SIZE = 32


def preprocess_crop(img_bgr: np.ndarray, size: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    """BGR uint8 crop -> RGB float32 in [0, 1] of shape (size, size, 3)."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image passed to preprocess_crop")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_gray(x: np.ndarray) -> np.ndarray:
    """(…, H, W, 3) RGB float -> (…, H, W) luminance."""
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def augment_crop(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild, label-preserving training augmentation (no horizontal flips: they can
    change sign meaning). Brightness/contrast jitter, small shift, Gaussian noise."""
    out = x.copy()
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)
    dx, dy = rng.integers(-2, 3, size=2)
    out = np.roll(out, (int(dy), int(dx)), axis=(0, 1))
    out = out + rng.normal(0, 0.02, size=out.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
