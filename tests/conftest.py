"""Tiny generated fixtures — tests never need the real datasets or a GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def gtsrb_folder(tmp_path):
    """3 classes x 2 tracks x 3 frames of 32x32 PNG crops + 2 test images per class."""
    rng = np.random.default_rng(0)
    root = tmp_path / "gtsrb"
    for c in range(3):
        d = root / "train" / f"{c:05d}"
        d.mkdir(parents=True)
        for t in range(2):
            for f in range(3):
                img = np.full((32, 32, 3), 40 * (c + 1), np.uint8) + rng.integers(0, 20, (32, 32, 3), dtype=np.uint8)
                cv2.imwrite(str(d / f"{t:05d}_{f:05d}.png"), img)
        d = root / "test" / f"{c:05d}"
        d.mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(d / f"{c * 10 + i:05d}.png"), rng.integers(0, 255, (30, 30, 3), dtype=np.uint8))
    return root


@pytest.fixture
def yolo_folder(tmp_path):
    """2 sequences x 2 frames of 64x48 images with YOLO labels (one with an empty label file)."""
    root = tmp_path / "idd"
    for s in range(2):
        for f in range(2):
            img_dir = root / "JPEGImages" / "frontFar" / f"SEQ{s}"
            lab_dir = root / "labels" / "frontFar" / f"SEQ{s}"
            img_dir.mkdir(parents=True, exist_ok=True)
            lab_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_dir / f"{f:06d}_r.jpg"), np.full((48, 64, 3), 100 + s * 50 + f, np.uint8))
            text = "" if (s, f) == (1, 1) else "0 0.5 0.5 0.25 0.25\n"
            (lab_dir / f"{f:06d}_r.txt").write_text(text)
    return root
