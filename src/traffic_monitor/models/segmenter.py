"""Road-surface segmentation with a per-pixel classifier on hand-crafted features.

Output class definition (project): ``road_surface`` = comma10k {road, lane_markings};
everything else is ``not_road``; ``my_car`` (ego bonnet/mount) is ignored in training and metrics.

A U-Net/DeepLab could not be trained in the build environment (no PyTorch), so this is a
compact, explainable baseline: each pixel of a 1/8-resolution image is described by colour,
position, local texture and multi-scale context, then scored by logistic regression or a
NumPy MLP. The probability map is upsampled to full resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .mlp_numpy import MLP, Standardizer

DOWNSCALE = 8
FEATURE_NAMES = ["r", "g", "b", "h", "s", "v", "y", "x", "gray_mean5", "gray_std5", "sobel_mag", "laplacian",
                 "r_ctx9", "g_ctx9", "b_ctx9", "r_ctx25", "g_ctx25", "b_ctx25"]
IGNORE = 255


def colour_mask_to_ids(mask_rgb: np.ndarray, colour_map: dict[str, str]) -> np.ndarray:
    """RGB colour mask -> project IDs: 1 road_surface, 0 not_road, 255 ignore."""
    ids = np.zeros(mask_rgb.shape[:2], np.uint8)
    lut = {"road": 1, "lane_markings": 1, "undrivable": 0, "movable": 0, "my_car": IGNORE}
    matched = np.zeros(mask_rgb.shape[:2], bool)
    for hexcol, name in colour_map.items():
        rgb = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        m = np.all(mask_rgb == rgb, axis=2)
        ids[m] = lut[name]
        matched |= m
    ids[~matched] = IGNORE          # unknown colours are never forced into a class
    return ids


def pixel_features(img_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (H'*W', 18) features at 1/DOWNSCALE resolution and the small shape (H', W')."""
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (w // DOWNSCALE, h // DOWNSCALE), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32) / np.array([180, 255, 255], np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    mean5 = cv2.blur(gray, (5, 5))
    std5 = np.sqrt(np.clip(cv2.blur(gray ** 2, (5, 5)) - mean5 ** 2, 0, None))
    sob = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    ctx9, ctx25 = cv2.blur(rgb, (9, 9)), cv2.blur(rgb, (25, 25))
    f = np.dstack([rgb, hsv, yy / sh, xx / sw, mean5, std5, sob, lap, ctx9, ctx25])
    return f.reshape(-1, f.shape[2]), (sh, sw)


class RoadSegmenter:
    def __init__(self, kind: str, std: Standardizer, w=None, b: float = 0.0, mlp: MLP | None = None,
                 threshold: float = 0.5):
        self.kind, self.std, self.w, self.b, self.mlp, self.threshold = kind, std, w, b, mlp, threshold

    def predict_proba_small(self, img_bgr: np.ndarray) -> np.ndarray:
        f, shape = pixel_features(img_bgr)
        z = self.std.transform(f)
        if self.kind == "logreg":
            p = 1 / (1 + np.exp(-np.clip(z @ self.w + self.b, -50, 50)))
        else:
            p = self.mlp.predict_proba(z)[:, 1]
        return p.reshape(shape)

    def predict_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Full-resolution uint8 mask: 1 road_surface, 0 not_road."""
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image")
        p = self.predict_proba_small(img_bgr)
        full = cv2.resize(p, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return (full >= self.threshold).astype(np.uint8)

    def save(self, path, meta=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {"model": "RoadSegmenter", "kind": self.kind, "threshold": self.threshold, "downscale": DOWNSCALE,
                "features": FEATURE_NAMES, "classes": {0: "not_road", 1: "road_surface"}, **(meta or {})}
        arrays = {"mean": self.std.mean, "std": self.std.std}
        if self.kind == "logreg":
            arrays.update(w=self.w, b=np.array(self.b))
        else:
            info["mlp_hyperparameters"] = self.mlp.hyperparameters()
            arrays.update({f"mlp__{k}": v for k, v in self.mlp.params.items()})
        np.savez_compressed(path, __meta__=np.array(json.dumps(info, default=str)), **arrays)
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Segmenter checkpoint not found: {path}. Run `train-segmenter` first.")
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["__meta__"]))
        std = Standardizer(d["mean"], d["std"])
        if info["kind"] == "logreg":
            return cls("logreg", std, d["w"], float(d["b"]), threshold=info["threshold"])
        hp = info["mlp_hyperparameters"]
        mlp = MLP(hp["sizes"], hp["activation"], hp["init"], hp["loss"], hp["weight_decay"])
        mlp.params = {k[5:]: d[k] for k in d.files if k.startswith("mlp__")}
        return cls("mlp", std, mlp=mlp, threshold=info["threshold"])
