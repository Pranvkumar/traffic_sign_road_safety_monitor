"""Class-agnostic traffic-sign detector: multi-scale HOG sliding window + window scorer + NMS.

This is a classical detector (Dalal–Triggs style), used because PyTorch/YOLO could not be
installed in the build environment. The scorer is pluggable: a linear logistic model or
a NumPy MLP. Both are trained on windows cut from the IDD-derived training images.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..models.mlp_numpy import MLP, Standardizer

WIN = 24
if not hasattr(cv2, "HOGDescriptor"):
    raise ImportError(f"OpenCV {cv2.__version__} has no cv2.HOGDescriptor. Install the tested version: "
                      "pip uninstall -y opencv-python-headless && pip install opencv-contrib-python-headless==5.0.0.93")
_HOG = cv2.HOGDescriptor((WIN, WIN), (12, 12), (6, 6), (6, 6), 9)
DEFAULT_SIGN_SIZES = (16, 24, 36, 54, 81, 120)   # sign side lengths (px) searched for


CELL = 8                       # colour features: mean colour-opponent value in a 3x3 grid of 8x8 cells
N_FEATURES = 324 + 27


def colour_maps(bgr: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 BGR -> (H,W,3) float32 [redness, blueness, yellowness] in [0,1].
    Sign borders in India (and most countries) are red, blue or yellow."""
    f = bgr.astype(np.float32) / 255
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    return np.dstack([np.clip(r - np.maximum(g, b), 0, 1), np.clip(b - np.maximum(r, g), 0, 1),
                      np.clip(np.minimum(r, g) - b, 0, 1)])


def _cell_means(maps: np.ndarray, x0: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Mean of each colour map in the 3x3 grid of CELLxCELL cells of windows at (x0, y0)."""
    integ = cv2.integral(maps)                       # (H+1, W+1, 3), float64
    out = []
    for i in range(3):
        for j in range(3):
            ya, yb, xa, xb = y0 + CELL * i, y0 + CELL * (i + 1), x0 + CELL * j, x0 + CELL * (j + 1)
            out.append((integ[yb, xb] - integ[ya, xb] - integ[yb, xa] + integ[ya, xa]) / CELL ** 2)
    return np.concatenate(out, axis=1).astype(np.float32)          # (K, 27)


def window_features(patches_bgr: np.ndarray) -> np.ndarray:
    """(N,24,24,3) uint8 BGR windows -> (N, 351) = HOG(gray) + 3x3 colour-opponent cell means."""
    hog = np.stack([_HOG.compute(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)).ravel() for p in patches_bgr])
    zeros = np.zeros(1, int)
    col = np.stack([_cell_means(colour_maps(p), zeros, zeros)[0] for p in patches_bgr])
    return np.hstack([hog, col]).astype(np.float32)


def crop_square(img: np.ndarray, box: np.ndarray, scale: float = 1.0) -> np.ndarray | None:
    """Square crop around a box (side = max(w,h)*scale), resized to the window size."""
    x1, y1, x2, y2 = box
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    if s < 4 or a < 0 or b < 0 or a + s > img.shape[1] or b + s > img.shape[0]:
        return None
    return cv2.resize(img[b:b + s, a:a + s], (WIN, WIN), interpolation=cv2.INTER_AREA)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter)
        order = rest[iou < iou_thr]
    return np.array(keep, dtype=int)


def dense_windows(img_bgr: np.ndarray, sign_sizes=DEFAULT_SIGN_SIZES, stride: int = 6):
    """Yield (features (K,351), boxes (K,4) in original pixels) for every scale.
    Features match `window_features` on a crop of the same region."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]
    for size in sign_sizes:
        s = WIN / size
        sw, sh = int(round(w * s)), int(round(h * s))
        if sw < WIN or sh < WIN:
            continue
        small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hog = _HOG.compute(g, winStride=(stride, stride)).reshape(-1, _HOG.getDescriptorSize())
        nx, ny = (sw - WIN) // stride + 1, (sh - WIN) // stride + 1
        ys, xs = np.mgrid[0:ny, 0:nx]
        x0, y0 = xs.ravel() * stride, ys.ravel() * stride
        col = _cell_means(colour_maps(small), x0, y0)
        boxes = np.stack([x0 / s, y0 / s, (x0 + WIN) / s, (y0 + WIN) / s], axis=1)
        yield np.hstack([hog[: len(boxes)], col]), boxes


class WindowScorer:
    """Wraps either a logistic model (w, b) or a NumPy MLP, with its feature standardiser."""

    def __init__(self, kind: str, std: Standardizer, w: np.ndarray | None = None, b: float = 0.0,
                 mlp: MLP | None = None):
        if kind not in ("logreg", "mlp"):
            raise ValueError(kind)
        self.kind, self.std, self.w, self.b, self.mlp = kind, std, w, b, mlp

    def score(self, feats: np.ndarray) -> np.ndarray:
        f = self.std.transform(feats)
        if self.kind == "logreg":
            z = f @ self.w + self.b
            return 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        return self.mlp.predict_proba(f)[:, 1]


class SignDetector:
    def __init__(self, scorer: WindowScorer, score_threshold: float = 0.5, nms_iou: float = 0.3,
                 max_detections: int = 20, sign_sizes=DEFAULT_SIGN_SIZES, stride: int = 6):
        self.scorer, self.score_threshold, self.nms_iou = scorer, score_threshold, nms_iou
        self.max_detections, self.sign_sizes, self.stride = max_detections, sign_sizes, stride

    def detect(self, img_bgr: np.ndarray, score_threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (boxes xyxy (N,4), scores (N,)) after NMS. Empty arrays if nothing found."""
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image")
        thr = self.score_threshold if score_threshold is None else score_threshold
        all_b, all_s = [], []
        for feats, boxes in dense_windows(img_bgr, self.sign_sizes, self.stride):
            s = self.scorer.score(feats)
            # Filtering at the threshold *before* NMS gives identical output: in greedy NMS a
            # lower-scoring window can never suppress a higher-scoring one.
            keep = s >= thr
            all_b.append(boxes[keep])
            all_s.append(s[keep])
        if not all_b or sum(len(b) for b in all_b) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        boxes, scores = np.concatenate(all_b), np.concatenate(all_s)
        keep = nms(boxes, scores, self.nms_iou)[: self.max_detections]
        return boxes[keep], scores[keep]

    # ---------- persistence ----------
    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sc = self.scorer
        info = {"model": "HOGSlidingWindowDetector", "scorer": sc.kind, "window": WIN,
                "hog": "win24 block12 stride6 cell6 bins9 on grayscale", "sign_sizes": list(self.sign_sizes),
                "stride": self.stride, "score_threshold": self.score_threshold, "nms_iou": self.nms_iou,
                "max_detections": self.max_detections, **(meta or {})}
        arrays = {"mean": sc.std.mean, "std": sc.std.std}
        if sc.kind == "logreg":
            arrays.update(w=sc.w, b=np.array(sc.b))
        else:
            info["mlp_hyperparameters"] = sc.mlp.hyperparameters()
            arrays.update({f"mlp__{k}": v for k, v in sc.mlp.params.items()})
        np.savez_compressed(path, __meta__=np.array(json.dumps(info, default=str)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SignDetector":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Detector checkpoint not found: {path}. Run `train-detector` first.")
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["__meta__"]))
        std = Standardizer(d["mean"], d["std"])
        if info["scorer"] == "logreg":
            scorer = WindowScorer("logreg", std, d["w"], float(d["b"]))
        else:
            hp = info["mlp_hyperparameters"]
            mlp = MLP(hp["sizes"], hp["activation"], hp["init"], hp["loss"], hp["weight_decay"])
            mlp.params = {k[5:]: d[k] for k in d.files if k.startswith("mlp__")}
            scorer = WindowScorer("mlp", std, mlp=mlp)
        det = cls(scorer, info["score_threshold"], info["nms_iou"], info["max_detections"],
                  tuple(info["sign_sizes"]), info["stride"])
        det.meta = info
        return det
