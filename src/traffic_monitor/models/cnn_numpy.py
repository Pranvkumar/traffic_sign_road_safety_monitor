"""A small convolutional network implemented from scratch in NumPy (im2col).

Architecture (default, 32x32x3 input):
    conv5x5(3->16, pad 2) -> ReLU -> maxpool2
    conv3x3(16->32, pad 1) -> ReLU -> maxpool2
    flatten(32*8*8) -> dense(128) -> ReLU -> dense(n_classes)

PyTorch was not available in the build environment, so this stands in for the
ResNet/GoogLeNet comparison in the brief; see docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .mlp_numpy import softmax


def im2col(x: np.ndarray, k: int, pad: int) -> tuple[np.ndarray, tuple]:
    """x (N,C,H,W) -> columns (N*oh*ow, C*k*k) for stride-1 convolution."""
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (k, k), axis=(2, 3))        # N,C,oh,ow,k,k
    n, c, oh, ow = win.shape[:4]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * oh * ow, c * k * k)
    return cols, (n, c, oh, ow, xp.shape)


def col2im(dcols: np.ndarray, meta: tuple, k: int, pad: int) -> np.ndarray:
    n, c, oh, ow, padded_shape = meta
    d = dcols.reshape(n, oh, ow, c, k, k).transpose(0, 3, 1, 2, 4, 5)  # N,C,oh,ow,k,k
    dx = np.zeros(padded_shape, dtype=dcols.dtype)
    for i in range(k):
        for j in range(k):
            dx[:, :, i:i + oh, j:j + ow] += d[:, :, :, :, i, j]
    return dx[:, :, pad:padded_shape[2] - pad, pad:padded_shape[3] - pad] if pad else dx


class SmallCNN:
    def __init__(self, n_classes: int, in_ch: int = 3, size: int = 32, c1: int = 16, c2: int = 32,
                 hidden: int = 128, seed: int = 0, dtype=np.float32):
        if size % 4:
            raise ValueError("Input size must be divisible by 4")
        rng = np.random.default_rng(seed)
        self.cfg = {"n_classes": n_classes, "in_ch": in_ch, "size": size, "c1": c1, "c2": c2, "hidden": hidden}
        flat = c2 * (size // 4) ** 2
        he = lambda fan_in, shape: (rng.normal(0, np.sqrt(2 / fan_in), shape)).astype(dtype)  # noqa: E731
        self.params = {
            "K1": he(in_ch * 25, (in_ch * 25, c1)), "b1": np.zeros(c1, dtype),
            "K2": he(c1 * 9, (c1 * 9, c2)), "b2": np.zeros(c2, dtype),
            "W3": he(flat, (flat, hidden)), "b3": np.zeros(hidden, dtype),
            "W4": he(hidden, (hidden, n_classes)), "b4": np.zeros(n_classes, dtype),
        }

    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    @staticmethod
    def _pool(x):
        n, c, h, w = x.shape
        r = x.reshape(n, c, h // 2, 2, w // 2, 2)
        out = r.max(axis=(3, 5))
        mask = (r == out[:, :, :, None, :, None])
        return out, mask

    @staticmethod
    def _unpool(d, mask):
        n, c, h2, _, w2, _ = mask.shape
        dd = mask * d[:, :, :, None, :, None]
        return dd.reshape(n, c, h2 * 2, w2 * 2)

    def _conv(self, x, kname, bname, k, pad):
        cols, meta = im2col(x, k, pad)
        out = cols @ self.params[kname] + self.params[bname]
        n, _, oh, ow, _ = meta
        return out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2), (cols, meta)

    def forward(self, x_nhwc: np.ndarray):
        x = x_nhwc.transpose(0, 3, 1, 2).astype(self.params["K1"].dtype)
        z1, c1 = self._conv(x, "K1", "b1", 5, 2)
        a1 = np.maximum(z1, 0)
        p1, m1 = self._pool(a1)
        z2, c2 = self._conv(p1, "K2", "b2", 3, 1)
        a2 = np.maximum(z2, 0)
        p2, m2 = self._pool(a2)
        f = p2.reshape(len(x), -1)
        z3 = f @ self.params["W3"] + self.params["b3"]
        a3 = np.maximum(z3, 0)
        out = a3 @ self.params["W4"] + self.params["b4"]
        return out, (c1, z1, m1, c2, z2, m2, p2.shape, f, z3, a3)

    def loss_and_grads(self, x, y):
        out, (c1, z1, m1, c2, z2, m2, p2_shape, f, z3, a3) = self.forward(x)
        n = len(x)
        p = softmax(out.astype(np.float64))
        loss = float(-np.mean(np.log(p[np.arange(n), y] + 1e-12)))
        d = p
        d[np.arange(n), y] -= 1
        d = (d / n).astype(out.dtype)
        g = {"W4": a3.T @ d, "b4": d.sum(0)}
        d = (d @ self.params["W4"].T) * (z3 > 0)
        g["W3"], g["b3"] = f.T @ d, d.sum(0)
        d = (d @ self.params["W3"].T).reshape(p2_shape)
        d = self._unpool(d, m2) * (z2 > 0)                    # N,C2,H,W
        dflat = d.transpose(0, 2, 3, 1).reshape(-1, d.shape[1])
        g["K2"], g["b2"] = c2[0].T @ dflat, dflat.sum(0)
        d = col2im(dflat @ self.params["K2"].T, c2[1], 3, 1)
        d = self._unpool(d, m1) * (z1 > 0)
        dflat = d.transpose(0, 2, 3, 1).reshape(-1, d.shape[1])
        g["K1"], g["b1"] = c1[0].T @ dflat, dflat.sum(0)
        return loss, g

    def predict_proba(self, x, batch: int = 512):
        outs = [self.forward(x[i:i + batch])[0] for i in range(0, len(x), batch)]
        return softmax(np.concatenate(outs).astype(np.float64))

    def predict(self, x):
        return self.predict_proba(x).argmax(1)

    def feature_maps(self, x_nhwc: np.ndarray) -> np.ndarray:
        """First-layer ReLU feature maps (N, C1, H, W) for visualisation."""
        x = x_nhwc.transpose(0, 3, 1, 2).astype(self.params["K1"].dtype)
        z1, _ = self._conv(x, "K1", "b1", 5, 2)
        return np.maximum(z1, 0)

    def save(self, path, meta: dict | None = None, extra_arrays: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {"model": "SmallCNN", "hyperparameters": self.cfg, **(meta or {})}
        arrays = {f"param__{k}": v for k, v in self.params.items()}
        arrays.update({f"extra__{k}": v for k, v in (extra_arrays or {}).items()})
        np.savez_compressed(path, __meta__=np.array(json.dumps(info, default=str)), **arrays)
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["__meta__"]))
        model = cls(**meta["hyperparameters"])
        for k in data.files:
            if k.startswith("param__"):
                model.params[k[7:]] = data[k]
        extras = {k[7:]: data[k] for k in data.files if k.startswith("extra__")}
        return model, extras, meta
