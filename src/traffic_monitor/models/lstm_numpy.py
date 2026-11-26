"""Single-layer LSTM sequence classifier in NumPy with full backprop-through-time.

Input  X: (N, T, D) ordered per-frame feature vectors
Mask   M: (N, T) 1 = frame present, 0 = missing/padded (state is carried over)
Output logits from the final hidden state -> softmax over classes.
Gate order in the stacked weight matrices: input, forget, output, candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..fundamentals.neurons import sigmoid
from .mlp_numpy import softmax


class LSTMClassifier:
    def __init__(self, input_dim: int, hidden: int, n_classes: int, seed: int = 0, weight_decay: float = 0.0):
        rng = np.random.default_rng(seed)
        self.cfg = {"input_dim": input_dim, "hidden": hidden, "n_classes": n_classes, "weight_decay": weight_decay}
        h = hidden
        self.params = {
            "Wx": rng.normal(0, np.sqrt(1 / input_dim), (input_dim, 4 * h)),
            "Wh": rng.normal(0, np.sqrt(1 / h), (h, 4 * h)),
            "b": np.zeros(4 * h),
            "Wy": rng.normal(0, np.sqrt(1 / h), (h, n_classes)),
            "by": np.zeros(n_classes),
        }
        self.params["b"][h:2 * h] = 1.0  # forget-gate bias 1: remember by default

    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None):
        n, t_len, _ = x.shape
        h_dim = self.cfg["hidden"]
        mask = np.ones((n, t_len)) if mask is None else mask
        h, c = np.zeros((n, h_dim)), np.zeros((n, h_dim))
        cache = []
        for t in range(t_len):
            a = x[:, t] @ self.params["Wx"] + h @ self.params["Wh"] + self.params["b"]
            i, f, o = sigmoid(a[:, :h_dim]), sigmoid(a[:, h_dim:2 * h_dim]), sigmoid(a[:, 2 * h_dim:3 * h_dim])
            g = np.tanh(a[:, 3 * h_dim:])
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            m = mask[:, t:t + 1]
            cache.append((x[:, t], h, c, i, f, o, g, c_new, m))
            h, c = m * h_new + (1 - m) * h, m * c_new + (1 - m) * c
        logits = h @ self.params["Wy"] + self.params["by"]
        return logits, (cache, h)

    def loss_and_grads(self, x, y, mask=None):
        logits, (cache, h_last) = self.forward(x, mask)
        n = len(x)
        p = softmax(logits)
        loss = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
        wd = self.cfg["weight_decay"]
        loss += 0.5 * wd * sum(np.sum(self.params[k] ** 2) for k in ("Wx", "Wh", "Wy"))
        d = p.copy()
        d[np.arange(n), y] -= 1
        d /= n
        g = {k: np.zeros_like(v) for k, v in self.params.items()}
        g["Wy"], g["by"] = h_last.T @ d + wd * self.params["Wy"], d.sum(0)
        dh, dc = d @ self.params["Wy"].T, np.zeros_like(h_last)
        h_dim = self.cfg["hidden"]
        for xt, h_prev, c_prev, i, f, o, gg, c_new, m in reversed(cache):
            dh_new, dc_new = dh * m, dc * m
            dh_carry, dc_carry = dh * (1 - m), dc * (1 - m)
            tanh_c = np.tanh(c_new)
            do = dh_new * tanh_c
            dc_tot = dc_new + dh_new * o * (1 - tanh_c ** 2)
            di, df, dg = dc_tot * gg, dc_tot * c_prev, dc_tot * i
            da = np.hstack([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - gg ** 2)])
            g["Wx"] += xt.T @ da
            g["Wh"] += h_prev.T @ da
            g["b"] += da.sum(0)
            dh = da @ self.params["Wh"].T + dh_carry
            dc = dc_tot * f + dc_carry
        g["Wx"] += wd * self.params["Wx"]
        g["Wh"] += wd * self.params["Wh"]
        assert h_dim == self.params["Wh"].shape[0]
        return float(loss), g

    def predict_proba(self, x, mask=None, batch: int = 2048):
        outs = [self.forward(x[i:i + batch], None if mask is None else mask[i:i + batch])[0]
                for i in range(0, len(x), batch)]
        return softmax(np.concatenate(outs))

    def predict(self, x, mask=None):
        return self.predict_proba(x, mask).argmax(1)

    def save(self, path, meta=None, extra_arrays=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"param__{k}": v for k, v in self.params.items()}
        arrays.update({f"extra__{k}": v for k, v in (extra_arrays or {}).items()})
        info = {"model": "LSTMClassifier", "hyperparameters": self.cfg, **(meta or {})}
        np.savez_compressed(path, __meta__=np.array(json.dumps(info, default=str)), **arrays)
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LSTM checkpoint not found: {path}")
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["__meta__"]))
        m = cls(**meta["hyperparameters"])
        m.params = {k[7:]: d[k] for k in d.files if k.startswith("param__")}
        return m, {k[7:]: d[k] for k in d.files if k.startswith("extra__")}, meta
