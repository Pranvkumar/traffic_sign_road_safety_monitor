"""Class-conditional variational autoencoder (CVAE) in NumPy for 32x32 grayscale sign crops.

encoder:  [x (1024), onehot(c)] -> 256 ReLU -> (mu, logvar) in R^latent
decoder:  [z, onehot(c)]        -> 256 ReLU -> 1024 sigmoid
loss   :  BCE(x, x_hat) summed over pixels + beta * KL(q(z|x,c) || N(0, I)), averaged over the batch
Used only at training time to synthesise extra crops for rare classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..fundamentals.neurons import sigmoid


class CVAE:
    def __init__(self, n_classes: int, input_dim: int = 1024, hidden: int = 256, latent: int = 16,
                 beta: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.cfg = {"n_classes": n_classes, "input_dim": input_dim, "hidden": hidden, "latent": latent, "beta": beta}
        he = lambda i, o: rng.normal(0, np.sqrt(2 / i), (i, o))  # noqa: E731
        self.params = {
            "We": he(input_dim + n_classes, hidden), "be": np.zeros(hidden),
            "Wmu": he(hidden, latent) * 0.1, "bmu": np.zeros(latent),
            "Wlv": he(hidden, latent) * 0.1, "blv": np.zeros(latent),
            "Wd": he(latent + n_classes, hidden), "bd": np.zeros(hidden),
            "Wo": he(hidden, input_dim) * 0.1, "bo": np.zeros(input_dim),
        }

    def _onehot(self, y):
        o = np.zeros((len(y), self.cfg["n_classes"]))
        o[np.arange(len(y)), y] = 1
        return o

    def encode(self, x, y):
        xin = np.hstack([x, self._onehot(y)])
        a = xin @ self.params["We"] + self.params["be"]
        h = np.maximum(a, 0)
        return h @ self.params["Wmu"] + self.params["bmu"], h @ self.params["Wlv"] + self.params["blv"], (xin, a, h)

    def decode(self, z, y):
        zin = np.hstack([z, self._onehot(y)])
        a = zin @ self.params["Wd"] + self.params["bd"]
        h = np.maximum(a, 0)
        return sigmoid(h @ self.params["Wo"] + self.params["bo"]), (zin, a, h)

    def loss_and_grads(self, x, y, eps):
        """eps: standard-normal noise (N, latent), passed in so gradients can be checked."""
        n, beta = len(x), self.cfg["beta"]
        mu, lv, (xin, a1, h1) = self.encode(x, y)
        lv = np.clip(lv, -10, 10)
        std = np.exp(0.5 * lv)
        z = mu + std * eps
        xh, (zin, a2, h2) = self.decode(z, y)
        xc = np.clip(xh, 1e-7, 1 - 1e-7)
        recon = -np.sum(x * np.log(xc) + (1 - x) * np.log(1 - xc)) / n
        kl = -0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv)) / n
        g = {}
        d_out = (xh - x) / n                                   # dBCE/dlogit for sigmoid output
        g["Wo"], g["bo"] = h2.T @ d_out, d_out.sum(0)
        d = (d_out @ self.params["Wo"].T) * (a2 > 0)
        g["Wd"], g["bd"] = zin.T @ d, d.sum(0)
        dz = (d @ self.params["Wd"].T)[:, : self.cfg["latent"]]
        dmu = dz + beta * mu / n
        dlv = dz * eps * 0.5 * std + beta * 0.5 * (np.exp(lv) - 1) / n
        g["Wmu"], g["bmu"] = h1.T @ dmu, dmu.sum(0)
        g["Wlv"], g["blv"] = h1.T @ dlv, dlv.sum(0)
        dh = (dmu @ self.params["Wmu"].T + dlv @ self.params["Wlv"].T) * (a1 > 0)
        g["We"], g["be"] = xin.T @ dh, dh.sum(0)
        return float(recon + beta * kl), float(recon), float(kl), g

    def sample(self, y, rng: np.random.Generator):
        z = rng.normal(size=(len(y), self.cfg["latent"]))
        return self.decode(z, np.asarray(y))[0]

    def save(self, path, meta=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {"model": "CVAE", "hyperparameters": self.cfg, **(meta or {})}
        np.savez_compressed(path, __meta__=np.array(json.dumps(info, default=str)),
                            **{f"param__{k}": v for k, v in self.params.items()})
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {path}")
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["__meta__"]))
        m = cls(**meta["hyperparameters"])
        m.params = {k[7:]: d[k] for k in d.files if k.startswith("param__")}
        return m, meta
