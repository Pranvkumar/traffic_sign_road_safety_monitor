"""From-scratch multilayer perceptron in NumPy: forward pass, loss, manual backprop,
optimizers (GD/SGD, Momentum, Nesterov, Adam), gradient checking and checkpoints.

Parameters (weights/biases) are learned; hyperparameters (layer sizes, learning
rate, batch size, weight decay, init scheme, …) are set by the user and stored
separately in the checkpoint's ``hyperparameters`` record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from ..fundamentals.neurons import relu, relu_grad, sigmoid, sigmoid_grad, tanh, tanh_grad

ACTIVATIONS = {"relu": (relu, relu_grad), "sigmoid": (sigmoid, sigmoid_grad), "tanh": (tanh, tanh_grad)}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Standardizer:
    """Feature standardisation fitted on training data only (saved with the model)."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return x
        return (x - self.mean) / self.std


class MLP:
    def __init__(self, sizes: list[int], activation: str = "relu", init: str = "he",
                 loss: str = "cross_entropy", weight_decay: float = 0.0, seed: int = 0):
        if len(sizes) < 2:
            raise ValueError("sizes needs at least input and output dimension")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation}")
        if loss not in ("cross_entropy", "mse"):
            raise ValueError(f"Unknown loss {loss}")
        self.sizes, self.activation, self.init, self.loss = list(sizes), activation, init, loss
        self.weight_decay = weight_decay
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
            if init == "he":
                scale = np.sqrt(2.0 / n_in)
            elif init == "xavier":
                scale = np.sqrt(1.0 / n_in)
            elif init == "small":
                scale = 0.01
            else:
                raise ValueError(f"Unknown init {init}")
            self.params[f"W{i}"] = rng.normal(0, scale, (n_in, n_out))
            self.params[f"b{i}"] = np.zeros(n_out)

    @property
    def n_layers(self) -> int:
        return len(self.sizes) - 1

    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    # ---------- forward / backward ----------
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list]:
        act, _ = ACTIVATIONS[self.activation]
        cache, a = [], x
        for i in range(1, self.n_layers + 1):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache.append((a, z))
            a = act(z) if i < self.n_layers else z
        return a, cache  # final layer returns logits / raw outputs

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        out, cache = self.forward(x)
        n = len(x)
        if self.loss == "cross_entropy":
            p = softmax(out)
            loss = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
            d = p.copy()
            d[np.arange(n), y] -= 1
            d /= n
        else:
            target = y if y.ndim == 2 else y.reshape(-1, 1)
            loss = 0.5 * np.mean(np.sum((out - target) ** 2, axis=1))
            d = (out - target) / n
        loss += 0.5 * self.weight_decay * sum(np.sum(self.params[f"W{i}"] ** 2) for i in range(1, self.n_layers + 1))

        _, act_grad = ACTIVATIONS[self.activation]
        grads = {}
        for i in range(self.n_layers, 0, -1):
            a_prev, z = cache[i - 1]
            if i < self.n_layers:
                d = d * act_grad(z)
            grads[f"W{i}"] = a_prev.T @ d + self.weight_decay * self.params[f"W{i}"]
            grads[f"b{i}"] = d.sum(axis=0)
            d = d @ self.params[f"W{i}"].T
        return float(loss), grads

    def predict_proba(self, x: np.ndarray, batch: int = 4096) -> np.ndarray:
        outs = [self.forward(x[i:i + batch])[0] for i in range(0, len(x), batch)]
        out = np.concatenate(outs) if outs else np.zeros((0, self.sizes[-1]))
        return softmax(out) if self.loss == "cross_entropy" else out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    # ---------- checkpoints ----------
    def hyperparameters(self) -> dict:
        return {"sizes": self.sizes, "activation": self.activation, "init": self.init,
                "loss": self.loss, "weight_decay": self.weight_decay}

    def save(self, path: str | Path, extra_arrays: dict[str, np.ndarray] | None = None,
             meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"param__{k}": v for k, v in self.params.items()}
        arrays.update({f"extra__{k}": v for k, v in (extra_arrays or {}).items()})
        info = {"model": "MLP", "hyperparameters": self.hyperparameters(), **(meta or {})}
        np.savez_compressed(path, __meta__=np.array(json.dumps(info, default=str)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple["MLP", dict[str, np.ndarray], dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}. Train it first (see docs/USER_GUIDE.md).")
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["__meta__"]))
        hp = meta["hyperparameters"]
        model = cls(hp["sizes"], hp["activation"], hp["init"], hp["loss"], hp["weight_decay"])
        for k in data.files:
            if k.startswith("param__"):
                model.params[k[len("param__"):]] = data[k]
        extras = {k[len("extra__"):]: data[k] for k in data.files if k.startswith("extra__")}
        return model, extras, meta


# ---------------- optimizers ----------------
class Optimizer:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, params: dict, grads: dict) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Plain gradient descent. Full-batch GD vs SGD is decided by the batch size in `train`."""

    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class Momentum(Optimizer):
    def __init__(self, lr: float, mu: float = 0.9):
        super().__init__(lr)
        self.mu, self.v = mu, {}

    def step(self, params, grads):
        for k in params:
            v = self.v.get(k, np.zeros_like(params[k]))
            v = self.mu * v - self.lr * grads[k]
            self.v[k] = v
            params[k] += v


class Nesterov(Optimizer):
    """Nesterov accelerated gradient in the common 'look-ahead' reformulation."""

    def __init__(self, lr: float, mu: float = 0.9):
        super().__init__(lr)
        self.mu, self.v = mu, {}

    def step(self, params, grads):
        for k in params:
            v_prev = self.v.get(k, np.zeros_like(params[k]))
            v = self.mu * v_prev - self.lr * grads[k]
            self.v[k] = v
            params[k] += -self.mu * v_prev + (1 + self.mu) * v


class Adam(Optimizer):
    def __init__(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.b1, self.b2, self.eps, self.m, self.v, self.t = b1, b2, eps, {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            m = self.b1 * self.m.get(k, 0) + (1 - self.b1) * grads[k]
            v = self.b2 * self.v.get(k, 0) + (1 - self.b2) * grads[k] ** 2
            self.m[k], self.v[k] = m, v
            mhat, vhat = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


OPTIMIZERS = {"gd": SGD, "sgd": SGD, "momentum": Momentum, "nesterov": Nesterov, "adam": Adam}


def make_optimizer(name: str, lr: float) -> Optimizer:
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](lr)


def train(model: MLP, x: np.ndarray, y: np.ndarray, optimizer: Optimizer, epochs: int = 20,
          batch_size: int | None = 128, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
          seed: int = 0, augment: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
          keep_best: bool = True) -> dict:
    """Mini-batch training loop. batch_size=None -> full-batch gradient descent.
    The checkpoint with the best *validation* accuracy is kept (never the test set)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = n if batch_size is None else batch_size
    hist = {"train_loss": [], "val_acc": []}
    best = (-1.0, None, -1)
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            xb = augment(x[idx], rng) if augment else x[idx]
            loss, grads = model.loss_and_grads(xb, y[idx])
            optimizer.step(model.params, grads)
            losses.append(loss)
        hist["train_loss"].append(float(np.mean(losses)))
        if x_val is not None:
            acc = float(np.mean(model.predict(x_val) == y_val))
            hist["val_acc"].append(acc)
            if keep_best and acc > best[0]:
                best = (acc, {k: v.copy() for k, v in model.params.items()}, epoch)
    if keep_best and best[1] is not None:
        model.params = best[1]
        hist["best_epoch"], hist["best_val_acc"] = best[2], best[0]
    return hist


def gradient_check(model: MLP, x: np.ndarray, y: np.ndarray, eps: float = 1e-5, n_checks: int = 20,
                   seed: int = 0) -> float:
    """Compare analytic gradients with central finite differences on random entries.
    Returns the max relative error (should be < 1e-5 for float64)."""
    rng = np.random.default_rng(seed)
    _, grads = model.loss_and_grads(x, y)
    worst = 0.0
    for _ in range(n_checks):
        key = list(model.params)[rng.integers(len(model.params))]
        p = model.params[key]
        idx = tuple(rng.integers(s) for s in p.shape)
        old = p[idx]
        p[idx] = old + eps
        lp, _ = model.loss_and_grads(x, y)
        p[idx] = old - eps
        lm, _ = model.loss_and_grads(x, y)
        p[idx] = old
        num, ana = (lp - lm) / (2 * eps), grads[key][idx]
        worst = max(worst, abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
    return float(worst)
