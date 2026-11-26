"""Classical learning rules (syllabus demonstrations, NOT the production model).

Hebbian, modified Hebbian (Oja), competitive (winner-take-all), error-correction
(delta rule / perceptron) and an energy-based Hopfield network.
"""

from __future__ import annotations

import numpy as np


def hebbian(x: np.ndarray, lr: float = 0.01, epochs: int = 20, seed: int = 0) -> np.ndarray:
    """Plain Hebb rule Δw = η·y·x with y = w·x. Weight norm grows without bound."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, x.shape[1])
    norms = []
    for _ in range(epochs):
        for xi in x:
            w += lr * (w @ xi) * xi
        norms.append(np.linalg.norm(w))
    return np.array(norms)


def oja(x: np.ndarray, lr: float = 0.01, epochs: int = 50, seed: int = 0) -> np.ndarray:
    """Oja's modified Hebb rule Δw = η·y·(x − y·w). Converges to the first principal component."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, x.shape[1])
    for _ in range(epochs):
        for xi in x:
            y = w @ xi
            w += lr * y * (xi - y * w)
    return w


def competitive(x: np.ndarray, k: int, lr: float = 0.1, epochs: int = 20, seed: int = 0) -> np.ndarray:
    """Winner-take-all: only the closest prototype moves toward each input."""
    rng = np.random.default_rng(seed)
    protos = x[rng.choice(len(x), k, replace=False)].astype(float).copy()
    for epoch in range(epochs):
        eta = lr * (1 - epoch / epochs)
        for xi in x[rng.permutation(len(x))]:
            winner = np.argmin(((protos - xi) ** 2).sum(1))
            protos[winner] += eta * (xi - protos[winner])
    return protos


def delta_rule(x: np.ndarray, t: np.ndarray, lr: float = 0.1, epochs: int = 50, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Error-correction (Widrow–Hoff / perceptron-style) learning for a single linear unit.

    Returns weights (last element is bias) and the per-epoch mean squared error.
    """
    rng = np.random.default_rng(seed)
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = rng.normal(0, 0.1, xb.shape[1])
    history = []
    for _ in range(epochs):
        for xi, ti in zip(xb, t):
            w += lr * (ti - xi @ w) * xi          # Δw = η (target − output) x
        history.append(float(np.mean((t - xb @ w) ** 2)))
    return w, history


class Hopfield:
    """Binary (±1) Hopfield network: Hebbian storage + energy-descending recall."""

    def __init__(self, n: int):
        self.w = np.zeros((n, n))

    def store(self, patterns: np.ndarray) -> None:
        for p in patterns:
            self.w += np.outer(p, p)
        np.fill_diagonal(self.w, 0)
        self.w /= len(patterns)

    def energy(self, s: np.ndarray) -> float:
        return float(-0.5 * s @ self.w @ s)

    def recall(self, s: np.ndarray, steps: int = 5, seed: int = 0) -> tuple[np.ndarray, list[float]]:
        rng = np.random.default_rng(seed)
        s = s.copy()
        energies = [self.energy(s)]
        for _ in range(steps):
            for i in rng.permutation(len(s)):          # asynchronous updates never raise energy
                s[i] = 1 if self.w[i] @ s >= 0 else -1
            energies.append(self.energy(s))
        return s, energies
