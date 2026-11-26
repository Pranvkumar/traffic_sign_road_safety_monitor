"""Tensor mathematics demonstrations (course brief item 3) in plain NumPy."""

from __future__ import annotations

from typing import Callable

import numpy as np


def tensor_basics() -> dict:
    """Creation, broadcasting, inner products, reshaping and transpose with small examples."""
    a = np.arange(6, dtype=float).reshape(2, 3)          # 2x3 matrix
    bias = np.array([10.0, 20.0, 30.0])                  # shape (3,) broadcast over rows
    v1, v2 = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    batch = np.zeros((4, 32, 32, 3))                     # NHWC image batch
    return {
        "a": a.tolist(),
        "a_plus_bias(broadcast)": (a + bias).tolist(),
        "inner_product_v1_v2": float(v1 @ v2),
        "outer_product_shape": list(np.outer(v1, v2).shape),
        "matmul_a_aT": (a @ a.T).tolist(),
        "reshape_2x3_to_3x2": a.reshape(3, 2).tolist(),
        "nhwc_to_nchw_shape": list(batch.transpose(0, 3, 1, 2).shape),
        "flatten_batch_shape": list(batch.reshape(len(batch), -1).shape),
    }


def svd_low_rank(image: np.ndarray, ranks: tuple[int, ...] = (1, 4, 8, 16)) -> dict:
    """SVD of a 2D (grayscale) image and its rank-k reconstructions + relative errors."""
    u, s, vt = np.linalg.svd(image, full_matrices=False)
    recon, errors = {}, {}
    for k in ranks:
        approx = (u[:, :k] * s[:k]) @ vt[:k]
        recon[k] = approx
        errors[k] = float(np.linalg.norm(image - approx) / np.linalg.norm(image))
    return {"singular_values": s, "reconstructions": recon, "relative_error": errors}


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of scalar f at x (x is modified then restored)."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """max |a-b| / max(|a|+|b|) — standard gradient-check metric (pass if < 1e-6 or so)."""
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))
