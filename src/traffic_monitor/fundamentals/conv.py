"""Educational 2D/3D convolution (cross-correlation, as in deep-learning libraries)."""

from __future__ import annotations

import numpy as np


def conv_output_size(n: int, k: int, stride: int = 1, padding: int = 0) -> int:
    """out = floor((n + 2p − k) / s) + 1"""
    if k > n + 2 * padding:
        raise ValueError("Kernel larger than padded input")
    return (n + 2 * padding - k) // stride + 1


def conv2d(x: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Single-channel 2D convolution of x (H, W) with kernel (kh, kw)."""
    x = np.pad(x, padding)
    kh, kw = kernel.shape
    oh, ow = (x.shape[0] - kh) // stride + 1, (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i * stride:i * stride + kh, j * stride:j * stride + kw] * kernel)
    return out


def conv2d_multichannel(x: np.ndarray, kernels: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """x (C, H, W), kernels (F, C, kh, kw) -> stacked feature maps (F, oh, ow).
    Each filter sums its per-channel 2D convolutions."""
    return np.stack([sum(conv2d(x[c], k[c], stride, padding) for c in range(x.shape[0])) for k in kernels])


def conv3d(x: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """x (T, H, W) frame volume, kernel (kt, kh, kw) -> (ot, oh, ow). Slides over time too."""
    kt, kh, kw = kernel.shape
    ot, oh, ow = ((np.array(x.shape) - np.array(kernel.shape)) // stride + 1)
    out = np.zeros((ot, oh, ow))
    for t in range(ot):
        for i in range(oh):
            for j in range(ow):
                out[t, i, j] = np.sum(x[t * stride:t * stride + kt, i * stride:i * stride + kh,
                                        j * stride:j * stride + kw] * kernel)
    return out


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_Y = SOBEL_X.T
