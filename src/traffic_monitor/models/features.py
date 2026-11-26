"""Hand-crafted features shared by the classifier, detector, LSTM and VAE experiments."""

from __future__ import annotations

import cv2
import numpy as np

from ..data.transforms import to_gray

if not hasattr(cv2, "HOGDescriptor"):
    raise ImportError(f"OpenCV {cv2.__version__} has no cv2.HOGDescriptor. Install the tested version: "
                      "pip uninstall -y opencv-python-headless && pip install opencv-contrib-python-headless==5.0.0.93")
_HOG32 = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)


def hog32(x_rgb: np.ndarray) -> np.ndarray:
    """(N,32,32,3) RGB float [0,1] -> (N, 324) HOG on the luminance channel."""
    gray = (to_gray(x_rgb) * 255).astype(np.uint8)
    return np.stack([_HOG32.compute(g).ravel() for g in gray]).astype(np.float32)


def colour_thumb(x_rgb: np.ndarray, size: int = 4) -> np.ndarray:
    """Coarse 4x4 RGB thumbnail (48 dims): keeps red-vs-blue sign colour information."""
    n = len(x_rgb)
    return np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for im in x_rgb]).reshape(n, -1)


def sign_features(x_rgb: np.ndarray) -> np.ndarray:
    """HOG + colour thumbnail: the feature vector used by the HOG-MLP classifier."""
    return np.hstack([hog32(x_rgb), colour_thumb(x_rgb)]).astype(np.float32)
