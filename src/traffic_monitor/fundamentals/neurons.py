"""Activation functions, McCulloch–Pitts and sigmoid neurons."""

from __future__ import annotations

import numpy as np


def step(z: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return (np.asarray(z) >= threshold).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_grad(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (np.asarray(z) > 0).astype(float)


def mcculloch_pitts(x: np.ndarray, weights: np.ndarray, threshold: float) -> np.ndarray:
    """Binary threshold unit: fires (1) when the weighted sum of binary inputs >= threshold."""
    x = np.asarray(x, dtype=float)
    return step(x @ np.asarray(weights, dtype=float), threshold)


def sigmoid_neuron(x: np.ndarray, weights: np.ndarray, bias: float) -> dict:
    """Return each stage explicitly: weighted sum z = w·x + b, then a = sigmoid(z)."""
    x = np.asarray(x, dtype=float)
    z = x @ np.asarray(weights, dtype=float) + bias
    return {"weighted_sum": z, "activation": sigmoid(z)}


def logic_gate_demo() -> dict:
    """McCulloch–Pitts units implementing AND / OR / NOT."""
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    return {
        "inputs": inputs.tolist(),
        "AND(w=[1,1], θ=2)": mcculloch_pitts(inputs, [1, 1], 2).tolist(),
        "OR(w=[1,1], θ=1)": mcculloch_pitts(inputs, [1, 1], 1).tolist(),
        "NOT(w=[-1], θ=0)": mcculloch_pitts(np.array([[0], [1]]), [-1], 0).tolist(),
    }
