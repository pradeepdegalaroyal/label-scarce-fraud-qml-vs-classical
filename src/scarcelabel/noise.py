"""Effective noise model: global depolarising fidelity plus exact finite-shot sampling."""
from __future__ import annotations

import numpy as np

from .sim import DIM

CX_VQC_PER_RING = 4  # one ring of 4 CZ gates between layers
CX_KERNEL = 32  # compute-uncompute: 2 x 2 reps x 4 ZZ rotations x 2 CNOT


def vqc_cx_count(layers: int, entangle: bool = True) -> int:
    """Two-qubit gate count of the re-uploading circuit (ring between layers, none after the last)."""
    return CX_VQC_PER_RING * (layers - 1) if entangle else 0


def fidelity(p_two_qubit: float, n_gates: int) -> float:
    """Effective global-depolarising fidelity f = (1 - p)^N."""
    return float((1.0 - p_two_qubit) ** n_gates)


def noisy_probs(probs: np.ndarray, f: float) -> np.ndarray:
    """Mix outcome probabilities with the uniform distribution: f P + (1 - f) / 2^n."""
    return f * probs + (1.0 - f) / DIM


def sample_mean_z(probs: np.ndarray, z_mean: np.ndarray, shots: int | None, rng: np.random.Generator) -> np.ndarray:
    """Expectation of the mean-Z observable, exact (shots None) or from `shots` samples per row."""
    if shots is None:
        return probs @ z_mean
    p = probs / probs.sum(axis=1, keepdims=True)
    counts = rng.multinomial(shots, p)
    return (counts @ z_mean) / shots


def sample_kernel(k: np.ndarray, shots: int | None, rng: np.random.Generator, symmetric: bool) -> np.ndarray:
    """Binomial shot noise on a fidelity-kernel matrix (upper triangle mirrored if symmetric)."""
    if shots is None:
        return k
    kc = np.clip(k, 0.0, 1.0)
    out = rng.binomial(shots, kc) / shots
    if symmetric:
        up = np.triu(out)
        out = up + np.triu(out, 1).T
    return out
