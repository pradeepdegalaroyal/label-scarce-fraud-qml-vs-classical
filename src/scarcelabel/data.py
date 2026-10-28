"""Synthetic imbalanced data sets: a fraud-like set and a quantum-feature-map control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .sim import N_QUBITS, DIM, Z_TABLE, RING, HADAMARD, apply_1q, apply_diag, zero_state

PREVALENCE = 0.15
LABEL_FLIP = 0.03
DATASETS = ("fraud_like", "qmap_control")


@dataclass(frozen=True)
class Split:
    """A train/test split with features in [0, pi]^4."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def zz_feature_states(x: np.ndarray, bandwidth: float = 1.0, reps: int = 2) -> np.ndarray:
    """ZZ-type feature-map states |psi(x)>, shape (N, 16), complex128.

    Each repetition applies Hadamards, then the diagonal phase
    exp(-i [a * sum_q x_q z_q + a * sum_(ij in ring) (pi - x_i)(pi - x_j) z_i z_j / pi]).
    """
    xt = torch.as_tensor(x, dtype=torch.float64)
    n = xt.shape[0]
    z = Z_TABLE
    angle = bandwidth * (xt @ z.T)
    for i, j in RING:
        pair = (np.pi - xt[:, i]) * (np.pi - xt[:, j]) / np.pi
        angle = angle + bandwidth * pair[:, None] * (z[:, i] * z[:, j])[None, :]
    state = zero_state(n)
    for _ in range(reps):
        for q in range(N_QUBITS):
            state = apply_1q(state, HADAMARD, q)
        state = apply_diag(state, angle)
    return state.reshape(n, DIM).numpy()


def _random_observable(seed: int = 7, terms: int = 8) -> np.ndarray:
    """Fixed random Pauli-sum observable (16 x 16, Hermitian) for the control labels."""
    rng = np.random.default_rng(seed)
    paulis = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]]),
    ]
    h = np.zeros((DIM, DIM), dtype=complex)
    for _ in range(terms):
        idx = np.zeros(N_QUBITS, dtype=int)
        pos = rng.choice(N_QUBITS, size=2, replace=False)
        idx[pos] = rng.integers(1, 4, size=2)
        mat = np.array([[1.0]])
        for k in idx:
            mat = np.kron(mat, paulis[k])
        h += rng.normal() * mat
    return h


_OBS = _random_observable()


def _scale_fraud(raw: np.ndarray) -> np.ndarray:
    return (np.clip(raw, -2.5, 2.5) + 2.5) / 5.0 * np.pi


def _fraud_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal((n, N_QUBITS))


def _fraud_score(raw: np.ndarray) -> np.ndarray:
    return raw[:, 0] * raw[:, 1] + 0.8 * raw[:, 2] * raw[:, 3] + 0.3 * raw[:, 0]


def _control_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.0, np.pi, size=(n, N_QUBITS))


def _control_score(x: np.ndarray) -> np.ndarray:
    psi = zz_feature_states(x, bandwidth=1.0, reps=2)
    return np.real(np.einsum("ni,ij,nj->n", psi.conj(), _OBS, psi))


_SPEC: dict[str, tuple[Callable, Callable, Callable]] = {
    "fraud_like": (_fraud_sampler, _fraud_score, _scale_fraud),
    "qmap_control": (_control_sampler, _control_score, lambda a: a),
}
_THRESHOLDS: dict[str, float] = {}


def _threshold(name: str) -> float:
    """Fixed score threshold giving ~15% positives (from a 100k reference draw)."""
    if name not in _THRESHOLDS:
        sampler, score, _ = _SPEC[name]
        ref = sampler(np.random.default_rng(12345), 100_000)
        _THRESHOLDS[name] = float(np.quantile(score(ref), 1.0 - PREVALENCE))
    return _THRESHOLDS[name]


def _draw_class(name: str, rng: np.random.Generator, n: int, positive: bool) -> np.ndarray:
    sampler, score, scale = _SPEC[name]
    tau = _threshold(name)
    kept: list[np.ndarray] = []
    have = 0
    while have < n:
        cand = sampler(rng, 4096)
        mask = (score(cand) >= tau) == positive
        kept.append(cand[mask])
        have += int(mask.sum())
    return scale(np.concatenate(kept)[:n])


def make_split(name: str, n_train: int, n_test: int, seed: int) -> Split:
    """Draw a split with exactly 15% positives; train labels carry 3% symmetric flip noise."""
    if name not in _SPEC:
        raise ValueError(f"unknown data set {name!r}")
    rng = np.random.default_rng(seed)

    def build(n: int) -> tuple[np.ndarray, np.ndarray]:
        n_pos = max(int(round(PREVALENCE * n)), 3)
        x = np.concatenate([_draw_class(name, rng, n_pos, True), _draw_class(name, rng, n - n_pos, False)])
        y = np.concatenate([np.ones(n_pos, int), np.zeros(n - n_pos, int)])
        perm = rng.permutation(n)
        return x[perm], y[perm]

    x_tr, y_tr = build(n_train)
    x_te, y_te = build(n_test)
    flip = rng.random(n_train) < LABEL_FLIP
    y_noisy = np.where(flip, 1 - y_tr, y_tr)
    if y_noisy.sum() < 2:  # keep at least two positives so every model can be fitted
        y_noisy = y_tr
    return Split(x_tr, y_noisy, x_te, y_te)
