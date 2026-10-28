"""Fidelity-kernel support vector machine with optional effective noise and finite shots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from .data import zz_feature_states
from .noise import CX_KERNEL, fidelity, noisy_probs, sample_kernel
from .sim import DIM

BANDWIDTHS = (0.5, 1.0, 2.0)
C_GRID = (0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class KernelResult:
    """Decision scores on the test set and the selected hyper-parameters."""

    scores: np.ndarray
    bandwidth: float
    c: float


def gram(psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
    """Exact fidelity kernel |<psi_a|psi_b>|^2."""
    return np.abs(psi_a.conj() @ psi_b.T) ** 2


def make_psd(k: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clipping negative eigenvalues."""
    w, v = np.linalg.eigh((k + k.T) / 2)
    return (v * np.clip(w, 0.0, None)) @ v.T


def cv_select(k_train: np.ndarray, y: np.ndarray, seed: int, c_grid=C_GRID) -> tuple[float, float]:
    """Return (best C, best mean cross-validated AP) on a precomputed Gram matrix."""
    n_pos = int(y.sum())
    folds = max(2, min(3, n_pos))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    best = (c_grid[0], -1.0)
    for c in c_grid:
        scores = []
        for tr, va in skf.split(k_train, y):
            if y[tr].sum() == 0 or y[va].sum() == 0:
                continue
            clf = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(k_train[np.ix_(tr, tr)], y[tr])
            scores.append(average_precision_score(y[va], clf.decision_function(k_train[np.ix_(va, tr)])))
        mean = float(np.mean(scores)) if scores else -1.0
        if mean > best[1]:
            best = (c, mean)
    return best


def fit_predict_qkernel(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    p_two_qubit: float = 0.0,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
    bandwidths=BANDWIDTHS,
) -> KernelResult:
    """Tune (bandwidth, C) by cross-validation, refit, and score the test set.

    Noise is applied to every Gram entry that is "measured": the kernel is mixed
    with the uniform value 1/16 (fidelity f = (1-p)^32) and sampled with `shots` shots.
    """
    rng = rng or np.random.default_rng(seed)
    f = fidelity(p_two_qubit, CX_KERNEL)
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for a in bandwidths:
        psi_tr = zz_feature_states(x_train, a)
        psi_te = zz_feature_states(x_test, a)
        k_tr = f * gram(psi_tr, psi_tr) + (1 - f) / DIM
        k_te = f * gram(psi_te, psi_tr) + (1 - f) / DIM
        k_tr = sample_kernel(k_tr, shots, rng, symmetric=True)
        k_te = sample_kernel(k_te, shots, rng, symmetric=False)
        if shots is not None or f < 1.0:
            k_tr = make_psd(k_tr)
        c, cv = cv_select(k_tr, y_train, seed)
        if best is None or cv > best[0]:
            best = (cv, a, c, k_tr, k_te)
    assert best is not None
    _, a, c, k_tr, k_te = best
    clf = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(k_tr, y_train)
    return KernelResult(clf.decision_function(k_te), a, c)
