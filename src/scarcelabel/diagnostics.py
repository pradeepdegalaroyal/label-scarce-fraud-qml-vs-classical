"""Parameter-redundancy diagnostic: spectrum of the logit Jacobian over the training set."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def jacobian_spectrum(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], theta: torch.Tensor, x: np.ndarray) -> np.ndarray:
    """Singular values of d logit_i / d theta_j (samples x parameters)."""
    xt = torch.as_tensor(x, dtype=torch.float64)
    jac = torch.func.jacrev(lambda t: fn(t, xt))(theta.detach())
    return torch.linalg.svdvals(jac).numpy()


def effective_rank(s: np.ndarray) -> float:
    """exp of the Shannon entropy of the normalised squared singular values."""
    p = s**2 / np.sum(s**2)
    p = p[p > 1e-16]
    return float(np.exp(-np.sum(p * np.log(p))))


def numerical_rank(s: np.ndarray, rel_tol: float = 1e-2) -> int:
    """Number of singular values above rel_tol times the largest."""
    return int(np.sum(s > rel_tol * s.max()))
