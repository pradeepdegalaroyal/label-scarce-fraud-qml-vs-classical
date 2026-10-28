"""Data re-uploading variational circuit (Jerbi et al. standard model) and a parameter-matched MLP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .noise import fidelity, noisy_probs, sample_mean_z, vqc_cx_count
from .sim import (
    CZ_RING_SIGN,
    N_QUBITS,
    Z_MEAN,
    apply_1q,
    probabilities,
    rx,
    ry,
    rz,
    zero_state,
)

EPOCHS = 150
LR = 0.05


@dataclass(frozen=True)
class VQCSpec:
    """Architecture of the re-uploading circuit."""

    layers: int = 3
    entangle: bool = True

    @property
    def n_params(self) -> int:
        return self.layers * N_QUBITS * 4 + 2


def vqc_probs(theta: torch.Tensor, x: torch.Tensor, spec: VQCSpec) -> torch.Tensor:
    """Basis-state probabilities of the circuit; each layer re-encodes x with trainable scales."""
    n_l, n_q = spec.layers, N_QUBITS
    scales = theta[: n_l * n_q].reshape(n_l, n_q)
    rot = theta[n_l * n_q : n_l * n_q * 4].reshape(n_l, n_q, 3)
    state = zero_state(x.shape[0]).reshape(x.shape[0], -1)
    cz = CZ_RING_SIGN.reshape(-1)
    for layer in range(n_l):
        enc = rx(scales[layer][None, :] * x)  # (B, 4, 2, 2)
        trainable = rz(rot[layer, :, 2]) @ ry(rot[layer, :, 1]) @ rz(rot[layer, :, 0])  # (4, 2, 2)
        g = trainable[None] @ enc
        big = torch.einsum("zij,zkl,zmn,zop->zikmojlnp", g[:, 0], g[:, 1], g[:, 2], g[:, 3])
        big = big.reshape(x.shape[0], 16, 16)
        state = torch.einsum("bij,bj->bi", big, state)
        if spec.entangle and layer < n_l - 1:
            state = state * cz
    state = state.reshape((x.shape[0],) + (2,) * N_QUBITS)
    return probabilities(state)


def vqc_logits(theta: torch.Tensor, x: torch.Tensor, spec: VQCSpec) -> torch.Tensor:
    """Logit = alpha * <mean Z> + beta."""
    expval = vqc_probs(theta, x, spec) @ Z_MEAN
    return theta[-2] * expval + theta[-1]


def init_vqc(spec: VQCSpec, gen: torch.Generator, prevalence: float) -> torch.Tensor:
    """Scales near 1, angles small-uniform, bias at the prior log-odds."""
    n_l, n_q = spec.layers, N_QUBITS
    theta = torch.empty(spec.n_params, dtype=torch.float64)
    theta[: n_l * n_q] = 1.0 + 0.1 * torch.randn(n_l * n_q, generator=gen, dtype=torch.float64)
    theta[n_l * n_q : n_l * n_q * 4] = (torch.rand(n_l * n_q * 3, generator=gen, dtype=torch.float64) - 0.5) * np.pi
    theta[-2] = 3.0
    theta[-1] = float(np.log(prevalence / (1 - prevalence)))
    return theta


MLP_HIDDEN = 8
MLP_PARAMS = 6 * MLP_HIDDEN + 1


def mlp_logits(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """4 -> 8 tanh -> 1 network on features standardised to zero mean (x - pi/2)."""
    h = MLP_HIDDEN
    w1 = theta[: 4 * h].reshape(4, h)
    b1 = theta[4 * h : 5 * h]
    w2 = theta[5 * h : 6 * h]
    b2 = theta[6 * h]
    return torch.tanh((x - np.pi / 2) @ w1 + b1) @ w2 + b2


def init_mlp(gen: torch.Generator, prevalence: float) -> torch.Tensor:
    theta = 0.5 * torch.randn(MLP_PARAMS, generator=gen, dtype=torch.float64)
    theta[-1] = float(np.log(prevalence / (1 - prevalence)))
    return theta


def train_logit_model(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    inits: list[torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> torch.Tensor:
    """Full-batch Adam on class-balanced cross-entropy; return the restart with lowest final loss."""
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    n_pos, n = float(yt.sum()), float(len(yt))
    w = torch.where(yt > 0.5, n / (2 * max(n_pos, 1.0)), n / (2 * max(n - n_pos, 1.0)))
    best_loss, best_theta = float("inf"), inits[0]
    for init in inits:
        theta = init.clone().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(fn(theta, xt), yt, reduction="none")).mean()
            loss.backward()
            opt.step()
        final = float(loss.detach())
        if final < best_loss:
            best_loss, best_theta = final, theta.detach().clone()
    return best_theta


def fit_vqc(x: np.ndarray, y: np.ndarray, spec: VQCSpec, seed: int, restarts: int = 2, epochs: int = EPOCHS) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    prev = max(float(np.mean(y)), 1e-3)
    inits = [init_vqc(spec, gen, prev) for _ in range(restarts)]
    return train_logit_model(lambda t, xx: vqc_logits(t, xx, spec), inits, x, y, epochs)


def fit_mlp(x: np.ndarray, y: np.ndarray, seed: int, restarts: int = 2, epochs: int = EPOCHS) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 500)
    prev = max(float(np.mean(y)), 1e-3)
    inits = [init_mlp(gen, prev) for _ in range(restarts)]
    return train_logit_model(mlp_logits, inits, x, y, epochs)


def predict_vqc(
    theta: torch.Tensor,
    x: np.ndarray,
    spec: VQCSpec,
    p_two_qubit: float = 0.0,
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Test-time logits under the effective noise model (exact expectation if shots is None)."""
    with torch.no_grad():
        probs = vqc_probs(theta, torch.as_tensor(x, dtype=torch.float64), spec).numpy()
    f = fidelity(p_two_qubit, vqc_cx_count(spec.layers, spec.entangle))
    rng = rng or np.random.default_rng(0)
    expval = sample_mean_z(noisy_probs(probs, f), Z_MEAN.numpy(), shots, rng)
    return float(theta[-2]) * expval + float(theta[-1])


def predict_mlp(theta: torch.Tensor, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return mlp_logits(theta, torch.as_tensor(x, dtype=torch.float64)).numpy()
