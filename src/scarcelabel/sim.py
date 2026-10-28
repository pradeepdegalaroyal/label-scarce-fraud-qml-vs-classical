"""Minimal batched statevector simulator (torch, complex128) for 4-qubit circuits."""
from __future__ import annotations

import itertools
import math

import torch

N_QUBITS = 4
DIM = 2**N_QUBITS
RING = [(i, (i + 1) % N_QUBITS) for i in range(N_QUBITS)]


def bit_table(n: int = N_QUBITS) -> torch.Tensor:
    """Return z = +-1 eigenvalues, shape (2**n, n); qubit 0 is the most significant bit."""
    rows = [[1.0 - 2.0 * b for b in bits] for bits in itertools.product((0, 1), repeat=n)]
    return torch.tensor(rows, dtype=torch.float64)


Z_TABLE = bit_table()
Z_MEAN = Z_TABLE.mean(dim=1)  # eigenvalue of (1/n) sum_q Z_q for each basis state
_bits = (Z_TABLE < 0).to(torch.float64)
_cz_phase = sum(_bits[:, i] * _bits[:, j] for i, j in RING)
CZ_RING_SIGN = ((-1.0) ** _cz_phase).to(torch.complex128).reshape((2,) * N_QUBITS)


def zero_state(batch: int) -> torch.Tensor:
    """Return |0000> repeated `batch` times, shape (batch, 2, 2, 2, 2)."""
    s = torch.zeros(batch, DIM, dtype=torch.complex128)
    s[:, 0] = 1.0
    return s.reshape((batch,) + (2,) * N_QUBITS)


def rx(theta: torch.Tensor) -> torch.Tensor:
    """X rotation; output shape theta.shape + (2, 2)."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    z = torch.zeros_like(c)
    m = torch.stack([torch.complex(c, z), torch.complex(z, -s), torch.complex(z, -s), torch.complex(c, z)], -1)
    return m.reshape(theta.shape + (2, 2))


def ry(theta: torch.Tensor) -> torch.Tensor:
    """Y rotation."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    m = torch.stack([c, -s, s, c], -1).to(torch.complex128)
    return m.reshape(theta.shape + (2, 2))


def rz(theta: torch.Tensor) -> torch.Tensor:
    """Z rotation."""
    e = torch.exp(-0.5j * theta.to(torch.complex128))
    z = torch.zeros_like(e)
    m = torch.stack([e, z, z, e.conj()], -1)
    return m.reshape(theta.shape + (2, 2))


HADAMARD = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2.0)


def apply_1q(state: torch.Tensor, u: torch.Tensor, q: int) -> torch.Tensor:
    """Apply a (2, 2) or batched (B, 2, 2) single-qubit gate to qubit `q`."""
    s = state.movedim(q + 1, -1)
    shape = s.shape
    s = s.reshape(shape[0], -1, 2)
    if u.dim() == 2:
        s = s @ u.T
    else:
        s = torch.einsum("bkj,bij->bki", s, u)
    return s.reshape(shape).movedim(-1, q + 1)


def apply_diag(state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Multiply the state by exp(-i * angles); angles has shape (B, 2**n)."""
    return state * torch.exp(-1j * angles.to(torch.complex128)).reshape(state.shape)


def probabilities(state: torch.Tensor) -> torch.Tensor:
    """Computational-basis probabilities, shape (B, 2**n)."""
    return (state.abs() ** 2).reshape(state.shape[0], -1)
