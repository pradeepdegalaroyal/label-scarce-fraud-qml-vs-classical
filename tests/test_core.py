"""Fast tests for the simulator, data generators, noise model and models."""
import numpy as np
import pytest
import torch

from scarcelabel.data import PREVALENCE, make_split, zz_feature_states
from scarcelabel.diagnostics import effective_rank
from scarcelabel.noise import fidelity, noisy_probs, sample_kernel, sample_mean_z, vqc_cx_count
from scarcelabel.qkernel import gram, make_psd
from scarcelabel.sim import HADAMARD, Z_MEAN, apply_1q, probabilities, rx, zero_state
from scarcelabel.vqc import VQCSpec, init_vqc, vqc_logits, vqc_probs


def test_hadamard_gives_uniform_probabilities():
    state = zero_state(1)
    for q in range(4):
        state = apply_1q(state, HADAMARD, q)
    assert torch.allclose(probabilities(state), torch.full((1, 16), 1 / 16, dtype=torch.float64))


def test_rx_pi_flips_qubit():
    state = apply_1q(zero_state(1), rx(torch.tensor([np.pi], dtype=torch.float64)), 2)
    p = probabilities(state)[0]
    assert p[0b0010] == pytest.approx(1.0)


def test_feature_states_normalised_and_kernel_diagonal_one():
    x = np.random.default_rng(0).uniform(0, np.pi, (6, 4))
    psi = zz_feature_states(x, 1.0)
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0)
    k = gram(psi, psi)
    assert np.allclose(np.diag(k), 1.0) and np.allclose(k, k.T) and k.min() >= -1e-12


def test_vqc_probabilities_sum_to_one_and_gradient_flows():
    spec = VQCSpec(3, True)
    theta = init_vqc(spec, torch.Generator().manual_seed(0), PREVALENCE).requires_grad_(True)
    x = torch.rand(5, 4, dtype=torch.float64) * np.pi
    assert torch.allclose(vqc_probs(theta, x, spec).sum(1), torch.ones(5, dtype=torch.float64))
    vqc_logits(theta, x, spec).sum().backward()
    assert torch.isfinite(theta.grad).all() and theta.grad.abs().sum() > 0


def test_entangling_ring_changes_output():
    gen = torch.Generator().manual_seed(1)
    on, off = VQCSpec(3, True), VQCSpec(3, False)
    theta = init_vqc(on, gen, 0.15)
    x = torch.rand(4, 4, dtype=torch.float64) * np.pi
    assert not torch.allclose(vqc_probs(theta, x, on), vqc_probs(theta, x, off))


@pytest.mark.parametrize("name", ["fraud_like", "qmap_control"])
def test_split_has_exact_prevalence_and_range(name):
    sp = make_split(name, 40, 100, seed=3)
    assert sp.x_train.shape == (40, 4) and sp.x_test.shape == (100, 4)
    assert sp.y_test.sum() == 15
    assert sp.x_train.min() >= 0 and sp.x_train.max() <= np.pi + 1e-9
    assert make_split(name, 40, 100, seed=3).x_train.tolist() == sp.x_train.tolist()


def test_noise_model():
    assert fidelity(0.0, 32) == 1.0 and fidelity(0.01, 8) == pytest.approx(0.99**8)
    assert vqc_cx_count(3, True) == 8 and vqc_cx_count(3, False) == 0
    p = np.full((3, 16), 1 / 16)
    assert np.allclose(noisy_probs(p, 0.5).sum(1), 1.0)
    rng = np.random.default_rng(0)
    est = sample_mean_z(np.eye(16)[[0]], Z_MEAN.numpy(), 50, rng)
    assert est[0] == pytest.approx(1.0)  # |0000> has mean Z = +1
    k = sample_kernel(np.full((6, 6), 0.5), 100, rng, symmetric=True)
    assert np.allclose(k, k.T)


def test_psd_projection_and_effective_rank():
    a = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert np.linalg.eigvalsh(make_psd(a)).min() >= -1e-12
    assert effective_rank(np.ones(5)) == pytest.approx(5.0)
    assert effective_rank(np.array([1.0, 1e-9])) == pytest.approx(1.0, abs=1e-6)
