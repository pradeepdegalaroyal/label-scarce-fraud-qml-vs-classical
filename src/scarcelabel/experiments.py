"""Experiment drivers: learning curves, noise sweep, ablation + redundancy diagnostic."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .classical import ap, fit_predict_classical
from .data import DATASETS, make_split
from .diagnostics import effective_rank, jacobian_spectrum, numerical_rank
from .qkernel import fit_predict_qkernel
from .vqc import (
    VQCSpec,
    fit_mlp,
    fit_vqc,
    mlp_logits,
    predict_mlp,
    predict_vqc,
    vqc_logits,
)

QUANTUM = ("QKernel", "ReUpload-VQC")
CLASSICAL = ("LogReg", "RBF-SVM", "MLP-8", "RandomForest")


@dataclass(frozen=True)
class RunConfig:
    """Sizes of one reproduction run."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ns: tuple[int, ...] = (32, 64, 128, 256)
    n_test: int = 2000
    noise_n: int = 128
    noise_ps: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
    noise_shots: tuple[int, ...] = (100, 1000, 10000, 0)  # 0 = exact expectation
    ablation_n: int = 128
    ablation_layers: tuple[int, ...] = (1, 2, 3)
    epochs: int = 150
    restarts: int = 2
    datasets: tuple[str, ...] = field(default=DATASETS)

    @classmethod
    def quick(cls) -> "RunConfig":
        return cls(seeds=(0, 1), ns=(32, 64), n_test=400, noise_n=48, noise_ps=(0.0, 0.01, 0.05),
                   noise_shots=(100, 0), ablation_n=48, ablation_layers=(1, 2), epochs=25, restarts=1)


def _metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    return {"ap": ap(y, s), "auc": float(roc_auc_score(y, s))}


def _learning(ds: str, seed: int, cfg: RunConfig) -> list[dict]:
    rows = []
    for n in cfg.ns:
        sp = make_split(ds, n, cfg.n_test, 1000 * seed + n)
        base = {"dataset": ds, "seed": seed, "n_train": n}
        qk = fit_predict_qkernel(sp.x_train, sp.y_train, sp.x_test, seed)
        rows.append({**base, "model": "QKernel", **_metrics(sp.y_test, qk.scores), "bandwidth": qk.bandwidth, "C": qk.c})
        spec = VQCSpec()
        th = fit_vqc(sp.x_train, sp.y_train, spec, seed, cfg.restarts, cfg.epochs)
        rows.append({**base, "model": "ReUpload-VQC", **_metrics(sp.y_test, predict_vqc(th, sp.x_test, spec))})
        mlp = fit_mlp(sp.x_train, sp.y_train, seed, cfg.restarts, cfg.epochs)
        rows.append({**base, "model": "MLP-8", **_metrics(sp.y_test, predict_mlp(mlp, sp.x_test))})
        for name, sc in fit_predict_classical(sp.x_train, sp.y_train, sp.x_test, seed).items():
            rows.append({**base, "model": name, **_metrics(sp.y_test, sc)})
    return rows


def _noise(ds: str, seed: int, cfg: RunConfig) -> list[dict]:
    n = cfg.noise_n
    sp = make_split(ds, n, cfg.n_test, 1000 * seed + n)
    spec = VQCSpec()
    th = fit_vqc(sp.x_train, sp.y_train, spec, seed, cfg.restarts, cfg.epochs)
    rows = []
    for p in cfg.noise_ps:
        for shots in cfg.noise_shots:
            s = None if shots == 0 else shots
            rng = np.random.default_rng(10_000 * seed + int(p * 1e4) + shots)
            base = {"dataset": ds, "seed": seed, "n_train": n, "p_two_qubit": p, "shots": shots}
            qk = fit_predict_qkernel(sp.x_train, sp.y_train, sp.x_test, seed, p, s, rng)
            rows.append({**base, "model": "QKernel", "ap": ap(sp.y_test, qk.scores)})
            sc = predict_vqc(th, sp.x_test, spec, p, s, rng)
            rows.append({**base, "model": "ReUpload-VQC", "ap": ap(sp.y_test, sc)})
    for name, sc in fit_predict_classical(sp.x_train, sp.y_train, sp.x_test, seed).items():
        rows.append({"dataset": ds, "seed": seed, "n_train": n, "p_two_qubit": np.nan, "shots": -1, "model": name,
                     "ap": ap(sp.y_test, sc)})
    return rows


def _ablation(ds: str, seed: int, cfg: RunConfig) -> list[dict]:
    n = cfg.ablation_n
    sp = make_split(ds, n, cfg.n_test, 1000 * seed + n)
    rows = []
    for layers in cfg.ablation_layers:
        for ent in (True, False):
            if layers == 1 and ent:
                continue  # no ring is applied after the last layer, so one layer is never entangled
            spec = VQCSpec(layers, ent)
            th = fit_vqc(sp.x_train, sp.y_train, spec, seed, cfg.restarts, cfg.epochs)
            sv = jacobian_spectrum(lambda t, x: vqc_logits(t, x, spec), th, sp.x_train)
            rows.append({"dataset": ds, "seed": seed, "variant": f"VQC L={layers} {'ring' if ent else 'no ring'}",
                         "layers": layers, "entangle": ent, "n_params": spec.n_params,
                         **_metrics(sp.y_test, predict_vqc(th, sp.x_test, spec)),
                         "eff_rank": effective_rank(sv), "num_rank": numerical_rank(sv)})
    mlp = fit_mlp(sp.x_train, sp.y_train, seed, cfg.restarts, cfg.epochs)
    sv = jacobian_spectrum(mlp_logits, mlp, sp.x_train)
    rows.append({"dataset": ds, "seed": seed, "variant": "MLP-8", "layers": 0, "entangle": False, "n_params": len(mlp),
                 **_metrics(sp.y_test, predict_mlp(mlp, sp.x_test)),
                 "eff_rank": effective_rank(sv), "num_rank": numerical_rank(sv)})
    return rows


def run_seed(seed: int, cfg: RunConfig, threads: int = 2) -> dict[str, list[dict]]:
    """Run all three experiments for one seed (used as the unit of parallel work)."""
    torch.set_num_threads(threads)
    out: dict[str, list[dict]] = {"learning": [], "noise": [], "ablation": []}
    for ds in cfg.datasets:
        out["learning"] += _learning(ds, seed, cfg)
        out["noise"] += _noise(ds, seed, cfg)
        out["ablation"] += _ablation(ds, seed, cfg)
    return out


def run_all(cfg: RunConfig, n_jobs: int = 3) -> dict[str, pd.DataFrame]:
    """Run every seed (in parallel processes) and return the three result tables."""
    from joblib import Parallel, delayed

    t0 = time.time()
    parts = Parallel(n_jobs=min(n_jobs, len(cfg.seeds)))(delayed(run_seed)(s, cfg, max(1, 6 // n_jobs)) for s in cfg.seeds)
    tables = {k: pd.DataFrame([r for part in parts for r in part[k]]) for k in ("learning", "noise", "ablation")}
    print(f"experiments finished in {time.time() - t0:.0f} s")
    return tables
