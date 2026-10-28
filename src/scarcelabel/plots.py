"""Figures (viridis palette, 200 dpi): log-log learning curves, violins, noise small multiples, ablation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import learning_slopes
from .experiments import CLASSICAL, QUANTUM

MODELS = list(QUANTUM) + list(CLASSICAL)
COLORS = dict(zip(MODELS, plt.cm.viridis(np.linspace(0.0, 0.92, len(MODELS)))))
TITLES = {"fraud_like": "Fraud-like data (interaction boundary)", "qmap_control": "Quantum-feature-map control (circular)"}
DPI = 200


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_learning_curves(learning: pd.DataFrame, out: Path) -> None:
    slopes = learning_slopes(learning).set_index(["dataset", "model"])["slope"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, ds in zip(axes, sorted(learning["dataset"].unique())):
        sub = learning[learning["dataset"] == ds]
        for m in MODELS:
            g = sub[sub["model"] == m].groupby("n_train")["ap"].agg(["mean", "std"]).fillna(0)
            err, err_sd = 1 - g["mean"], g["std"]
            quantum = m in QUANTUM
            ax.errorbar(g.index, err, yerr=err_sd, color=COLORS[m], lw=2.6 if quantum else 1.4,
                        ls="-" if quantum else "--", marker="o" if quantum else "s", ms=5, capsize=2, label=m)
            ax.annotate(f"{slopes[(ds, m)]:+.2f}", (g.index[-1], err.iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", color=COLORS[m], fontsize=8, va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sorted(sub["n_train"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("labelled training samples n (15% positives)")
        ax.set_title(TITLES[ds], fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("error = 1 - average precision (lower is better)")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("Learning curves on log-log axes; numbers on the right are fitted slopes of log error vs log n", fontsize=10)
    _save(fig, out)


def fig_violins(learning: pd.DataFrame, out: Path, n: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, sorted(learning["dataset"].unique())):
        sub = learning[(learning["dataset"] == ds) & (learning["n_train"] == n)]
        data = [sub[sub["model"] == m]["ap"].values for m in MODELS]
        parts = ax.violinplot(data, showextrema=False, widths=0.8)
        for body, m in zip(parts["bodies"], MODELS):
            body.set_facecolor(COLORS[m])
            body.set_edgecolor("black")
            body.set_alpha(0.55)
        for i, (vals, m) in enumerate(zip(data, MODELS), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, color=COLORS[m], edgecolor="black", s=26, zorder=3)
            ax.hlines(np.mean(vals), i - 0.3, i + 0.3, color="black", lw=1.8, zorder=4)
        ax.axhline(0.15, color="gray", ls=":", lw=1)
        ax.set_xticks(range(1, len(MODELS) + 1))
        ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=8)
        ax.set_title(TITLES[ds], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"average precision at n = {n} (dots = seeds, bar = mean)")
    fig.suptitle("Seed-level spread at the smallest informative label budget (dotted line = no-skill 0.15)", fontsize=10)
    _save(fig, out)


def fig_noise(noise: pd.DataFrame, out: Path) -> None:
    ref = noise[noise["shots"] == -1]
    q = noise[noise["model"].isin(QUANTUM)]
    dss = sorted(q["dataset"].unique())
    shots_all = sorted(s for s in q["shots"].unique() if s > 0)
    ps = sorted(q["p_two_qubit"].unique())
    xs = np.arange(len(ps))
    fig, axes = plt.subplots(len(dss), len(shots_all), figsize=(3.9 * len(shots_all), 3.4 * len(dss)),
                             sharex=True, sharey="row", squeeze=False)
    for r, ds in enumerate(dss):
        best = ref[ref["dataset"] == ds].groupby(["seed", "model"])["ap"].mean().groupby("seed").max().mean()
        rbf = ref[(ref["dataset"] == ds) & (ref["model"] == "RBF-SVM")]["ap"].mean()
        for c, shots in enumerate(shots_all):
            ax = axes[r, c]
            for m in QUANTUM:
                col = COLORS[m]
                g = q[(q["dataset"] == ds) & (q["model"] == m) & (q["shots"] == shots)].groupby("p_two_qubit")["ap"].agg(["mean", "std"]).reindex(ps)
                ax.errorbar(xs, g["mean"], yerr=g["std"].fillna(0), color=col, marker="o", lw=2, capsize=2, label=f"{m}, {shots} shots")
                ex = q[(q["dataset"] == ds) & (q["model"] == m) & (q["shots"] == 0)].groupby("p_two_qubit")["ap"].mean().reindex(ps)
                if len(ex.dropna()):
                    ax.plot(xs, ex.values, color=col, ls=":", lw=1.3, label=f"{m}, exact" if (r == 0 and c == 0) else None)
            ax.axhline(best, color="black", ls="--", lw=1.2, label="best classical (noise-free)")
            ax.axhline(rbf, color=COLORS["RBF-SVM"], ls="-.", lw=1.2, label="RBF-SVM")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{100 * p:g}" for p in ps])
            ax.set_title(f"{TITLES[ds].split(' (')[0]}, {shots} shots", fontsize=9)
            ax.grid(alpha=0.3)
            if r == len(dss) - 1:
                ax.set_xlabel("two-qubit gate error p (%)")
            if c == 0:
                ax.set_ylabel("average precision")
    axes[0, 0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Noise sweep at n = 128: dotted = exact expectation, solid = finite shots, dashed = classical references", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def fig_ablation(ablation: pd.DataFrame, out: Path) -> None:
    dss = sorted(ablation["dataset"].unique())
    order = list(dict.fromkeys(ablation.sort_values(["layers", "entangle"])["variant"]))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(order)))
    fig, axes = plt.subplots(len(dss), 2, figsize=(11, 3.6 * len(dss)), squeeze=False)
    rng = np.random.default_rng(1)
    for r, ds in enumerate(dss):
        sub = ablation[ablation["dataset"] == ds]
        for c, (col, label) in enumerate([("ap", "average precision"), ("eff_rank", "Jacobian effective rank")]):
            ax = axes[r, c]
            data = [sub[sub["variant"] == v][col].values for v in order]
            bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "black"})
            for patch, colr in zip(bp["boxes"], cmap):
                patch.set_facecolor(colr)
                patch.set_alpha(0.6)
            for i, vals in enumerate(data, start=1):
                ax.scatter(i + rng.uniform(-0.1, 0.1, len(vals)), vals, color="black", s=12, zorder=3)
            ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(label)
            ax.set_title(f"{TITLES[ds].split(' (')[0]}: {label}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation at n = 128: depth, entangling ring and parameter redundancy (dots = seeds)", fontsize=10)
    fig.tight_layout()
    _save(fig, out)
