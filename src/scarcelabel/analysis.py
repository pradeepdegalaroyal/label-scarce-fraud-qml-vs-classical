"""Aggregate raw result tables into slopes, paired gaps and a Markdown summary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .experiments import CLASSICAL, QUANTUM


def mean_std(df: pd.DataFrame, by: list[str], col: str = "ap") -> pd.DataFrame:
    g = df.groupby(by)[col].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def learning_slopes(learning: pd.DataFrame) -> pd.DataFrame:
    """Slope of log(1 - AP) vs log(n) for each data set and model (fit on seed means)."""
    rows = []
    for (ds, model), sub in learning.groupby(["dataset", "model"]):
        m = sub.groupby("n_train")["ap"].mean()
        slope, _ = np.polyfit(np.log(m.index.values), np.log(1.0 - m.values), 1)
        rows.append({"dataset": ds, "model": model, "slope": float(slope)})
    return pd.DataFrame(rows)


def paired_gaps(learning: pd.DataFrame) -> pd.DataFrame:
    """Per-seed AP differences (quantum minus classical) with mean, std and win counts."""
    wide = learning.pivot_table(index=["dataset", "n_train", "seed"], columns="model", values="ap")
    comp = pd.DataFrame(index=wide.index)
    comp["best quantum - best classical"] = wide[list(QUANTUM)].max(axis=1) - wide[list(CLASSICAL)].max(axis=1)
    comp["QKernel - RBF-SVM"] = wide["QKernel"] - wide["RBF-SVM"]
    comp["ReUpload-VQC - MLP-8"] = wide["ReUpload-VQC"] - wide["MLP-8"]
    rows = []
    for (ds, n), sub in comp.groupby(level=["dataset", "n_train"]):
        for c in comp.columns:
            rows.append({"dataset": ds, "n_train": n, "comparison": c, "mean_diff": sub[c].mean(),
                         "std_diff": sub[c].std(ddof=1) if len(sub) > 1 else 0.0,
                         "wins": int((sub[c] > 0).sum()), "seeds": len(sub)})
    return pd.DataFrame(rows)


def noise_crossing(noise: pd.DataFrame) -> pd.DataFrame:
    """Mean AP per (data set, model, p, shots), with the best classical AP for reference."""
    q = noise[noise["model"].isin(QUANTUM)]
    out = q.groupby(["dataset", "model", "p_two_qubit", "shots"])["ap"].agg(["mean", "std"]).reset_index()
    ref = noise[noise["shots"] == -1].groupby(["dataset", "seed", "model"])["ap"].mean().reset_index()
    best = ref.groupby(["dataset", "seed"])["ap"].max().groupby("dataset").agg(["mean", "std"]).reset_index()
    best = best.rename(columns={"mean": "best_classical_mean", "std": "best_classical_std"})
    return out.merge(best, on="dataset")


def summary_markdown(learning: pd.DataFrame) -> str:
    """Markdown table of mean +- std AP for every (data set, n, model)."""
    ms = mean_std(learning, ["dataset", "n_train", "model"])
    lines = []
    for ds, sub in ms.groupby("dataset"):
        lines.append(f"**{ds}** (average precision, mean +- std over seeds)\n")
        models = list(QUANTUM) + list(CLASSICAL)
        lines.append("| n labelled | " + " | ".join(models) + " |")
        lines.append("|---|" + "---|" * len(models))
        for n, g in sub.groupby("n_train"):
            cells = []
            for m in models:
                r = g[g["model"] == m].iloc[0]
                cells.append(f"{r['mean']:.3f} +- {r['std']:.3f}")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
