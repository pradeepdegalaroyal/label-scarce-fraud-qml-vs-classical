"""Reproduce every result and figure: `python experiments/run_all.py` (add --quick for a smoke run)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from scarcelabel import analysis, plots  # noqa: E402
from scarcelabel.experiments import RunConfig, run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny run (< 1 minute) written to results_quick/")
    parser.add_argument("--jobs", type=int, default=3, help="parallel worker processes over seeds")
    args = parser.parse_args()

    torch.set_num_threads(6)
    cfg = RunConfig.quick() if args.quick else RunConfig()
    res_dir = ROOT / ("results_quick" if args.quick else "results")
    fig_dir = ROOT / ("figures_quick" if args.quick else "figures")
    res_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    tables = run_all(cfg, n_jobs=2 if args.quick else args.jobs)
    for name, df in tables.items():
        df.to_csv(res_dir / f"{name}_raw.csv", index=False)

    slopes = analysis.learning_slopes(tables["learning"])
    gaps = analysis.paired_gaps(tables["learning"])
    slopes.to_csv(res_dir / "learning_slopes.csv", index=False)
    gaps.to_csv(res_dir / "paired_gaps.csv", index=False)
    analysis.noise_crossing(tables["noise"]).to_csv(res_dir / "noise_summary.csv", index=False)
    abl = tables["ablation"].groupby(["dataset", "variant"])[["ap", "auc", "eff_rank", "num_rank", "n_params"]].agg(["mean", "std"])
    abl.to_csv(res_dir / "ablation_summary.csv")
    (res_dir / "summary_table.md").write_text(analysis.summary_markdown(tables["learning"]), encoding="utf8")
    (res_dir / "config.json").write_text(json.dumps(cfg.__dict__, default=list, indent=2), encoding="utf8")

    n_violin = cfg.ns[1] if len(cfg.ns) > 1 else cfg.ns[0]
    plots.fig_learning_curves(tables["learning"], fig_dir / "fig1_learning_curves.png")
    plots.fig_violins(tables["learning"], fig_dir / "fig2_seed_violins.png", n_violin)
    plots.fig_noise(tables["noise"], fig_dir / "fig3_noise_small_multiples.png")
    plots.fig_ablation(tables["ablation"], fig_dir / "fig4_ablation_redundancy.png")
    print(f"results in {res_dir}, figures in {fig_dir}")


if __name__ == "__main__":
    main()
