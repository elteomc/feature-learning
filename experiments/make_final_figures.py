from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


SummaryMetric = Dict[str, float]
SummaryGroup = Dict[str, Dict[str, SummaryMetric]]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_compact_summary(path: Path) -> Dict[str, SummaryGroup]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(
    summary: Dict[str, SummaryGroup],
    family: str,
    block: str,
    metric: str,
    stat: str = "mean",
) -> float:
    try:
        return float(summary[family][block][metric][stat])
    except KeyError:
        return float("nan")


def sorted_families(summary: Dict[str, SummaryGroup]) -> List[str]:
    preferred = [
        "isotropic",
        "anisotropic",
        "low_rank_signal",
        "clustered_gaussian",
        "mixture_subspaces",
        "rare_region_outliers",
        "two_region_gating",
        "xor_feature",
    ]
    return [family for family in preferred if family in summary]


def save_bar_plot(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(len(labels)))
    plotted_values = [value if math.isfinite(value) else 0.0 for value in values]
    ax.bar(xs, plotted_values)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_summary_figures(summary_path: Path, outdir: Path) -> None:
    summary = load_compact_summary(summary_path)
    families = sorted_families(summary)

    figure_specs: List[Tuple[str, str, str, str, str, bool]] = [
        (
            "weighted_residual_by_family.png",
            "best_stationarity",
            "gamma_tilde_eff_op",
            "Weighted law residual",
            "operator norm residual",
            True,
        ),
        (
            "theorem_bound_ratio_by_family.png",
            "best_stationarity",
            "theorem_bound_ratio",
            "Observed residual divided by theorem bound",
            "bound ratio",
            False,
        ),
        (
            "beta_over_residual_energy_by_family.png",
            "all_checkpoints",
            "beta_over_resid_mean_sq",
            "Beta bridge tracks residual energy",
            "beta_fit / mean residual squared",
            False,
        ),
        (
            "pushed_pair_error_by_family.png",
            "best_stationarity",
            "pair_push_scaled_op",
            "Pushed pair error",
            "scaled pushed pair error",
            True,
        ),
        (
            "symmetric_relative_error_by_family.png",
            "best_stationarity",
            "gamma_tilde_eff_rel",
            "Symmetric relative weighted error",
            "relative error",
            False,
        ),
        (
            "bridge_operator_error_by_family.png",
            "all_checkpoints",
            "M_bridge_rel_frob",
            "Beta bridge operator error",
            "relative Frobenius error",
            False,
        ),
        (
            "pair_gain_weighted_contribution_by_family.png",
            "best_stationarity",
            "pair_weighted_contribution_max",
            "Gain-weighted pair contribution",
            "max contribution",
            True,
        ),
        (
            "pair_gain_correlation_by_family.png",
            "best_stationarity",
            "pair_gain_defect_corr_abs",
            "Pair defect versus stationarity gain",
            "absolute correlation",
            False,
        ),
        (
            "beta_identity_error_by_family.png",
            "all_checkpoints",
            "beta_corr_cv_product_abs",
            "Beta identity right-hand side",
            "absolute corr times CV product",
            False,
        ),
        (
            "leverage_damping_corr_by_family.png",
            "all_checkpoints",
            "leverage_damping_corr",
            "Residual damping versus leverage",
            "correlation",
            False,
        ),
        (
            "high_gain_closure_by_family.png",
            "best_stationarity",
            "pair_high_gain_closure_op",
            "High-gain scalar closure",
            "operator norm",
            True,
        ),
        (
            "pair_damping_bound_proxy_by_family.png",
            "best_stationarity",
            "pair_damping_bound_proxy",
            "Pair damping bound proxy",
            "proxy scale",
            True,
        ),
    ]

    for filename, block, metric, title, ylabel, logy in figure_specs:
        values = [metric_value(summary, family, block, metric) for family in families]
        save_bar_plot(families, values, title, ylabel, outdir / filename, logy=logy)


def copy_representative_trajectories(results_dir: Path, outdir: Path, families: Iterable[str]) -> None:
    for family in families:
        run_dir = results_dir / f"{family}_seed_0"
        for name in [
            "beta_collapse.png",
            "beta_fit_diagnostics.png",
            "beta_decomposition_identity.png",
            "high_gain_closure.png",
            "pair_defect_gain_scatter.png",
            "pair_direction_cumulative.png",
            "pair_gain_diagnostics.png",
            "phase_diagram.png",
            "residual_damping_vs_leverage.png",
            "two_regime_trajectory.png",
        ]:
            source = run_dir / name
            if source.exists():
                target = outdir / f"{family}_{name}"
                shutil.copyfile(source, target)


def write_manifest(outdir: Path) -> None:
    manifest = sorted(path.name for path in outdir.glob("*.png"))
    manifest_path = outdir / "manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for name in manifest:
            f.write(name + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final weighted-law figures.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/runs/pair_isotropy_with_low_rank"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("paper/figures"),
    )
    args = parser.parse_args()

    ensure_dir(args.outdir)
    summary_path = args.results_dir / "compact_summary.json"
    summary = load_compact_summary(summary_path)
    families = sorted_families(summary)

    make_summary_figures(summary_path, args.outdir)
    copy_representative_trajectories(args.results_dir, args.outdir, families)
    write_manifest(args.outdir)

    print(f"Saved final figures to: {args.outdir}")


if __name__ == "__main__":
    main()
