from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt

from src.failure_regimes import run_all_toy_regimes
from src.train_two_layer import ExperimentConfig, train_one

from experiments.run_pair_isotropy import build_summary


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _labels_and_values(rows: Iterable[Dict[str, object]], key: str) -> tuple[List[str], List[float]]:
    labels: List[str] = []
    values: List[float] = []
    for row in rows:
        labels.append(str(row["name"]))
        values.append(float(row[key]))
    return labels, values


def save_bar_plot(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = list(range(len(labels)))
    ax.bar(xs, values)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_beta_failure_plot(beta_examples: List[Dict[str, object]], output_path: Path) -> None:
    labels, ratios = _labels_and_values(beta_examples, "beta_over_mean_residual_sq")
    save_bar_plot(
        labels=labels,
        values=ratios,
        title="Beta bridge failure toy examples",
        ylabel="beta_fit / mean residual squared",
        output_path=output_path,
        logy=True,
    )


def save_pair_failure_plot(pair_examples: List[Dict[str, object]], output_path: Path) -> None:
    labels = [str(row["name"]) for row in pair_examples]
    a_pair = [float(row["global_a_pair"]) for row in pair_examples]
    pushed = [float(row["pushed_pair_proxy"]) for row in pair_examples]
    xs = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar([x - width / 2 for x in xs], a_pair, width=width, label="global A_pair")
    ax.bar([x + width / 2 for x in xs], pushed, width=width, label="pushed pair proxy")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_title("Pair closure failure toy examples")
    ax.set_ylabel("diagnostic scale")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_regime_map(regime_points: List[Dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for point in regime_points:
        x = float(point["beta_covariance_abs"])
        y = float(point["high_gain_pair_defect"])
        label = str(point["name"])
        ax.scatter(x, y, s=90)
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("absolute leverage-residual covariance")
    ax.set_ylabel("high-gain pair defect")
    ax.set_title("Failure-mode regime map")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def metric_from_row(row: Dict[str, object], block: str, metric: str) -> float:
    metrics = row.get(block)
    if not isinstance(metrics, dict):
        return float("nan")
    value = metrics.get(metric)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def save_trained_smoke_plots(rows: List[Dict[str, object]], figure_dir: Path) -> None:
    ensure_dir(figure_dir)
    labels = [str(row["name"]) for row in rows]
    save_bar_plot(
        labels=labels,
        values=[
            metric_from_row(row, "best_metrics_by_stationarity", "beta_over_resid_mean_sq")
            for row in rows
        ],
        title="Trained smoke beta ratio",
        ylabel="beta_fit / mean residual squared",
        output_path=figure_dir / "trained_beta_ratio_smoke.png",
        logy=False,
    )
    save_bar_plot(
        labels=labels,
        values=[
            metric_from_row(row, "best_metrics_by_stationarity", "pair_high_gain_closure_op")
            for row in rows
        ],
        title="Trained smoke high-gain pair closure",
        ylabel="operator norm",
        output_path=figure_dir / "trained_high_gain_closure_smoke.png",
        logy=True,
    )
    save_bar_plot(
        labels=labels,
        values=[
            metric_from_row(row, "best_metrics_by_stationarity", "pair_push_scaled_op")
            for row in rows
        ],
        title="Trained smoke pushed pair error",
        ylabel="scaled pushed pair error",
        output_path=figure_dir / "trained_pushed_pair_smoke.png",
        logy=True,
    )


def run_toy(output_root: Path, figure_dir: Path) -> None:
    payload = run_all_toy_regimes()
    write_json(output_root / "toy" / "summary.json", payload)

    ensure_dir(figure_dir)
    save_beta_failure_plot(
        beta_examples=payload["beta_examples"],  # type: ignore[arg-type]
        output_path=figure_dir / "beta_failure_toy.png",
    )
    save_pair_failure_plot(
        pair_examples=payload["pair_examples"],  # type: ignore[arg-type]
        output_path=figure_dir / "pair_failure_toy.png",
    )
    save_regime_map(
        regime_points=payload["regime_points"],  # type: ignore[arg-type]
        output_path=figure_dir / "regime_map.png",
    )


def trained_configs(output_root: Path, fast: bool, seeds: List[int] | None = None) -> List[ExperimentConfig]:
    seeds = list(seeds) if seeds else [0]
    trained_root = str(output_root / "trained")
    configs: List[ExperimentConfig] = []
    for seed in seeds:
        configs.extend(
            [
                ExperimentConfig(
                    name=f"rare_hard_cluster_seed_{seed}",
                    seed=seed,
                    data_family="rare_region_outliers",
                    rare_fraction=0.1,
                    rare_shift=4.0,
                    rare_label_scale=4.0,
                    noise_std=0.0,
                    output_root=trained_root,
                ),
                ExperimentConfig(
                    name=f"rare_easy_cluster_seed_{seed}",
                    seed=seed,
                    data_family="rare_region_outliers",
                    rare_fraction=0.1,
                    rare_shift=4.0,
                    rare_label_scale=0.4,
                    noise_std=0.0,
                    output_root=trained_root,
                ),
                ExperimentConfig(
                    name=f"two_region_gating_stress_seed_{seed}",
                    seed=seed,
                    data_family="two_region_gating",
                    signal_noise_std=0.25,
                    noise_std=0.0,
                    output_root=trained_root,
                ),
                # Low-rank, low-noise mixture: X concentrates on a small union of
                # subspaces, so X^T X is far from scalar on the directions QR uses.
                ExperimentConfig(
                    name=f"mixture_subspaces_stress_seed_{seed}",
                    seed=seed,
                    data_family="mixture_subspaces",
                    signal_rank=3,
                    signal_noise_std=0.05,
                    noise_std=0.0,
                    output_root=trained_root,
                ),
                # Strongly anisotropic Gaussian inputs (covariance spectrum 1 down
                # to 1e-4). Full-rank, so the sample Gram X^T X stays close to a
                # scalar in high dimension; included as the "anisotropy alone does
                # not break the weighted law" reference point.
                ExperimentConfig(
                    name=f"strong_anisotropic_seed_{seed}",
                    seed=seed,
                    data_family="gaussian",
                    anisotropic=True,
                    spectrum_min=1e-4,
                    noise_std=0.0,
                    output_root=trained_root,
                ),
                # Low-rank signal with a steeply anisotropic within-subspace
                # covariance; the teacher uses the same subspace, so X^T X is
                # genuinely non-scalar on the directions the network relies on.
                # This is the intended pure high-gain pair-closure failure.
                ExperimentConfig(
                    name=f"anisotropic_low_rank_seed_{seed}",
                    seed=seed,
                    data_family="anisotropic_low_rank",
                    signal_rank=10,
                    spectrum_min=0.08,
                    signal_noise_std=0.05,
                    noise_std=0.0,
                    output_root=trained_root,
                ),
            ]
        )
    if fast:
        for cfg in configs:
            cfg.sgd_steps = 300
            cfg.checkpoint_every = 100
            cfg.lbfgs_outer_steps = 8
            cfg.lbfgs_log_every = 4
    return configs


def run_trained(output_root: Path, figure_dir: Path, fast: bool, seeds: List[int] | None = None) -> None:
    configs = trained_configs(output_root, fast, seeds)
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        result = train_one(cfg)
        rows.append(
            {
                "name": cfg.name,
                "output_dir": str(result["output_dir"]),
                "config": asdict(cfg),
                "final_metrics": result["final_metrics"],
                "best_metrics_by_stationarity": result["best_metrics_by_stationarity"],
                "best_metrics_by_weighted_law": result["best_metrics_by_weighted_law"],
                "best_metrics_by_raw_conditioning": result["best_metrics_by_raw_conditioning"],
                "crossover_metrics": result["crossover_metrics"],
                "_history": result["history"],
            }
        )

    summary = build_summary(rows)
    write_json(output_root / "trained" / "summary.json", summary)
    write_json(output_root / "trained" / "compact_summary.json", summary["groups"])  # type: ignore[arg-type]
    write_json(output_root / "trained" / "run_summary.json", {"runs": rows})
    save_trained_smoke_plots(rows, figure_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run failure-mode taxonomy checks.")
    parser.add_argument("--toy", action="store_true", help="Run deterministic toy regimes.")
    parser.add_argument(
        "--trained",
        action="store_true",
        help="Reserved for trained-network failure-mode smoke runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/failure_modes"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("paper/figures/failure_modes"),
    )
    parser.add_argument("--fast", action="store_true", help="Use a short trained smoke schedule.")
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seeds for the trained failure-mode families.",
    )
    args = parser.parse_args()

    if not args.toy and not args.trained:
        args.toy = True

    if args.toy:
        run_toy(args.output_root, args.figure_dir)

    if args.trained:
        seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
        run_trained(args.output_root, args.figure_dir, args.fast, seeds)


if __name__ == "__main__":
    main()
