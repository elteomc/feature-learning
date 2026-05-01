from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt

from src.failure_regimes import run_all_toy_regimes


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
    args = parser.parse_args()

    if not args.toy and not args.trained:
        args.toy = True

    if args.toy:
        run_toy(args.output_root, args.figure_dir)

    if args.trained:
        raise NotImplementedError(
            "Trained failure-mode smoke runs will be added after toy regimes are validated."
        )


if __name__ == "__main__":
    main()
