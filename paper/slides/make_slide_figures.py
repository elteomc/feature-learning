"""Generate the figures used by paper/slides/slides.tex.

Reads per-run ``history.json`` files produced by the experiment scripts and
writes vector PDFs into ``paper/slides/figures/``:

- ``two_regime_isotropic_seed_0.pdf``  trajectory of raw vs weighted law error
- ``weighted_law_residual.pdf``        weighted residual vs theorem bound, by family
- ``beta_tracking.pdf``                beta_fit / mean residual squared, by family
- ``pushed_pair_error.pdf``            pushed pair error vs global A_pair, by family
- ``phase_diagram.pdf``                beta-link error vs pushed pair error, all families

Run from the repository root:

    python paper/slides/make_slide_figures.py

Optional flags let you point at different result roots; the defaults match the
runs created by ``experiments/run_pair_isotropy.py`` and
``experiments/run_failure_modes.py --trained``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------

POSITIVE_FAMILIES = [
    ("isotropic", "Isotropic"),
    ("anisotropic", "Anisotropic"),
    ("low_rank_signal", "Low-rank signal"),
]

ADVERSARIAL_FAMILIES = [
    ("rare_hard_cluster", "Rare hard cluster"),
    ("rare_easy_cluster", "Rare easy cluster"),
    ("two_region_gating_stress", "Two-region gating"),
    ("mixture_subspaces_stress", "Mixture subspaces"),
    ("strong_anisotropic", "Strong anisotropic"),
    ("anisotropic_low_rank", "Anisotropic low-rank"),
]


def _load_history(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value: object) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _run_dirs(root: Path, family: str, seeds: Sequence[int]) -> List[Path]:
    dirs = []
    for seed in seeds:
        candidate = root / f"{family}_seed_{seed}"
        if (candidate / "history.json").exists():
            dirs.append(candidate)
    return dirs


def collect_family(
    root: Path, family: str, seeds: Sequence[int]
) -> List[Dict[str, object]]:
    payloads = []
    for run_dir in _run_dirs(root, family, seeds):
        payload = _load_history(run_dir / "history.json")
        if payload is not None:
            payloads.append(payload)
    return payloads


# ----------------------------------------------------------------------------
# small plotting helpers
# ----------------------------------------------------------------------------


def _bar_with_error(
    ax: "matplotlib.axes.Axes",
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
    *,
    logy: bool = False,
    ylabel: str = "",
    title: str = "",
    hline: Optional[float] = None,
) -> None:
    n_groups = len(labels)
    n_series = len(series)
    width = 0.8 / max(1, n_series)
    xs = list(range(n_groups))
    for k, (name, values, color) in enumerate(series):
        offsets = [x + (k - (n_series - 1) / 2) * width for x in xs]
        ax.bar(offsets, values, width=width, label=name, color=color, edgecolor="white", linewidth=0.4)
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8.5)
    if logy:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    if n_series > 1:
        ax.legend(fontsize=8, frameon=False)


def _agg(values: Sequence[Optional[float]]) -> Tuple[float, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return float("nan"), 0.0
    return mean(clean), (pstdev(clean) if len(clean) > 1 else 0.0)


# ----------------------------------------------------------------------------
# figure 1: two-regime trajectory
# ----------------------------------------------------------------------------


def make_trajectory_figure(history_payload: Dict[str, object], out_path: Path) -> None:
    history = history_payload["history"]  # type: ignore[index]
    best_stat = history_payload.get("best_metrics_by_stationarity", {})

    def series(key: str) -> Tuple[List[float], List[float]]:
        xs, ys = [], []
        for cp in history:  # type: ignore[union-attr]
            v = _finite(cp.get(key))
            s = _finite(cp.get("step"))
            if v is not None and v > 0 and s is not None:
                xs.append(s)
                ys.append(v)
        return xs, ys

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    specs = [
        ("gamma_tilde_eff_op", "weighted-law error $\\|H^2-\\kappa_{\\mathrm{eff}}\\widetilde G\\|$", "#1f7a3d", "-", "o"),
        ("gamma_eff_op", "raw-law error $\\|H^2-\\kappa_{\\mathrm{eff}}\\beta G\\|$", "#c8501e", "-", "s"),
        ("beta_fit", r"$\beta_{\mathrm{fit}}$", "#2b5fa8", "--", "^"),
        ("resid_mean_sq", r"mean residual $\bar r^2$", "#888888", ":", None),
    ]
    for key, label, color, ls, marker in specs:
        xs, ys = series(key)
        if not xs:
            continue
        ax.plot(xs, ys, ls, color=color, marker=marker, markersize=3, linewidth=1.4, label=label)

    s_stat = _finite(best_stat.get("step")) if isinstance(best_stat, dict) else None
    if s_stat is not None:
        ax.axvline(s_stat, color="#1f7a3d", linestyle="-.", linewidth=1.0, alpha=0.7)
        ax.text(s_stat, ax.get_ylim()[1], " best stationarity ", color="#1f7a3d", fontsize=8,
                ha="right", va="top", rotation=90)

    ax.set_yscale("log")
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("operator norm / energy", fontsize=9)
    ax.set_title("Two-regime trajectory (isotropic, seed 0)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# figures 2-4: per-family bars
# ----------------------------------------------------------------------------


def _best_stat_metric(payloads: Sequence[Dict[str, object]], key: str) -> List[Optional[float]]:
    out = []
    for p in payloads:
        block = p.get("best_metrics_by_stationarity", {})
        out.append(_finite(block.get(key)) if isinstance(block, dict) else None)
    return out


def _all_checkpoint_metric(payloads: Sequence[Dict[str, object]], key: str) -> List[Optional[float]]:
    out = []
    for p in payloads:
        for cp in p.get("history", []):  # type: ignore[union-attr]
            out.append(_finite(cp.get(key)))
    return out


def make_family_bar_figures(
    positive: Dict[str, List[Dict[str, object]]], outdir: Path
) -> None:
    labels = [label for _, label in POSITIVE_FAMILIES]
    keys = [key for key, _ in POSITIVE_FAMILIES]

    # weighted-law residual vs theorem bound
    resid = [_agg(_best_stat_metric(positive[k], "gamma_tilde_eff_op")) for k in keys]
    bound = [_agg(_best_stat_metric(positive[k], "theorem_bound_op")) for k in keys]
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    _bar_with_error(
        ax,
        labels,
        [
            ("observed residual", [m for m, _ in resid], "#1f7a3d"),
            ("theorem bound", [m for m, _ in bound], "#9ec9ad"),
        ],
        logy=True,
        ylabel="operator norm",
        title="Weighted law: residual vs. bound",
    )
    fig.tight_layout()
    fig.savefig(outdir / "weighted_law_residual.pdf")
    plt.close(fig)

    # beta tracking
    beta = [_agg(_all_checkpoint_metric(positive[k], "beta_over_resid_mean_sq")) for k in keys]
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    means = [m for m, _ in beta]
    errs = [s for _, s in beta]
    xs = list(range(len(labels)))
    ax.bar(xs, means, yerr=errs, capsize=3, color="#2b5fa8", edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$\beta_{\mathrm{fit}}\,/\,\bar r^2$", fontsize=9)
    ax.set_title("Beta link: tracks mean residual energy", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "beta_tracking.pdf")
    plt.close(fig)

    # pushed pair error vs global A_pair
    pushed = [_agg(_best_stat_metric(positive[k], "pair_push_scaled_op")) for k in keys]
    a_pair = [_agg(_best_stat_metric(positive[k], "A_pair_op")) for k in keys]
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    _bar_with_error(
        ax,
        labels,
        [
            (r"global $\mathcal{A}_{\mathrm{pair}}$", [m for m, _ in a_pair], "#c8501e"),
            ("pushed pair error", [m for m, _ in pushed], "#1f7a3d"),
        ],
        logy=True,
        ylabel="operator norm",
        title="Pair: global defect is conservative",
    )
    fig.tight_layout()
    fig.savefig(outdir / "pushed_pair_error.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# figure 5: phase diagram
# ----------------------------------------------------------------------------


_PHASE_STYLE = {
    # positive families: greens
    "isotropic": ("#1f7a3d", "o", "Isotropic"),
    "anisotropic": ("#2b8f4f", "v", "Anisotropic (mild)"),
    "low_rank_signal": ("#5bb87f", "D", "Low-rank signal"),
    # adversarial families: warm / cool accents
    "rare_hard_cluster": ("#c0392b", "s", "Rare hard cluster"),
    "rare_easy_cluster": ("#e08a3c", "P", "Rare easy cluster"),
    "two_region_gating_stress": ("#8e44ad", "X", "Two-region gating"),
    "mixture_subspaces_stress": ("#2980b9", "^", "Mixture subspaces"),
    "strong_anisotropic": ("#7f8c8d", "*", "Anisotropic (strong)"),
    "anisotropic_low_rank": ("#d4348e", "h", "Anisotropic low-rank"),
}


def _phase_point(payload: Dict[str, object]) -> Optional[Tuple[float, float]]:
    """(x, y) for one run: x = beta-link error at the best raw-AGOP checkpoint,
    y = pushed pair error = weighted-law residual at the best-stationarity checkpoint."""
    raw_block = payload.get("best_metrics_by_raw_conditioning", {})
    stat_block = payload.get("best_metrics_by_stationarity", {})
    if not isinstance(raw_block, dict) or not isinstance(stat_block, dict):
        return None
    beta_err = _finite(raw_block.get("beta_corr_cv_product_abs"))
    if beta_err is None:
        ratio = _finite(raw_block.get("beta_over_resid_mean_sq"))
        beta_err = abs(ratio - 1.0) if ratio is not None else None
    pushed = _finite(stat_block.get("pair_push_scaled_op"))
    if beta_err is None or pushed is None:
        return None
    return (beta_err, pushed)


def make_phase_diagram(
    positive: Dict[str, List[Dict[str, object]]],
    adversarial: Dict[str, List[Dict[str, object]]],
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    eps_x, eps_y = 1e-3, 3e-9

    all_groups: List[Tuple[str, List[Dict[str, object]]]] = list(positive.items()) + list(adversarial.items())
    for family, payloads in all_groups:
        if family not in _PHASE_STYLE:
            continue
        color, marker, label = _PHASE_STYLE[family]
        pts = [p for p in (_phase_point(pl) for pl in payloads) if p is not None]
        if not pts:
            continue
        xs = sorted(x + eps_x for x, _ in pts)
        ys = sorted(y + eps_y for _, y in pts)

        def _median(v: List[float]) -> float:
            m = len(v) // 2
            return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])

        mx, my = _median(xs), _median(ys)
        # faint per-seed cloud
        ax.scatter(xs, ys, s=24, c=color, marker=marker, alpha=0.28, linewidth=0, zorder=2)
        # min--max whiskers
        ax.plot([min(xs), max(xs)], [my, my], color=color, linewidth=1.0, alpha=0.5, zorder=2)
        ax.plot([mx, mx], [min(ys), max(ys)], color=color, linewidth=1.0, alpha=0.5, zorder=2)
        # family median marker
        ax.scatter([mx], [my], s=150, c=color, marker=marker, edgecolor="white", linewidth=1.1,
                   label=label, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"beta-link error at best raw-AGOP checkpoint $\;|\beta_{\mathrm{fit}}/\bar r^2 - 1|$", fontsize=9.5)
    ax.set_ylabel(r"weighted-law error $\;\|B^\top(S-d_{\mathrm{eff}}T)B\|/(\lambda^2 n^2)$", fontsize=9.5)
    ax.set_title("Trained families on the bridge map (medians over 5 seeds, whiskers span the seeds)", fontsize=9.5)
    ax.grid(alpha=0.3, which="both")

    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    ax.text(xl * 1.25, yb * 1.4, "both links healthy\n(raw + weighted OK)", fontsize=7.6, color="#1f7a3d",
            ha="left", va="bottom")
    ax.text(xr / 1.08, yb * 1.4, "beta link stressed\n(raw biased, weighted OK)", fontsize=7.6, color="#c0392b",
            ha="right", va="bottom")
    ax.text(xl * 1.25, yt / 1.06, "weighted law degrades", fontsize=7.6, color="#8e44ad",
            ha="left", va="top")

    ax.legend(fontsize=7.8, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(outdir / "phase_diagram.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build slide figures from experiment outputs.")
    parser.add_argument("--positive-root", type=Path, default=Path("results/runs/phase_diagram_positive"))
    parser.add_argument(
        "--adversarial-root",
        type=Path,
        default=Path("results/runs/phase_diagram_adversarial/trained"),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--outdir", type=Path, default=Path("paper/slides/figures"))
    args = parser.parse_args()

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    positive = {key: collect_family(args.positive_root, key, seeds) for key, _ in POSITIVE_FAMILIES}
    adversarial = {key: collect_family(args.adversarial_root, key, seeds) for key, _ in ADVERSARIAL_FAMILIES}

    missing_pos = [k for k, v in positive.items() if not v]
    if missing_pos:
        raise SystemExit(f"No runs found for positive families {missing_pos} under {args.positive_root}")

    iso0 = _load_history(args.positive_root / "isotropic_seed_0" / "history.json")
    if iso0 is None and positive["isotropic"]:
        iso0 = positive["isotropic"][0]
    if iso0 is not None:
        make_trajectory_figure(iso0, args.outdir / "two_regime_isotropic_seed_0.pdf")

    make_family_bar_figures(positive, args.outdir)

    missing_adv = [k for k, v in adversarial.items() if not v]
    if missing_adv:
        print(f"[warn] no runs for adversarial families {missing_adv}; phase diagram will omit them")
    make_phase_diagram(positive, adversarial, args.outdir)

    print(f"Wrote slide figures to {args.outdir}")


if __name__ == "__main__":
    main()
