"""Distill per-checkpoint history into a small JSON the demo can fetch().

The full per-seed history files are >300 KB each. The demo only needs a few
metrics for one representative seed per family, so we extract them here and
write a tiny file that the static page can ship.

Run from the repository root:

    python apps/weighted-law-explorer/build_trajectories.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from src.train_two_layer import (
    ExperimentConfig,
    generate_teacher_student_dataset,
    set_seed,
    torch_dtype_from_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "results" / "runs" / "pair_isotropy_with_low_rank"
OUTPUT = Path(__file__).resolve().parent / "data" / "trajectories.json"

FAMILY_LABELS = {
    "isotropic": "Isotropic Gaussian",
    "anisotropic": "Anisotropic Gaussian",
    "low_rank_signal": "Low-rank signal",
}

METRIC_KEYS = (
    "loss_total",
    "resid_rms",
    "resid_mean_sq",
    "beta_fit",
    "gamma_tilde_eff_rel_h2",
    "pair_push_scaled_op",
    "theorem_bound_ratio",
    "stationarity_rel",
    "H2_op",
    "G_tilde_op",
    "kappa_eff",
)


def _safe(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _var_y_for_family(config: dict, seed: int) -> float:
    """Recompute var(y) for the recorded family by replaying data generation."""
    cfg = ExperimentConfig(
        name=f"var_y_seed_{seed}",
        seed=seed,
        data_family=config.get("data_family", "gaussian"),
        n=int(config.get("n", 128)),
        d=int(config.get("d", 256)),
        signal_rank=int(config.get("signal_rank", 8)),
        signal_noise_std=float(config.get("signal_noise_std", 0.5)),
        m_teacher=int(config.get("m_teacher", 4)),
        m_student=int(config.get("m_student", 16)),
        anisotropic=bool(config.get("anisotropic", False)),
        spectrum_min=float(config.get("spectrum_min", 1e-2)),
        noise_std=float(config.get("noise_std", 0.0)),
        rare_fraction=float(config.get("rare_fraction", 0.1)),
        rare_shift=float(config.get("rare_shift", 4.0)),
        rare_label_scale=float(config.get("rare_label_scale", 3.0)),
    )
    dtype = torch_dtype_from_name(cfg.dtype)
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    data = generate_teacher_student_dataset(cfg, dtype=dtype, device=device)
    return float(torch.var(data["y"], unbiased=False).item())


def extract(family: str, seed: int = 0) -> dict:
    history_path = RUN_DIR / f"{family}_seed_{seed}" / "history.json"
    raw = json.loads(history_path.read_text(encoding="utf-8"))
    history = raw.get("history", [])
    rows = []
    for entry in history:
        row = {"step": entry.get("step"), "phase": entry.get("phase")}
        for key in METRIC_KEYS:
            row[key] = _safe(entry.get(key))
        rows.append(row)

    best_stat = raw.get("best_metrics_by_stationarity", {})
    config = raw.get("config", {})
    var_y = _var_y_for_family(config, seed)
    return {
        "label": FAMILY_LABELS.get(family, family),
        "seed": seed,
        "config": {
            "n": config.get("n"),
            "d": config.get("d"),
            "m_teacher": config.get("m_teacher"),
            "m_student": config.get("m_student"),
            "data_family": config.get("data_family"),
        },
        "var_y": var_y,
        "best_step": best_stat.get("step"),
        "best_metrics": {key: _safe(best_stat.get(key)) for key in METRIC_KEYS},
        "history": rows,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "source": str(RUN_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "metric_keys": list(METRIC_KEYS),
        "families": {family: extract(family) for family in FAMILY_LABELS},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload), encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
