"""Build a small, demo-friendly snapshot of feature matrices.

For each of the three reported families (isotropic, anisotropic, low-rank
signal) and seed 0, this script reruns training (the same seed and config used
for `results/runs/pair_isotropy_with_low_rank`) and captures, at three
checkpoints (init, crossover, best stationarity):

- the student-feature gram matrix `B B^T` (m_student x m_student)
- the alignment matrix `B B_star^T` (m_student x m_teacher)
- the top-k eigenvalues of `H^2 = (B^T B)^2` and of `kappa_eff * G_tilde`

It also stores `var(y)` and the per-checkpoint `resid_mean_sq` so the demo can
plot R^2.

Run from the repository root:

    python apps/weighted-law-explorer/build_matrix_snapshots.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.train_two_layer import (
    ExperimentConfig,
    TwoLayerSoftplus,
    compute_q,
    generate_teacher_student_dataset,
    objective,
    set_seed,
    torch_dtype_from_name,
)
from src.weighted_metrics import compute_weighted_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "results" / "runs" / "pair_isotropy_with_low_rank"
OUTPUT = Path(__file__).resolve().parent / "data" / "matrix_snapshots.json"

FAMILY_LABELS = {
    "isotropic": "Isotropic Gaussian",
    "anisotropic": "Anisotropic Gaussian",
    "low_rank_signal": "Low-rank signal",
}


def family_config(family: str, seed: int = 0) -> ExperimentConfig:
    """Return the same config used for the reported run."""
    if family == "isotropic":
        return ExperimentConfig(
            name=f"isotropic_seed_{seed}",
            seed=seed,
            data_family="gaussian",
            anisotropic=False,
            noise_std=0.0,
            spectrum_min=1e-2,
            beta_threshold=1e-5,
        )
    if family == "anisotropic":
        return ExperimentConfig(
            name=f"anisotropic_seed_{seed}",
            seed=seed,
            data_family="gaussian",
            anisotropic=True,
            noise_std=0.0,
            spectrum_min=1e-2,
            beta_threshold=1e-5,
        )
    if family == "low_rank_signal":
        return ExperimentConfig(
            name=f"low_rank_signal_seed_{seed}",
            seed=seed,
            data_family="low_rank_signal",
            anisotropic=False,
            signal_rank=8,
            signal_noise_std=0.5,
            noise_std=0.0,
            beta_threshold=1e-5,
        )
    raise ValueError(family)


def reference_steps(cfg: ExperimentConfig) -> Dict[str, int]:
    """Pick three SGD checkpoints that span the visible feature-learning phase.

    The reported run does extra LBFGS polishing after SGD, but for visualization
    we want three SGD checkpoints that illustrate the random init, an
    early-training state, and the late-training state.
    """
    return {
        "init": 0,
        "mid": min(100, cfg.sgd_steps),
        "late": cfg.sgd_steps,
    }


def _matrix_to_list(matrix: torch.Tensor) -> List[List[float]]:
    return [[float(value) for value in row] for row in matrix.detach().cpu()]


def _vector_to_list(vector: torch.Tensor) -> List[float]:
    return [float(value) for value in vector.detach().cpu()]


def snapshot(
    *,
    model: TwoLayerSoftplus,
    data: Dict[str, torch.Tensor],
    cfg: ExperimentConfig,
    label: str,
    step: int,
    extra_keys: int,
) -> Dict[str, object]:
    """Compute matrices and spectra at the current model state."""
    X = data["X"]
    y = data["y"]
    B_star = data["B_star"]
    with torch.no_grad():
        B = model.B.detach().clone()
        Q = compute_q(model, X)
        preds = model(X)
        residuals = preds - y
        loss_value = float(objective(model, X, y, cfg.lambda_B).item())

    metrics = compute_weighted_metrics(
        B=B,
        X=X,
        Q=Q,
        residuals=residuals,
        lambda_B=cfg.lambda_B,
        loss_total=loss_value,
    )

    BBt = B @ B.T
    align = B @ B_star.T
    align_abs = align.abs()

    sigma = torch.linalg.svdvals(B)
    h2_eigs = (sigma * sigma * sigma * sigma)
    h2_eigs, _ = torch.sort(h2_eigs, descending=True)

    M_tilde = (Q.T @ (residuals.square().unsqueeze(1) * Q)) / float(X.shape[0])
    M_tilde_sym = 0.5 * (M_tilde + M_tilde.T)
    eigvals, eigvecs = torch.linalg.eigh(M_tilde_sym)
    eigvals = eigvals.clamp(min=0)
    M_tilde_sqrt = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T
    G_compact = M_tilde_sqrt @ (B @ B.T) @ M_tilde_sqrt
    G_compact = 0.5 * (G_compact + G_compact.T)
    g_eigs_compact = torch.linalg.eigvalsh(G_compact)
    g_eigs_compact, _ = torch.sort(g_eigs_compact.clamp(min=0), descending=True)
    kappa = metrics["kappa_eff"] if math.isfinite(metrics["kappa_eff"]) else float("nan")
    kappa_g_eigs = (kappa * g_eigs_compact) if math.isfinite(kappa) else g_eigs_compact

    extra = max(0, extra_keys - h2_eigs.shape[0])
    if extra:
        h2_eigs = torch.cat([h2_eigs, torch.zeros(extra)])
        kappa_g_eigs = torch.cat([kappa_g_eigs, torch.zeros(extra)])

    h2_eigs = h2_eigs[:extra_keys]
    kappa_g_eigs = kappa_g_eigs[:extra_keys]

    return {
        "label": label,
        "step": int(step),
        "BBt": _matrix_to_list(BBt),
        "alignment": _matrix_to_list(align),
        "alignment_abs": _matrix_to_list(align_abs),
        "h2_eigs": _vector_to_list(h2_eigs),
        "kappa_g_tilde_eigs": _vector_to_list(kappa_g_eigs),
        "kappa_eff": float(kappa) if math.isfinite(kappa) else None,
        "loss_total": float(loss_value),
        "resid_mean_sq": float(metrics["resid_mean_sq"]),
        "resid_max_sq": float(metrics["resid_max_sq"]),
        "beta_fit": float(metrics["beta_fit"]),
        "gamma_tilde_eff_rel_h2": float(metrics["gamma_tilde_eff_rel_h2"]),
        "theorem_bound_ratio": float(metrics["theorem_bound_ratio"]),
        "stationarity_rel": float(metrics["stationarity_rel"]),
    }


def run_family(family: str, seed: int = 0) -> Dict[str, object]:
    cfg = family_config(family, seed=seed)
    steps_of_interest = reference_steps(cfg)

    dtype = torch_dtype_from_name(cfg.dtype)
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    data = generate_teacher_student_dataset(cfg, dtype=dtype, device=device)
    model = TwoLayerSoftplus(cfg, dtype=dtype, device=device)
    var_y = float(torch.var(data["y"], unbiased=False).item())

    snapshots: List[Dict[str, object]] = []
    target_steps = sorted(set(steps_of_interest.values()))
    label_for_step = {step: label for label, step in steps_of_interest.items()}

    if 0 in target_steps:
        snapshots.append(snapshot(
            model=model, data=data, cfg=cfg,
            label=label_for_step[0], step=0, extra_keys=cfg.m_student,
        ))
        target_steps = [s for s in target_steps if s > 0]

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.sgd_lr)
    last_step = target_steps[-1] if target_steps else 0

    for step in range(1, last_step + 1):
        optimizer.zero_grad()
        loss = objective(model, data["X"], data["y"], cfg.lambda_B)
        loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        optimizer.step()

        if step in target_steps:
            snapshots.append(snapshot(
                model=model, data=data, cfg=cfg,
                label=label_for_step[step], step=step, extra_keys=cfg.m_student,
            ))

    return {
        "label": FAMILY_LABELS.get(family, family),
        "seed": seed,
        "config": {
            "n": cfg.n,
            "d": cfg.d,
            "m_teacher": cfg.m_teacher,
            "m_student": cfg.m_student,
            "data_family": cfg.data_family,
        },
        "var_y": var_y,
        "snapshots": snapshots,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "source": str(RUN_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "families": {family: run_family(family) for family in FAMILY_LABELS},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload), encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
