#!/usr/bin/env python3
"""
Proposal 3 synthetic experiment (patched version)

Main changes vs the original script
-----------------------------------
1. Better theory-side scaling:
   The old script used rho^2 where rho = mean residual, which becomes unstable near
   interpolation because positive/negative residuals cancel.
   We now log two improved candidates:

       tau_r2 = mean_i r_i^2
       tau_q2 = (sum_i r_i^2 ||q_i||^2) / (sum_i ||q_i||^2)

   The more theory-motivated one is tau_q2, since
       Q^T R^2 Q = sum_i r_i^2 q_i q_i^T
   and if this is approximately scalar on the hidden-feature space, the scalar should
   be a Q-weighted residual second moment rather than (mean residual)^2.

   We therefore define:
       kappa_q2 = d * tau_q2 / (lambda^2 n)
       c_q2     = 1 / kappa_q2

   and log the corresponding errors:
       delta_q2 = ||B B^T - kappa_q2 M||_op
       gamma_q2 = ||G - c_q2 H^2||_op

2. Better anisotropy diagnostics:
   The old alpha_dataset = ||XX^T/d - I|| can be misleading when d > n.
   We now separate:
       - row-Gram sampling defect: alpha_row_raw
       - whitened row-Gram defect: alpha_row_whitened
       - population anisotropy of Sigma itself:
            cov_condition_number
            cov_iso_error = max_j |lambda_j / mean(lambda) - 1|

   Since this is synthetic data, the population covariance eigenvalues are known.
   That makes the anisotropic ablation much cleaner.

Outputs
-------
For each config, saves:
  - history.json
  - training_curves.png
  - eigen_comparison.png
  - final_matrix_summary.json

Run:
  python proposal3_baseline_experiment_v2.py
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def symmetrize(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.T)


def op_norm_symmetric(mat: torch.Tensor) -> float:
    mat = symmetrize(mat)
    evals = torch.linalg.eigvalsh(mat)
    return float(torch.max(torch.abs(evals)).item())


def frob_norm(mat: torch.Tensor) -> float:
    return float(torch.linalg.norm(mat, ord="fro").item())


def best_scalar_fit(target: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12) -> float:
    denom = torch.sum(basis * basis).item()
    if abs(denom) < eps:
        return float("nan")
    numer = torch.sum(target * basis).item()
    return numer / denom


def subspace_overlap(A: torch.Tensor, B: torch.Tensor, rank: int) -> float:
    A = symmetrize(A)
    B = symmetrize(B)
    _, evecs_A = torch.linalg.eigh(A)
    _, evecs_B = torch.linalg.eigh(B)
    UA = evecs_A[:, -rank:]
    UB = evecs_B[:, -rank:]
    overlap = torch.linalg.norm(UA.T @ UB, ord="fro").item() ** 2
    return float(overlap)


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return mapping[name]


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 0
    n: int = 128
    d: int = 256
    m_teacher: int = 4
    m_student: int = 16
    teacher_B_scale: float = 1.0
    teacher_a_scale: float = 1.0
    student_B_scale: float = 0.5
    student_a_scale: float = 0.5
    steps: int = 3000
    lr: float = 5e-2
    lambda_B: float = 1e-2
    checkpoint_every: int = 20
    grad_tol: float = 1e-8
    noise_std: float = 0.0
    anisotropic: bool = False
    spectrum_min: float = 1e-2
    dtype: str = "float64"
    device: str = "cpu"
    top_rank: int = 4
    rho_tol: float = 1e-12
    output_root: str = "proposal3_outputs_v2"


def sample_inputs(cfg: ExperimentConfig, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    Z = torch.randn(cfg.n, cfg.d, dtype=dtype, device=device)
    if not cfg.anisotropic:
        eigvals = torch.ones(cfg.d, dtype=dtype, device=device)
        return Z, eigvals

    log_min = math.log(cfg.spectrum_min)
    exponents = torch.linspace(0.0, 1.0, cfg.d, dtype=dtype, device=device)
    eigvals = torch.exp(log_min * exponents)
    X = Z * torch.sqrt(eigvals).unsqueeze(0)
    return X, eigvals


def softplus_hidden(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return F.softplus(X @ B.T)


def sample_teacher(cfg: ExperimentConfig, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    B_star = (
        cfg.teacher_B_scale
        * torch.randn(cfg.m_teacher, cfg.d, dtype=dtype, device=device)
        / math.sqrt(cfg.d)
    )
    a_star = (
        cfg.teacher_a_scale
        * torch.randn(cfg.m_teacher, dtype=dtype, device=device)
        / math.sqrt(cfg.m_teacher)
    )
    return B_star, a_star


def generate_teacher_student_dataset(cfg: ExperimentConfig, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor]:
    X, eigvals = sample_inputs(cfg, dtype=dtype, device=device)
    B_star, a_star = sample_teacher(cfg, dtype=dtype, device=device)
    y_clean = softplus_hidden(X, B_star) @ a_star
    y = y_clean + cfg.noise_std * torch.randn_like(y_clean)
    return {
        "X": X,
        "y": y,
        "y_clean": y_clean,
        "B_star": B_star,
        "a_star": a_star,
        "cov_eigvals": eigvals,
    }


class TwoLayerSoftplus(nn.Module):
    def __init__(self, cfg: ExperimentConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        B0 = (
            cfg.student_B_scale
            * torch.randn(cfg.m_student, cfg.d, dtype=dtype, device=device)
            / math.sqrt(cfg.d)
        )
        a0 = (
            cfg.student_a_scale
            * torch.randn(cfg.m_student, dtype=dtype, device=device)
            / math.sqrt(cfg.m_student)
        )
        self.B = nn.Parameter(B0)
        self.a = nn.Parameter(a0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return F.softplus(X @ self.B.T) @ self.a


@torch.no_grad()
def compute_q(model: TwoLayerSoftplus, X: torch.Tensor) -> torch.Tensor:
    Z = X @ model.B.T
    return torch.sigmoid(Z) * model.a.unsqueeze(0)


@torch.no_grad()
def compute_dataset_diagnostics(X: torch.Tensor, cov_eigvals: torch.Tensor) -> Dict[str, float]:
    n, d = X.shape
    I_n = torch.eye(n, dtype=X.dtype, device=X.device)

    alpha_row_raw = op_norm_symmetric((X @ X.T) / float(d) - I_n)

    Xw = X / torch.sqrt(cov_eigvals).unsqueeze(0)
    alpha_row_whitened = op_norm_symmetric((Xw @ Xw.T) / float(d) - I_n)

    eig_mean = float(torch.mean(cov_eigvals).item())
    eig_max = float(torch.max(cov_eigvals).item())
    eig_min = float(torch.min(cov_eigvals).item())
    cond = eig_max / eig_min
    cov_iso_error = float(torch.max(torch.abs(cov_eigvals / eig_mean - 1.0)).item())
    logspread = float(torch.std(torch.log(cov_eigvals)).item())

    return {
        "alpha_row_raw": alpha_row_raw,
        "alpha_row_whitened": alpha_row_whitened,
        "cov_eigvals_min": eig_min,
        "cov_eigvals_max": eig_max,
        "cov_mean": eig_mean,
        "cov_condition_number": cond,
        "cov_iso_error": cov_iso_error,
        "cov_logspread": logspread,
    }


@torch.no_grad()
def compute_teacher_alignment(model: TwoLayerSoftplus, B_star: torch.Tensor, top_rank: int) -> Dict[str, float]:
    H_student = model.B.T @ model.B
    H_teacher = B_star.T @ B_star
    rank = min(top_rank, H_student.shape[0], H_teacher.shape[0])
    return {"teacher_overlap_topr": subspace_overlap(H_student, H_teacher, rank=rank)}


@torch.no_grad()
def compute_metrics(
    model: TwoLayerSoftplus,
    X: torch.Tensor,
    y: torch.Tensor,
    lambda_B: float,
    dataset_diag: Dict[str, float],
    top_rank: int,
    B_star: torch.Tensor | None = None,
    rho_tol: float = 1e-12,
) -> Dict[str, float]:
    n, d = X.shape
    preds = model(X)
    r = preds - y

    B = model.B.detach()
    Q = compute_q(model, X)
    M = (Q.T @ Q) / float(n)
    H = B.T @ B
    G = B.T @ M @ B
    BBt = B @ B.T
    H2 = H @ H

    q_sq = torch.sum(Q * Q, dim=1)
    q_sq_sum = float(torch.sum(q_sq).item())
    r_sq = r * r

    rho = float(r.mean().item())
    resid_rms = float(torch.sqrt(torch.mean(r_sq)).item())
    resid_mean_abs = float(torch.mean(torch.abs(r)).item())
    eps_resid = float(torch.max(torch.abs(r - rho)).item())

    tau_r2 = float(torch.mean(r_sq).item())
    tau_q2 = float(torch.sum(r_sq * q_sq).item() / max(q_sq_sum, 1e-16))

    if abs(rho) > rho_tol:
        kappa_rho = d * (rho ** 2) / (lambda_B ** 2 * n)
        c_rho = 1.0 / kappa_rho
        delta_rho_op = op_norm_symmetric(BBt - kappa_rho * M)
        gamma_rho_op = op_norm_symmetric(G - c_rho * H2)
    else:
        kappa_rho = float("nan")
        c_rho = float("nan")
        delta_rho_op = float("nan")
        gamma_rho_op = float("nan")

    kappa_r2 = d * tau_r2 / (lambda_B ** 2 * n)
    c_r2 = 1.0 / kappa_r2
    delta_r2_op = op_norm_symmetric(BBt - kappa_r2 * M)
    gamma_r2_op = op_norm_symmetric(G - c_r2 * H2)

    kappa_q2 = d * tau_q2 / (lambda_B ** 2 * n)
    c_q2 = 1.0 / kappa_q2
    delta_q2_op = op_norm_symmetric(BBt - kappa_q2 * M)
    gamma_q2_op = op_norm_symmetric(G - c_q2 * H2)

    kappa_fit = best_scalar_fit(BBt, M)
    c_fit = best_scalar_fit(G, H2)
    delta_fit_op = op_norm_symmetric(BBt - kappa_fit * M)
    gamma_fit_op = op_norm_symmetric(G - c_fit * H2)
    delta_fit_frob = frob_norm(BBt - kappa_fit * M)
    gamma_fit_frob = frob_norm(G - c_fit * H2)

    evals_G, evecs_G = torch.linalg.eigh(symmetrize(G))
    evals_G = torch.clamp(evals_G, min=0.0)
    G_sqrt = evecs_G @ torch.diag(torch.sqrt(evals_G)) @ evecs_G.T
    s_fit = best_scalar_fit(H, G_sqrt)
    H_vs_Gsqrt_op = op_norm_symmetric(H - s_fit * G_sqrt)
    H_vs_Gsqrt_frob = frob_norm(H - s_fit * G_sqrt)

    rank = min(top_rank, d)
    overlap_H_G = subspace_overlap(H, G, rank=rank)
    overlap_H_Gsqrt = subspace_overlap(H, G_sqrt, rank=rank)

    loss_mse = 0.5 * torch.mean((preds - y) ** 2)
    reg = 0.5 * lambda_B * torch.sum(B * B)
    loss_total = float((loss_mse + reg).item())

    out = {
        "loss_total": loss_total,
        "loss_mse": float(loss_mse.item()),
        "reg_term": float(reg.item()),
        "rho": rho,
        "resid_rms": resid_rms,
        "resid_mean_abs": resid_mean_abs,
        "eps_resid": eps_resid,
        "tau_r2": tau_r2,
        "tau_q2": tau_q2,
        "kappa_rho": kappa_rho,
        "c_rho": c_rho,
        "delta_rho_op": delta_rho_op,
        "gamma_rho_op": gamma_rho_op,
        "kappa_r2": kappa_r2,
        "c_r2": c_r2,
        "delta_r2_op": delta_r2_op,
        "gamma_r2_op": gamma_r2_op,
        "kappa_q2": kappa_q2,
        "c_q2": c_q2,
        "delta_q2_op": delta_q2_op,
        "gamma_q2_op": gamma_q2_op,
        "kappa_fit": float(kappa_fit),
        "c_fit": float(c_fit),
        "delta_fit_op": delta_fit_op,
        "gamma_fit_op": gamma_fit_op,
        "delta_fit_frob": delta_fit_frob,
        "gamma_fit_frob": gamma_fit_frob,
        "H_vs_Gsqrt_op": H_vs_Gsqrt_op,
        "H_vs_Gsqrt_frob": H_vs_Gsqrt_frob,
        "overlap_H_G_topr": overlap_H_G,
        "overlap_H_Gsqrt_topr": overlap_H_Gsqrt,
        "B_op": op_norm_symmetric(H) ** 0.5,
        "Q_op": float(torch.linalg.matrix_norm(Q, ord=2).item()),
    }

    out.update(dataset_diag)

    if B_star is not None:
        out.update(compute_teacher_alignment(model, B_star, rank))

    return out


def grad_norm_sq(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).item())
    return total


def train_one(cfg: ExperimentConfig) -> Dict[str, object]:
    dtype = torch_dtype_from_name(cfg.dtype)
    device = torch.device(cfg.device)

    set_seed(cfg.seed)

    out_dir = Path(cfg.output_root) / cfg.name
    ensure_dir(out_dir)

    data = generate_teacher_student_dataset(cfg, dtype=dtype, device=device)
    X = data["X"]
    y = data["y"]
    B_star = data["B_star"]

    dataset_diag = compute_dataset_diagnostics(X, data["cov_eigvals"])
    model = TwoLayerSoftplus(cfg, dtype=dtype, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    history: List[Dict[str, float]] = []

    for step in range(cfg.steps + 1):
        optimizer.zero_grad()

        preds = model(X)
        mse = 0.5 * torch.mean((preds - y) ** 2)
        reg = 0.5 * cfg.lambda_B * torch.sum(model.B * model.B)
        loss = mse + reg
        loss.backward()

        gnorm = math.sqrt(grad_norm_sq(model))

        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            metrics = compute_metrics(
                model=model,
                X=X,
                y=y,
                lambda_B=cfg.lambda_B,
                dataset_diag=dataset_diag,
                top_rank=cfg.top_rank,
                B_star=B_star,
                rho_tol=cfg.rho_tol,
            )
            metrics["step"] = step
            metrics["grad_norm"] = gnorm
            history.append(metrics)

            print(
                f"[{cfg.name}] step={step:4d} "
                f"loss={metrics['loss_total']:.4e} "
                f"rms={metrics['resid_rms']:.4e} "
                f"delta_q2={metrics['delta_q2_op']:.4e} "
                f"gamma_q2={metrics['gamma_q2_op']:.4e} "
                f"delta_fit={metrics['delta_fit_op']:.4e} "
                f"gamma_fit={metrics['gamma_fit_op']:.4e} "
                f"grad={gnorm:.4e}"
            )

        if gnorm < cfg.grad_tol:
            print(f"[{cfg.name}] Early stop at step {step}: grad norm {gnorm:.3e} < tol.")
            break

        optimizer.step()

    with torch.no_grad():
        Q = compute_q(model, X)
        M = (Q.T @ Q) / float(cfg.n)
        H = model.B.T @ model.B
        G = model.B.T @ M @ model.B

    save_history(out_dir, cfg, history, dataset_diag)
    plot_training_curves(out_dir, cfg, history)
    plot_eigen_comparison(out_dir, H, G)
    save_final_matrices(out_dir, H, G)

    return {
        "config": asdict(cfg),
        "history": history,
        "output_dir": str(out_dir),
    }


def save_history(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]], dataset_diag: Dict[str, float]) -> None:
    payload = {
        "config": asdict(cfg),
        "dataset_diagnostics": dataset_diag,
        "history": history,
    }
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_training_curves(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    steps = [h["step"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(steps, [h["loss_total"] for h in history], label="loss_total")
    ax.plot(steps, [h["loss_mse"] for h in history], label="loss_mse")
    ax.set_title("Loss")
    ax.set_yscale("log")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(steps, [max(h["grad_norm"], 1e-16) for h in history], linestyle="--", label="grad_norm")
    ax2.set_yscale("log")
    ax2.set_ylabel("grad norm")

    ax = axes[0, 1]
    ax.plot(steps, [max(h["eps_resid"], 1e-16) for h in history], label="eps_resid")
    ax.plot(steps, [max(h["resid_rms"], 1e-16) for h in history], label="resid_rms")
    ax.set_title("Residual diagnostics")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, [max(h["delta_fit_op"], 1e-16) for h in history], label="delta_fit_op")
    ax.plot(steps, [max(h["delta_q2_op"], 1e-16) for h in history], label="delta_q2_op")
    if any(not math.isnan(h["delta_rho_op"]) for h in history):
        ax.plot(
            [h["step"] for h in history if not math.isnan(h["delta_rho_op"])],
            [max(h["delta_rho_op"], 1e-16) for h in history if not math.isnan(h["delta_rho_op"])],
            label="delta_rho_op",
            alpha=0.6,
        )
    ax.set_title(r"$\Delta_t = \|BB^T - \kappa M\|_{op}$")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, [max(h["gamma_fit_op"], 1e-16) for h in history], label="gamma_fit_op")
    ax.plot(steps, [max(h["gamma_q2_op"], 1e-16) for h in history], label="gamma_q2_op")
    if any(not math.isnan(h["gamma_rho_op"]) for h in history):
        ax.plot(
            [h["step"] for h in history if not math.isnan(h["gamma_rho_op"])],
            [max(h["gamma_rho_op"], 1e-16) for h in history if not math.isnan(h["gamma_rho_op"])],
            label="gamma_rho_op",
            alpha=0.6,
        )
    ax.set_title(r"$\Gamma_t = \|G - c H^2\|_{op}$")
    ax.set_yscale("log")
    ax.legend()

    for ax in axes.ravel():
        ax.set_xlabel("step")

    fig.suptitle(cfg.name)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_eigen_comparison(out_dir: Path, H: torch.Tensor, G: torch.Tensor) -> None:
    H = symmetrize(H)
    G = symmetrize(G)

    evals_H = torch.flip(torch.linalg.eigvalsh(H), dims=[0]).cpu()
    evals_G = torch.flip(torch.linalg.eigvalsh(G), dims=[0]).cpu()
    evals_G = torch.clamp(evals_G, min=0.0)
    evals_Gsqrt = torch.sqrt(evals_G)

    denom = float(torch.sum(evals_Gsqrt ** 2).item())
    s_fit = float(torch.sum(evals_H * evals_Gsqrt).item() / denom) if denom > 1e-12 else float("nan")

    ranks = list(range(1, len(evals_H) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(ranks, evals_H.numpy(), marker="o", markersize=3, label="eig(H = B^T B)")
    if not math.isnan(s_fit):
        plt.plot(ranks, (s_fit * evals_Gsqrt).numpy(), marker="x", markersize=3, label=r"best scale · eig($G^{1/2}$)")
    plt.yscale("log")
    plt.xlabel("eigenvalue rank")
    plt.ylabel("magnitude")
    plt.title("Late-training spectral comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eigen_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_final_matrices(out_dir: Path, H: torch.Tensor, G: torch.Tensor) -> None:
    payload = {
        "H_shape": list(H.shape),
        "G_shape": list(G.shape),
        "H_trace": float(torch.trace(H).item()),
        "G_trace": float(torch.trace(G).item()),
    }
    with open(out_dir / "final_matrix_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def default_configs() -> List[ExperimentConfig]:
    baseline = ExperimentConfig(
        name="baseline_isotropic",
        seed=0,
        n=128,
        d=256,
        m_teacher=4,
        m_student=16,
        lambda_B=1e-2,
        lr=5e-2,
        steps=3000,
        checkpoint_every=20,
        anisotropic=False,
        noise_std=0.0,
        top_rank=4,
    )

    anisotropic = ExperimentConfig(
        name="ablation_anisotropic",
        seed=0,
        n=128,
        d=256,
        m_teacher=4,
        m_student=16,
        lambda_B=1e-2,
        lr=5e-2,
        steps=3000,
        checkpoint_every=20,
        anisotropic=True,
        spectrum_min=1e-2,
        noise_std=0.0,
        top_rank=4,
    )

    return [baseline, anisotropic]


def main() -> None:
    configs = default_configs()

    all_results = []
    for cfg in configs:
        results = train_one(cfg)
        final_metrics = results["history"][-1]
        all_results.append({
            "name": cfg.name,
            "output_dir": results["output_dir"],
            "final_metrics": final_metrics,
        })

    summary_path = Path(configs[0].output_root) / "summary.json"
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved summary to:", summary_path)
    for row in all_results:
        fm = row["final_metrics"]
        print(
            f"- {row['name']}: {row['output_dir']} "
            f"| delta_q2={fm['delta_q2_op']:.3e}, gamma_q2={fm['gamma_q2_op']:.3e}, "
            f"delta_fit={fm['delta_fit_op']:.3e}, gamma_fit={fm['gamma_fit_op']:.3e}"
        )


if __name__ == "__main__":
    main()
