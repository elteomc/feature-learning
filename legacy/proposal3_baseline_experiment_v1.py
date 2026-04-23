
#!/usr/bin/env python3
"""
Baseline synthetic experiment for Proposal 3:
stationarity -> hidden covariance -> AGOP square-law in a two-layer network.

What this script does
---------------------
1. Generates teacher-student synthetic data:
       x_i ~ N(0, I_d)           (or anisotropic N(0, Sigma))
       y_i = a_*^T softplus(B_* x_i) + noise
2. Trains a student model:
       f(x) = a^T softplus(B x)
   with full-batch gradient descent on square loss + L2 regularization on B.
3. Logs the core matrices and diagnostics:
       H_t      = B_t^T B_t
       M_t      = (1/n) Q_t^T Q_t
       G_t      = B_t^T M_t B_t
       eps_t    = ||R_t - rho_t I||_op
       Delta_t  = ||B_t B_t^T - kappa M_t||_op
       Gamma_t  = ||G_t - c H_t^2||_op
   with both:
       - theory-scaled kappa = d rho_t^2 / (lambda^2 n), when rho_t is usable
       - best-fit kappa_fit minimizing Frobenius error
4. Saves:
       - JSON logs
       - training-curve PNG
       - eigenvalue-comparison PNG

Run:
    python proposal3_baseline_experiment.py

You can edit the CONFIGS list near the bottom to add sweeps.
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


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def op_norm_symmetric(mat: torch.Tensor) -> float:
    """
    Operator norm for a symmetric matrix.
    """
    mat = 0.5 * (mat + mat.T)
    evals = torch.linalg.eigvalsh(mat)
    return float(torch.max(torch.abs(evals)).item())


def frob_norm(mat: torch.Tensor) -> float:
    return float(torch.linalg.norm(mat, ord='fro').item())


def best_scalar_fit(target: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Return scalar s minimizing ||target - s * basis||_F^2.
    """
    denom = torch.sum(basis * basis).item()
    if abs(denom) < eps:
        return float("nan")
    numer = torch.sum(target * basis).item()
    return numer / denom


def subspace_overlap(A: torch.Tensor, B: torch.Tensor, rank: int) -> float:
    """
    Squared Frobenius overlap between top-r eigenspaces of two symmetric PSD matrices.
    Returns a number in [0, rank].
    """
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    evals_A, evecs_A = torch.linalg.eigh(A)
    evals_B, evecs_B = torch.linalg.eigh(B)

    UA = evecs_A[:, -rank:]
    UB = evecs_B[:, -rank:]
    overlap = torch.linalg.norm(UA.T @ UB, ord='fro').item() ** 2
    return float(overlap)


# ----------------------------
# Config
# ----------------------------

@dataclass
class ExperimentConfig:
    name: str
    seed: int = 0

    # Dataset / model sizes
    n: int = 128
    d: int = 256
    m_teacher: int = 4
    m_student: int = 16

    # Teacher / student scales
    teacher_B_scale: float = 1.0
    teacher_a_scale: float = 1.0
    student_B_scale: float = 0.5
    student_a_scale: float = 0.5

    # Training
    steps: int = 3000
    lr: float = 5e-2
    lambda_B: float = 1e-2
    checkpoint_every: int = 20
    grad_tol: float = 1e-8

    # Data generation
    noise_std: float = 0.0
    anisotropic: bool = False
    spectrum_min: float = 1e-2  # only used when anisotropic=True

    # Numerics
    dtype: str = "float64"
    device: str = "cpu"

    # Diagnostics
    top_rank: int = 4
    rho_tol: float = 1e-10

    # Output
    output_root: str = "proposal3_outputs"


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}. Use one of {list(mapping)}.")
    return mapping[name]


# ----------------------------
# Teacher-student data
# ----------------------------

def sample_inputs(cfg: ExperimentConfig, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        X: [n, d] inputs
        eigvals: [d] covariance eigenvalues used
    """
    Z = torch.randn(cfg.n, cfg.d, dtype=dtype, device=device)
    if not cfg.anisotropic:
        eigvals = torch.ones(cfg.d, dtype=dtype, device=device)
        return Z, eigvals

    # Diagonal covariance with exponentially decaying spectrum from 1 down to spectrum_min.
    log_min = math.log(cfg.spectrum_min)
    exponents = torch.linspace(0.0, 1.0, cfg.d, dtype=dtype, device=device)
    eigvals = torch.exp(log_min * exponents)  # from 1 down to spectrum_min
    X = Z * torch.sqrt(eigvals).unsqueeze(0)
    return X, eigvals


def softplus_hidden(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    X: [n, d], B: [m, d]
    Returns hidden activations [n, m].
    """
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
    noise = cfg.noise_std * torch.randn_like(y_clean)
    y = y_clean + noise
    return {
        "X": X,
        "y": y,
        "y_clean": y_clean,
        "B_star": B_star,
        "a_star": a_star,
        "cov_eigvals": eigvals,
    }


# ----------------------------
# Student model
# ----------------------------

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
        H = F.softplus(X @ self.B.T)  # [n, m]
        return H @ self.a            # [n]


# ----------------------------
# Diagnostics
# ----------------------------

@torch.no_grad()
def compute_q(model: TwoLayerSoftplus, X: torch.Tensor) -> torch.Tensor:
    """
    q_i = a \odot softplus'(B x_i) = a \odot sigmoid(B x_i)
    Returns Q of shape [n, m].
    """
    Z = X @ model.B.T
    return torch.sigmoid(Z) * model.a.unsqueeze(0)


@torch.no_grad()
def compute_alpha(X: torch.Tensor) -> float:
    """
    alpha = || (1/d) X X^T - I_n ||_op
    """
    n, d = X.shape
    gram = (X @ X.T) / float(d)
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    return op_norm_symmetric(gram - I)


@torch.no_grad()
def compute_teacher_alignment(model: TwoLayerSoftplus, B_star: torch.Tensor, top_rank: int) -> Dict[str, float]:
    """
    Optional extra diagnostic: overlap with teacher feature geometry.
    """
    H_student = model.B.T @ model.B
    H_teacher = B_star.T @ B_star
    rank = min(top_rank, H_student.shape[0], H_teacher.shape[0])
    return {
        "teacher_overlap_topr": subspace_overlap(H_student, H_teacher, rank=rank)
    }


@torch.no_grad()
def compute_metrics(
    model: TwoLayerSoftplus,
    X: torch.Tensor,
    y: torch.Tensor,
    lambda_B: float,
    alpha_dataset: float,
    top_rank: int,
    B_star: torch.Tensor | None = None,
    rho_tol: float = 1e-10,
) -> Dict[str, float]:
    """
    Core objects:
        r      : residual vector [n]
        R      : diag(r)
        Q      : derivative feature matrix [n, m]
        M      : (1/n) Q^T Q
        H      : B^T B
        G      : B^T M B
    """
    n, d = X.shape
    preds = model(X)
    r = preds - y
    rho = float(r.mean().item())

    B = model.B.detach()
    Q = compute_q(model, X)
    M = (Q.T @ Q) / float(n)
    H = B.T @ B
    G = B.T @ M @ B
    BBt = B @ B.T
    H2 = H @ H

    # Residual spread: R is diagonal, so ||R - rho I||_op = max_i |r_i - rho|
    eps_resid = float(torch.max(torch.abs(r - rho)).item())

    # Theory scalar. Can blow up / become unstable when rho is too close to zero.
    if abs(rho) > rho_tol:
        kappa_theory = d * (rho ** 2) / (lambda_B ** 2 * n)
        c_theory = 1.0 / kappa_theory
        delta_theory = op_norm_symmetric(BBt - kappa_theory * M)
        gamma_theory = op_norm_symmetric(G - c_theory * H2)
    else:
        kappa_theory = float("nan")
        c_theory = float("nan")
        delta_theory = float("nan")
        gamma_theory = float("nan")

    # Best-fit scalars: robust way to test "same shape up to scale"
    kappa_fit = best_scalar_fit(BBt, M)
    c_fit = best_scalar_fit(G, H2)
    delta_fit_op = op_norm_symmetric(BBt - kappa_fit * M)
    gamma_fit_op = op_norm_symmetric(G - c_fit * H2)
    delta_fit_frob = frob_norm(BBt - kappa_fit * M)
    gamma_fit_frob = frob_norm(G - c_fit * H2)

    # Compare H with sqrt(G), again up to best scalar.
    evals_G, evecs_G = torch.linalg.eigh(0.5 * (G + G.T))
    evals_G_clipped = torch.clamp(evals_G, min=0.0)
    G_sqrt = evecs_G @ torch.diag(torch.sqrt(evals_G_clipped)) @ evecs_G.T
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
        "eps_resid": eps_resid,
        "alpha_dataset": alpha_dataset,
        "kappa_theory": kappa_theory,
        "c_theory": c_theory,
        "delta_theory_op": delta_theory,
        "gamma_theory_op": gamma_theory,
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
        "B_op": op_norm_symmetric(B.T @ B) ** 0.5,
        "Q_op": float(torch.linalg.matrix_norm(Q, ord=2).item()),
    }

    if B_star is not None:
        out.update(compute_teacher_alignment(model, B_star, top_rank=rank))

    return out


def grad_norm_sq(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).item())
    return total


# ----------------------------
# Training
# ----------------------------

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

    model = TwoLayerSoftplus(cfg, dtype=dtype, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    alpha_dataset = compute_alpha(X)

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
                alpha_dataset=alpha_dataset,
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
                f"eps={metrics['eps_resid']:.4e} "
                f"delta_fit={metrics['delta_fit_op']:.4e} "
                f"gamma_fit={metrics['gamma_fit_op']:.4e} "
                f"grad={gnorm:.4e}"
            )

        if gnorm < cfg.grad_tol:
            print(f"[{cfg.name}] Early stop at step {step}: grad norm {gnorm:.3e} < tol.")
            break

        optimizer.step()

    # Final matrices for eig plots
    with torch.no_grad():
        Q = compute_q(model, X)
        M = (Q.T @ Q) / float(cfg.n)
        H = model.B.T @ model.B
        G = model.B.T @ M @ model.B

    save_history(out_dir, cfg, history, data_summary={
        "cov_eigvals_min": float(torch.min(data["cov_eigvals"]).item()),
        "cov_eigvals_max": float(torch.max(data["cov_eigvals"]).item()),
    })
    plot_training_curves(out_dir, cfg, history)
    plot_eigen_comparison(out_dir, H, G)
    save_final_matrices(out_dir, H, G)

    return {
        "config": asdict(cfg),
        "history": history,
        "output_dir": str(out_dir),
    }


# ----------------------------
# Output helpers
# ----------------------------

def save_history(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]], data_summary: Dict[str, float]) -> None:
    payload = {
        "config": asdict(cfg),
        "data_summary": data_summary,
        "history": history,
    }
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_training_curves(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    steps = [h["step"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss / grad
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

    # Residual spread
    ax = axes[0, 1]
    ax.plot(steps, [max(h["eps_resid"], 1e-16) for h in history], label="eps_resid")
    ax.axhline(history[0]["alpha_dataset"], linestyle="--", label="alpha_dataset")
    ax.set_title("Residual spread / dataset isotropy")
    ax.set_yscale("log")
    ax.legend()

    # Hidden covariance relation
    ax = axes[1, 0]
    ax.plot(steps, [max(h["delta_fit_op"], 1e-16) for h in history], label="delta_fit_op")
    theory_vals = [h["delta_theory_op"] for h in history if not math.isnan(h["delta_theory_op"])]
    if theory_vals:
        ax.plot(
            [h["step"] for h in history if not math.isnan(h["delta_theory_op"])],
            [max(h["delta_theory_op"], 1e-16) for h in history if not math.isnan(h["delta_theory_op"])],
            label="delta_theory_op",
        )
    ax.set_title(r"$\Delta_t = \|BB^T - \kappa M\|_{op}$")
    ax.set_yscale("log")
    ax.legend()

    # AGOP square law
    ax = axes[1, 1]
    ax.plot(steps, [max(h["gamma_fit_op"], 1e-16) for h in history], label="gamma_fit_op")
    theory_vals = [h["gamma_theory_op"] for h in history if not math.isnan(h["gamma_theory_op"])]
    if theory_vals:
        ax.plot(
            [h["step"] for h in history if not math.isnan(h["gamma_theory_op"])],
            [max(h["gamma_theory_op"], 1e-16) for h in history if not math.isnan(h["gamma_theory_op"])],
            label="gamma_theory_op",
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
    H = 0.5 * (H + H.T)
    G = 0.5 * (G + G.T)

    evals_H = torch.flip(torch.linalg.eigvalsh(H), dims=[0]).cpu()
    evals_G = torch.flip(torch.linalg.eigvalsh(G), dims=[0]).cpu()
    evals_G = torch.clamp(evals_G, min=0.0)
    evals_Gsqrt = torch.sqrt(evals_G)

    # Best scalar fit of H against sqrt(G), in eigenvalue space for a quick visual.
    denom = float(torch.sum(evals_Gsqrt ** 2).item())
    s_fit = float(torch.sum(evals_H * evals_Gsqrt).item() / denom) if denom > 1e-12 else float("nan")

    ranks = list(range(1, len(evals_H) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(ranks, evals_H.numpy(), marker="o", markersize=3, label="eig(H = B^T B)")
    if not math.isnan(s_fit):
        plt.plot(ranks, (s_fit * evals_Gsqrt).numpy(), marker="x", markersize=3, label=r"best scale $\cdot$ eig($G^{1/2}$)")
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


# ----------------------------
# Suggested configs
# ----------------------------

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
        all_results.append({
            "name": cfg.name,
            "output_dir": results["output_dir"],
            "final_metrics": results["history"][-1],
        })

    summary_path = Path(configs[0].output_root) / "summary.json"
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved summary to:", summary_path)
    for row in all_results:
        print(f"- {row['name']}: {row['output_dir']}")


if __name__ == "__main__":
    main()
