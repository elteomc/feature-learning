
#!/usr/bin/env python3
"""
Proposal 3 synthetic experiment (v5.1, robust)

Changes vs v5
-------------
1. Safer symmetric operator norm / eigendecompositions:
   - fallback from eigvalsh/eigh to svdvals if needed
   - returns NaN instead of crashing on non-finite matrices

2. Safer sqrt(G):
   - robust PSD square root with fallback through SVD

3. LBFGS is run in outer iterations with max_iter=1 each time:
   - easier to monitor
   - easier to catch instability
   - records intermediate checkpoints

4. Optional gradient clipping during SGD

5. Best checkpoint is still selected by smallest delta_stationary_op,
   tie-broken by grad_norm.
"""

from __future__ import annotations

import copy
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


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def safe_op_norm_symmetric(mat: torch.Tensor) -> float:
    """
    Robust operator norm for a symmetric matrix.
    Tries eigvalsh first, then falls back to svdvals.
    Returns NaN if the matrix is not finite.
    """
    mat = symmetrize(mat)
    if not is_finite_tensor(mat):
        return float("nan")
    try:
        evals = torch.linalg.eigvalsh(mat)
        return float(torch.max(torch.abs(evals)).item())
    except Exception:
        try:
            svals = torch.linalg.svdvals(mat)
            return float(torch.max(svals).item())
        except Exception:
            return float("nan")


def safe_frob_norm(mat: torch.Tensor) -> float:
    if not is_finite_tensor(mat):
        return float("nan")
    try:
        return float(torch.linalg.norm(mat, ord="fro").item())
    except Exception:
        return float("nan")


def best_scalar_fit(target: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12) -> float:
    if (not is_finite_tensor(target)) or (not is_finite_tensor(basis)):
        return float("nan")
    denom = torch.sum(basis * basis).item()
    if abs(denom) < eps:
        return float("nan")
    numer = torch.sum(target * basis).item()
    return numer / denom


def safe_subspace_overlap(A: torch.Tensor, B: torch.Tensor, rank: int) -> float:
    A = symmetrize(A)
    B = symmetrize(B)
    if (not is_finite_tensor(A)) or (not is_finite_tensor(B)):
        return float("nan")
    try:
        _, evecs_A = torch.linalg.eigh(A)
        _, evecs_B = torch.linalg.eigh(B)
    except Exception:
        try:
            _, _, vA = torch.linalg.svd(A)
            _, _, vB = torch.linalg.svd(B)
            evecs_A = vA.T
            evecs_B = vB.T
        except Exception:
            return float("nan")

    UA = evecs_A[:, -rank:]
    UB = evecs_B[:, -rank:]
    try:
        return float(torch.linalg.norm(UA.T @ UB, ord="fro").item() ** 2)
    except Exception:
        return float("nan")


def safe_psd_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """
    Robust PSD square root of a symmetric matrix.
    If eigh fails, falls back to SVD.
    Negative eigenvalues are clipped to zero.
    """
    mat = symmetrize(mat)
    if not is_finite_tensor(mat):
        return torch.full_like(mat, float("nan"))
    try:
        evals, evecs = torch.linalg.eigh(mat)
        evals = torch.clamp(evals, min=0.0)
        return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T
    except Exception:
        try:
            U, S, Vh = torch.linalg.svd(mat)
            S = torch.clamp(S, min=0.0)
            return U @ torch.diag(torch.sqrt(S)) @ Vh
        except Exception:
            return torch.full_like(mat, float("nan"))


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float64": torch.float64}
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

    # Stage 1: SGD
    sgd_steps: int = 3000
    sgd_lr: float = 5e-2
    checkpoint_every: int = 20
    clip_grad_norm: float = 10.0

    # Stage 2: LBFGS outer loop
    use_lbfgs: bool = True
    lbfgs_outer_steps: int = 100
    lbfgs_history_size: int = 50
    lbfgs_lr: float = 0.5
    lbfgs_tolerance_grad: float = 1e-10
    lbfgs_tolerance_change: float = 1e-12
    lbfgs_log_every: int = 5

    # Regularization
    lambda_B: float = 1e-2

    # Stopping/selection
    grad_tol: float = 1e-8

    # Data
    noise_std: float = 0.0
    anisotropic: bool = False
    spectrum_min: float = 1e-2

    # Numerics
    dtype: str = "float64"
    device: str = "cpu"

    # Diagnostics
    top_rank: int = 4

    # Output
    output_root: str = "proposal3_outputs_v5_1"


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
    B_star = cfg.teacher_B_scale * torch.randn(cfg.m_teacher, cfg.d, dtype=dtype, device=device) / math.sqrt(cfg.d)
    a_star = cfg.teacher_a_scale * torch.randn(cfg.m_teacher, dtype=dtype, device=device) / math.sqrt(cfg.m_teacher)
    return B_star, a_star


def generate_teacher_student_dataset(cfg: ExperimentConfig, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor]:
    X, eigvals = sample_inputs(cfg, dtype=dtype, device=device)
    B_star, a_star = sample_teacher(cfg, dtype=dtype, device=device)
    y_clean = softplus_hidden(X, B_star) @ a_star
    y = y_clean + cfg.noise_std * torch.randn_like(y_clean)
    return {"X": X, "y": y, "y_clean": y_clean, "B_star": B_star, "a_star": a_star, "cov_eigvals": eigvals}


class TwoLayerSoftplus(nn.Module):
    def __init__(self, cfg: ExperimentConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        B0 = cfg.student_B_scale * torch.randn(cfg.m_student, cfg.d, dtype=dtype, device=device) / math.sqrt(cfg.d)
        a0 = cfg.student_a_scale * torch.randn(cfg.m_student, dtype=dtype, device=device) / math.sqrt(cfg.m_student)
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

    alpha_row_raw = safe_op_norm_symmetric((X @ X.T) / float(d) - I_n)
    Xw = X / torch.sqrt(cov_eigvals).unsqueeze(0)
    alpha_row_whitened = safe_op_norm_symmetric((Xw @ Xw.T) / float(d) - I_n)

    eig_mean = float(torch.mean(cov_eigvals).item())
    eig_max = float(torch.max(cov_eigvals).item())
    eig_min = float(torch.min(cov_eigvals).item())
    cond = eig_max / eig_min

    return {
        "alpha_row_raw": alpha_row_raw,
        "alpha_row_whitened": alpha_row_whitened,
        "cov_eigvals_min": eig_min,
        "cov_eigvals_max": eig_max,
        "cov_mean": eig_mean,
        "cov_condition_number": cond,
        "cov_iso_error": float(torch.max(torch.abs(cov_eigvals / eig_mean - 1.0)).item()),
        "cov_logspread": float(torch.std(torch.log(cov_eigvals)).item()),
    }


@torch.no_grad()
def compute_teacher_alignment(model: TwoLayerSoftplus, B_star: torch.Tensor, top_rank: int) -> Dict[str, float]:
    H_student = model.B.T @ model.B
    H_teacher = B_star.T @ B_star
    rank = min(top_rank, H_student.shape[0], H_teacher.shape[0])
    return {"teacher_overlap_topr": safe_subspace_overlap(H_student, H_teacher, rank=rank)}


@torch.no_grad()
def orthonormal_basis_for_columns(A: torch.Tensor, tol: float = 1e-12) -> Tuple[torch.Tensor, int]:
    if A.numel() == 0 or not is_finite_tensor(A):
        return A.new_zeros((A.shape[0], 0)), 0
    try:
        U, S, _ = torch.linalg.svd(A, full_matrices=False)
        if S.numel() == 0:
            return A.new_zeros((A.shape[0], 0)), 0
        thresh = max(float(S[0].item()) * max(A.shape) * torch.finfo(A.dtype).eps, tol)
        rank = int(torch.sum(S > thresh).item())
        return U[:, :rank], rank
    except Exception:
        return A.new_zeros((A.shape[0], 0)), 0


@torch.no_grad()
def compute_metrics(
    model: TwoLayerSoftplus,
    X: torch.Tensor,
    y: torch.Tensor,
    lambda_B: float,
    dataset_diag: Dict[str, float],
    top_rank: int,
    B_star: torch.Tensor | None = None,
) -> Dict[str, float]:
    n, d = X.shape
    preds = model(X)
    r = preds - y

    B = model.B.detach()
    Q = compute_q(model, X)

    M = (Q.T @ Q) / float(n)

    RX = r.unsqueeze(1) * X
    RQ = r.unsqueeze(1) * Q

    T = Q.T @ (r.square().unsqueeze(1) * Q)
    M_tilde = T / float(n)

    A = Q.T @ RX
    S = A @ A.T

    H = B.T @ B
    BBt = B @ B.T
    G = B.T @ M @ B
    H2 = H @ H
    G_tilde = B.T @ M_tilde @ B

    r_sq = r.square()
    rho = float(r.mean().item())
    resid_rms = float(torch.sqrt(torch.mean(r_sq)).item())
    resid_mean_abs = float(torch.mean(torch.abs(r)).item())
    eps_resid = float(torch.max(torch.abs(r - rho)).item())

    # Stationarity
    B_stationary = -(Q.T @ RX) / (lambda_B * n)
    stationarity_B_frob = safe_frob_norm(B - B_stationary)
    if is_finite_tensor(B - B_stationary):
        try:
            stationarity_B_op = float(torch.linalg.matrix_norm(B - B_stationary, ord=2).item())
        except Exception:
            stationarity_B_op = float("nan")
    else:
        stationarity_B_op = float("nan")
    BBt_stationary = B_stationary @ B_stationary.T
    delta_stationary_op = safe_op_norm_symmetric(BBt - BBt_stationary)
    delta_stationary_frob = safe_frob_norm(BBt - BBt_stationary)

    # Ambient-d closure
    kappa_tilde = d / (lambda_B ** 2 * n)
    delta_tilde_op = safe_op_norm_symmetric(BBt - kappa_tilde * M_tilde)
    delta_tilde_frob = safe_frob_norm(BBt - kappa_tilde * M_tilde)

    # Effective-d closure
    d_eff = best_scalar_fit(S, T)
    kappa_eff = d_eff / (lambda_B ** 2 * n) if math.isfinite(d_eff) else float("nan")
    delta_eff_op = safe_op_norm_symmetric(BBt - kappa_eff * M_tilde) if math.isfinite(kappa_eff) else float("nan")
    delta_eff_frob = safe_frob_norm(BBt - kappa_eff * M_tilde) if math.isfinite(kappa_eff) else float("nan")

    # Bridge
    beta_fit = best_scalar_fit(M_tilde, M)
    M_bridge_op = safe_op_norm_symmetric(M_tilde - beta_fit * M) if math.isfinite(beta_fit) else float("nan")
    M_bridge_frob = safe_frob_norm(M_tilde - beta_fit * M) if math.isfinite(beta_fit) else float("nan")

    c_eff = float("nan")
    gamma_eff_op = float("nan")
    gamma_eff_frob = float("nan")
    if math.isfinite(kappa_eff) and math.isfinite(beta_fit) and abs(kappa_eff * beta_fit) > 1e-16:
        c_eff = 1.0 / (kappa_eff * beta_fit)
        gamma_eff_op = safe_op_norm_symmetric(G - c_eff * H2)
        gamma_eff_frob = safe_frob_norm(G - c_eff * H2)

    gamma_tilde_eff_op = safe_op_norm_symmetric(H2 - kappa_eff * G_tilde) if math.isfinite(kappa_eff) else float("nan")
    gamma_tilde_eff_frob = safe_frob_norm(H2 - kappa_eff * G_tilde) if math.isfinite(kappa_eff) else float("nan")

    # Best unconstrained fit
    kappa_fit = best_scalar_fit(BBt, M)
    c_fit = best_scalar_fit(G, H2)
    delta_fit_op = safe_op_norm_symmetric(BBt - kappa_fit * M) if math.isfinite(kappa_fit) else float("nan")
    gamma_fit_op = safe_op_norm_symmetric(G - c_fit * H2) if math.isfinite(c_fit) else float("nan")
    delta_fit_frob = safe_frob_norm(BBt - kappa_fit * M) if math.isfinite(kappa_fit) else float("nan")
    gamma_fit_frob = safe_frob_norm(G - c_fit * H2) if math.isfinite(c_fit) else float("nan")

    # H vs sqrt(G)
    G_sqrt = safe_psd_sqrt(G)
    s_fit = best_scalar_fit(H, G_sqrt)
    H_vs_Gsqrt_op = safe_op_norm_symmetric(H - s_fit * G_sqrt) if math.isfinite(s_fit) else float("nan")
    H_vs_Gsqrt_frob = safe_frob_norm(H - s_fit * G_sqrt) if math.isfinite(s_fit) else float("nan")

    rank = min(top_rank, d)
    overlap_H_G = safe_subspace_overlap(H, G, rank=rank)
    overlap_H_Gsqrt = safe_subspace_overlap(H, G_sqrt, rank=rank)

    # Relevant-subspace isotropy
    U_rel, rank_rel = orthonormal_basis_for_columns(RQ)
    alpha_rel = float("nan")
    alpha_rel_d = float("nan")
    if rank_rel > 0 and math.isfinite(d_eff) and abs(d_eff) > 1e-16:
        I_rel = torch.eye(rank_rel, dtype=X.dtype, device=X.device)
        X_rel_eff = U_rel.T @ ((X @ X.T) / d_eff) @ U_rel - I_rel
        alpha_rel = safe_op_norm_symmetric(X_rel_eff)

        X_rel_d = U_rel.T @ ((X @ X.T) / float(d)) @ U_rel - I_rel
        alpha_rel_d = safe_op_norm_symmetric(X_rel_d)

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
        "stationarity_B_frob": stationarity_B_frob,
        "stationarity_B_op": stationarity_B_op,
        "delta_stationary_op": delta_stationary_op,
        "delta_stationary_frob": delta_stationary_frob,
        "kappa_tilde": float(kappa_tilde),
        "delta_tilde_op": delta_tilde_op,
        "delta_tilde_frob": delta_tilde_frob,
        "d_eff": float(d_eff),
        "kappa_eff": float(kappa_eff),
        "delta_eff_op": delta_eff_op,
        "delta_eff_frob": delta_eff_frob,
        "beta_fit": float(beta_fit),
        "M_bridge_op": M_bridge_op,
        "M_bridge_frob": M_bridge_frob,
        "c_eff": float(c_eff),
        "gamma_eff_op": float(gamma_eff_op),
        "gamma_eff_frob": float(gamma_eff_frob),
        "gamma_tilde_eff_op": gamma_tilde_eff_op,
        "gamma_tilde_eff_frob": gamma_tilde_eff_frob,
        "alpha_rel": float(alpha_rel),
        "alpha_rel_d": float(alpha_rel_d),
        "rank_rel": int(rank_rel),
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
        "B_op": safe_op_norm_symmetric(H) ** 0.5 if math.isfinite(safe_op_norm_symmetric(H)) else float("nan"),
    }
    if is_finite_tensor(Q):
        try:
            out["Q_op"] = float(torch.linalg.matrix_norm(Q, ord=2).item())
        except Exception:
            out["Q_op"] = float("nan")
    else:
        out["Q_op"] = float("nan")

    out.update(dataset_diag)
    if B_star is not None:
        out.update(compute_teacher_alignment(model, B_star, rank))
    return out


def grad_norm_sq(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            total += float(torch.sum(p.grad.detach() ** 2).item())
        elif p.grad is not None:
            return float("inf")
    return total


def objective(model: TwoLayerSoftplus, X: torch.Tensor, y: torch.Tensor, lambda_B: float) -> torch.Tensor:
    preds = model(X)
    mse = 0.5 * torch.mean((preds - y) ** 2)
    reg = 0.5 * lambda_B * torch.sum(model.B * model.B)
    return mse + reg


def is_better_metric(candidate: Dict[str, float], incumbent: Dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    cand_delta = candidate["delta_stationary_op"]
    inc_delta = incumbent["delta_stationary_op"]
    if not math.isfinite(cand_delta):
        return False
    if not math.isfinite(inc_delta):
        return True
    if cand_delta < inc_delta - 1e-15:
        return True
    if abs(cand_delta - inc_delta) <= 1e-15:
        cand_grad = candidate["grad_norm"]
        inc_grad = incumbent["grad_norm"]
        if math.isfinite(cand_grad) and (not math.isfinite(inc_grad) or cand_grad < inc_grad):
            return True
    return False


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

    history: List[Dict[str, float]] = []
    best_metrics = None
    best_state = None

    def record(step: int, phase: str) -> None:
        nonlocal best_metrics, best_state
        metrics = compute_metrics(
            model=model,
            X=X,
            y=y,
            lambda_B=cfg.lambda_B,
            dataset_diag=dataset_diag,
            top_rank=cfg.top_rank,
            B_star=B_star,
        )
        metrics["step"] = step
        metrics["phase"] = phase
        metrics["grad_norm"] = math.sqrt(grad_norm_sq(model))
        history.append(metrics)

        if is_better_metric(metrics, best_metrics):
            best_metrics = copy.deepcopy(metrics)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(
            f"[{cfg.name}][{phase}] step={step:4d} "
            f"loss={metrics['loss_total']:.4e} "
            f"grad={metrics['grad_norm']:.4e} "
            f"delta_stat={metrics['delta_stationary_op']:.4e} "
            f"delta_eff={metrics['delta_eff_op']:.4e} "
            f"gamma_eff={metrics['gamma_eff_op']:.4e}"
        )

    # Stage 1: SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.sgd_lr)

    for step in range(cfg.sgd_steps + 1):
        optimizer.zero_grad()
        loss = objective(model, X, y, cfg.lambda_B)
        loss.backward()

        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

        if step % cfg.checkpoint_every == 0 or step == cfg.sgd_steps:
            record(step, phase="sgd")

        gnorm = math.sqrt(grad_norm_sq(model))
        if gnorm < cfg.grad_tol:
            print(f"[{cfg.name}] Early stop in SGD at step {step}: grad norm {gnorm:.3e} < tol.")
            break

        optimizer.step()

    # Stage 2: LBFGS outer loop
    if cfg.use_lbfgs:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=cfg.lbfgs_lr,
            max_iter=1,
            history_size=cfg.lbfgs_history_size,
            tolerance_grad=cfg.lbfgs_tolerance_grad,
            tolerance_change=cfg.lbfgs_tolerance_change,
            line_search_fn="strong_wolfe",
        )

        for outer in range(1, cfg.lbfgs_outer_steps + 1):
            def closure():
                lbfgs.zero_grad()
                loss = objective(model, X, y, cfg.lambda_B)
                loss.backward()
                return loss

            try:
                lbfgs.step(closure)
            except Exception as e:
                print(f"[{cfg.name}] LBFGS stopped early at outer step {outer} due to: {e}")
                break

            if outer % cfg.lbfgs_log_every == 0 or outer == cfg.lbfgs_outer_steps:
                record(cfg.sgd_steps + outer, phase="lbfgs")

            gnorm = math.sqrt(grad_norm_sq(model))
            if gnorm < cfg.grad_tol:
                print(f"[{cfg.name}] Early stop in LBFGS at outer step {outer}: grad norm {gnorm:.3e} < tol.")
                break

    final_metrics = history[-1]

    if best_state is not None:
        best_model = TwoLayerSoftplus(cfg, dtype=dtype, device=device)
        best_model.load_state_dict(best_state)
    else:
        best_model = model

    with torch.no_grad():
        Q_final = compute_q(best_model, X)
        M_final = (Q_final.T @ Q_final) / float(cfg.n)
        H_final = best_model.B.T @ best_model.B
        G_final = best_model.B.T @ M_final @ best_model.B

    save_history(out_dir, cfg, history, dataset_diag, best_metrics)
    plot_training_curves(out_dir, cfg, history)
    plot_eigen_comparison(out_dir, H_final, G_final)
    save_final_matrices(out_dir, H_final, G_final)

    return {
        "config": asdict(cfg),
        "history": history,
        "final_metrics": final_metrics,
        "best_metrics_by_stationarity": best_metrics,
        "output_dir": str(out_dir),
    }


def save_history(
    out_dir: Path,
    cfg: ExperimentConfig,
    history: List[Dict[str, float]],
    dataset_diag: Dict[str, float],
    best_metrics: Dict[str, float] | None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "dataset_diagnostics": dataset_diag,
        "best_metrics_by_stationarity": best_metrics,
        "history": history,
    }
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_training_curves(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    steps = list(range(len(history)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(steps, [h["loss_total"] for h in history], label="loss_total")
    ax.plot(steps, [h["loss_mse"] for h in history], label="loss_mse")
    ax.set_title("Loss (checkpoints)")
    ax.set_yscale("log")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(steps, [max(h["grad_norm"], 1e-16) if math.isfinite(h["grad_norm"]) else float("nan") for h in history], linestyle="--", label="grad_norm")
    ax2.set_yscale("log")
    ax2.set_ylabel("grad norm")

    ax = axes[0, 1]
    ax.plot(steps, [max(h["eps_resid"], 1e-16) if math.isfinite(h["eps_resid"]) else float("nan") for h in history], label="eps_resid")
    ax.plot(steps, [max(h["resid_rms"], 1e-16) if math.isfinite(h["resid_rms"]) else float("nan") for h in history], label="resid_rms")
    ax.plot(steps, [max(h["stationarity_B_frob"], 1e-16) if math.isfinite(h["stationarity_B_frob"]) else float("nan") for h in history], label="stationarity_B_frob")
    ax.set_title("Residuals / stationarity")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, [max(h["delta_stationary_op"], 1e-16) if math.isfinite(h["delta_stationary_op"]) else float("nan") for h in history], label="delta_stationary_op")
    ax.plot(steps, [max(h["delta_eff_op"], 1e-16) if math.isfinite(h["delta_eff_op"]) else float("nan") for h in history], label="delta_eff_op")
    ax.plot(steps, [max(h["delta_fit_op"], 1e-16) if math.isfinite(h["delta_fit_op"]) else float("nan") for h in history], label="delta_fit_op")
    ax.set_title("Closure errors for $BB^T$")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, [max(h["gamma_eff_op"], 1e-16) if math.isfinite(h["gamma_eff_op"]) else float("nan") for h in history], label="gamma_eff_op")
    ax.plot(steps, [max(h["gamma_fit_op"], 1e-16) if math.isfinite(h["gamma_fit_op"]) else float("nan") for h in history], label="gamma_fit_op")
    ax.plot(steps, [max(h["alpha_rel"], 1e-16) if math.isfinite(h["alpha_rel"]) else float("nan") for h in history], label="alpha_rel")
    ax.set_title("Square law / relevant-subspace isotropy")
    ax.set_yscale("log")
    ax.legend()

    for ax in axes.ravel():
        ax.set_xlabel("checkpoint index")

    fig.suptitle(cfg.name)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_eigen_comparison(out_dir: Path, H: torch.Tensor, G: torch.Tensor) -> None:
    H = symmetrize(H)
    G = symmetrize(G)

    if (not is_finite_tensor(H)) or (not is_finite_tensor(G)):
        return

    try:
        evals_H = torch.flip(torch.linalg.eigvalsh(H), dims=[0]).cpu()
        evals_G = torch.flip(torch.linalg.eigvalsh(G), dims=[0]).cpu()
    except Exception:
        try:
            evals_H = torch.flip(torch.linalg.svdvals(H), dims=[0]).cpu()
            evals_G = torch.flip(torch.linalg.svdvals(G), dims=[0]).cpu()
        except Exception:
            return

    evals_G = torch.clamp(evals_G, min=0.0)
    evals_Gsqrt = torch.sqrt(evals_G)
    denom = float(torch.sum(evals_Gsqrt ** 2).item())
    s_fit = float(torch.sum(evals_H * evals_Gsqrt).item() / denom) if denom > 1e-12 else float("nan")

    ranks = list(range(1, len(evals_H) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(ranks, evals_H.numpy(), marker="o", markersize=3, label="eig(H = B^T B)")
    if math.isfinite(s_fit):
        plt.plot(ranks, (s_fit * evals_Gsqrt).numpy(), marker="x", markersize=3, label=r"best scale · eig($G^{1/2}$)")
    plt.yscale("log")
    plt.xlabel("eigenvalue rank")
    plt.ylabel("magnitude")
    plt.title("Best-checkpoint spectral comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eigen_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_final_matrices(out_dir: Path, H: torch.Tensor, G: torch.Tensor) -> None:
    payload = {
        "H_shape": list(H.shape),
        "G_shape": list(G.shape),
        "H_trace": float(torch.trace(H).item()) if is_finite_tensor(H) else float("nan"),
        "G_trace": float(torch.trace(G).item()) if is_finite_tensor(G) else float("nan"),
    }
    with open(out_dir / "final_matrix_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def default_configs() -> List[ExperimentConfig]:
    baseline = ExperimentConfig(name="baseline_isotropic", seed=0, anisotropic=False)
    anisotropic = ExperimentConfig(name="ablation_anisotropic", seed=0, anisotropic=True, spectrum_min=1e-2)
    return [baseline, anisotropic]


def main() -> None:
    configs = default_configs()
    all_results = []

    for cfg in configs:
        results = train_one(cfg)
        all_results.append({
            "name": cfg.name,
            "output_dir": results["output_dir"],
            "final_metrics": results["final_metrics"],
            "best_metrics_by_stationarity": results["best_metrics_by_stationarity"],
        })

    summary_path = Path(configs[0].output_root) / "summary.json"
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved summary to:", summary_path)
    for row in all_results:
        fm = row["final_metrics"]
        bm = row["best_metrics_by_stationarity"]
        print(
            f"- {row['name']}: {row['output_dir']} "
            f"| final delta_stat={fm['delta_stationary_op']:.3e}, delta_eff={fm['delta_eff_op']:.3e}, gamma_eff={fm['gamma_eff_op']:.3e} "
            f"| best delta_stat={bm['delta_stationary_op']:.3e}, delta_eff={bm['delta_eff_op']:.3e}, gamma_eff={bm['gamma_eff_op']:.3e}, gamma_fit={bm['gamma_fit_op']:.3e}"
        )


if __name__ == "__main__":
    main()
