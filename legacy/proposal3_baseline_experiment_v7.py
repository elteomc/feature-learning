#!/usr/bin/env python3
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


def best_scalar_fit(target: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12) -> float:
    if (not is_finite_tensor(target)) or (not is_finite_tensor(basis)):
        return float("nan")
    denom = torch.sum(basis * basis).item()
    if abs(denom) < eps:
        return float("nan")
    numer = torch.sum(target * basis).item()
    return numer / denom


def safe_psd_sqrt(mat: torch.Tensor) -> torch.Tensor:
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
    sgd_steps: int = 3000
    sgd_lr: float = 5e-2
    checkpoint_every: int = 20
    clip_grad_norm: float = 10.0
    use_lbfgs: bool = True
    lbfgs_outer_steps: int = 100
    lbfgs_history_size: int = 50
    lbfgs_lr: float = 0.5
    lbfgs_tolerance_grad: float = 1e-10
    lbfgs_tolerance_change: float = 1e-12
    lbfgs_log_every: int = 5
    lambda_B: float = 1e-2
    noise_std: float = 0.0
    anisotropic: bool = False
    spectrum_min: float = 1e-2
    beta_threshold: float = 1e-5
    dtype: str = "float64"
    device: str = "cpu"
    top_rank: int = 4
    output_root: str = "proposal3_outputs_v7"


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
    return {"X": X, "y": y, "B_star": B_star, "a_star": a_star, "cov_eigvals": eigvals}


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
def compute_metrics(model: TwoLayerSoftplus, X: torch.Tensor, y: torch.Tensor, lambda_B: float) -> Dict[str, float]:
    n, d = X.shape
    preds = model(X)
    r = preds - y

    B = model.B.detach()
    Q = compute_q(model, X)

    M = (Q.T @ Q) / float(n)
    RX = r.unsqueeze(1) * X
    T = Q.T @ (r.square().unsqueeze(1) * Q)
    M_tilde = T / float(n)
    A = Q.T @ RX
    S = A @ A.T

    H = B.T @ B
    BBt = B @ B.T
    G = B.T @ M @ B
    H2 = H @ H
    G_tilde = B.T @ M_tilde @ B

    B_stationary = -(Q.T @ RX) / (lambda_B * n)
    delta_stationary_op = safe_op_norm_symmetric(BBt - B_stationary @ B_stationary.T)

    d_eff = best_scalar_fit(S, T)
    kappa_eff = d_eff / (lambda_B ** 2 * n) if math.isfinite(d_eff) else float("nan")
    delta_eff_op = safe_op_norm_symmetric(BBt - kappa_eff * M_tilde) if math.isfinite(kappa_eff) else float("nan")
    gamma_tilde_eff_op = safe_op_norm_symmetric(H2 - kappa_eff * G_tilde) if math.isfinite(kappa_eff) else float("nan")

    beta_fit = best_scalar_fit(M_tilde, M)
    c_eff = float("nan")
    gamma_eff_op = float("nan")
    if math.isfinite(kappa_eff) and math.isfinite(beta_fit) and abs(kappa_eff * beta_fit) > 1e-16:
        c_eff = 1.0 / (kappa_eff * beta_fit)
        gamma_eff_op = safe_op_norm_symmetric(G - c_eff * H2)

    c_fit = best_scalar_fit(G, H2)
    gamma_fit_op = safe_op_norm_symmetric(G - c_fit * H2) if math.isfinite(c_fit) else float("nan")

    out = {
        "loss_total": float((0.5 * torch.mean((preds - y) ** 2) + 0.5 * lambda_B * torch.sum(B * B)).item()),
        "resid_rms": float(torch.sqrt(torch.mean(r.square())).item()),
        "delta_stationary_op": delta_stationary_op,
        "d_eff": float(d_eff),
        "kappa_eff": float(kappa_eff),
        "delta_eff_op": delta_eff_op,
        "beta_fit": float(beta_fit),
        "c_eff": float(c_eff),
        "gamma_eff_op": float(gamma_eff_op),
        "gamma_tilde_eff_op": float(gamma_tilde_eff_op),
        "c_fit": float(c_fit),
        "gamma_fit_op": float(gamma_fit_op),
    }
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


def metric_better(candidate: Dict[str, float], incumbent: Dict[str, float] | None, key: str, tie_key: str = "grad_norm") -> bool:
    if incumbent is None:
        return True
    cand = candidate.get(key, float("nan"))
    inc = incumbent.get(key, float("nan"))
    if not math.isfinite(cand):
        return False
    if not math.isfinite(inc):
        return True
    if cand < inc - 1e-15:
        return True
    if abs(cand - inc) <= 1e-15:
        cand_t = candidate.get(tie_key, float("inf"))
        inc_t = incumbent.get(tie_key, float("inf"))
        if math.isfinite(cand_t) and (not math.isfinite(inc_t) or cand_t < inc_t):
            return True
    return False


def qualifies_raw_conditioning(metrics: Dict[str, float], beta_threshold: float) -> bool:
    beta = metrics.get("beta_fit", float("nan"))
    return math.isfinite(beta) and abs(beta) >= beta_threshold


def annotate_phase_metrics(history: List[Dict[str, float]]) -> None:
    finite_gtilde = [h["gamma_tilde_eff_op"] for h in history if math.isfinite(h["gamma_tilde_eff_op"]) and h["gamma_tilde_eff_op"] > 0]
    best_gtilde = min(finite_gtilde) if finite_gtilde else float("nan")
    for h in history:
        ge = h.get("gamma_eff_op", float("nan"))
        gf = h.get("gamma_fit_op", float("nan"))
        gt = h.get("gamma_tilde_eff_op", float("nan"))

        raw_quality = float("nan")
        if math.isfinite(ge) and math.isfinite(gf) and gf > 0:
            raw_quality = ge / gf

        weighted_quality = float("nan")
        if math.isfinite(gt) and math.isfinite(best_gtilde) and best_gtilde > 0:
            weighted_quality = gt / best_gtilde

        crossover_score = float("nan")
        if math.isfinite(raw_quality) and raw_quality > 0 and math.isfinite(weighted_quality) and weighted_quality > 0:
            crossover_score = abs(math.log(raw_quality) - math.log(weighted_quality))

        h["raw_quality"] = raw_quality
        h["weighted_quality"] = weighted_quality
        h["crossover_score"] = crossover_score


def train_one(cfg: ExperimentConfig) -> Dict[str, object]:
    dtype = torch_dtype_from_name(cfg.dtype)
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    out_dir = Path(cfg.output_root) / cfg.name
    ensure_dir(out_dir)

    data = generate_teacher_student_dataset(cfg, dtype=dtype, device=device)
    X = data["X"]
    y = data["y"]

    model = TwoLayerSoftplus(cfg, dtype=dtype, device=device)

    history: List[Dict[str, float]] = []
    best_stationarity = None
    best_weighted = None
    best_raw_conditioning = None

    def maybe_update_best(metrics: Dict[str, float]) -> None:
        nonlocal best_stationarity, best_weighted, best_raw_conditioning
        if metric_better(metrics, best_stationarity, key="delta_stationary_op"):
            best_stationarity = copy.deepcopy(metrics)
        if metric_better(metrics, best_weighted, key="gamma_tilde_eff_op"):
            best_weighted = copy.deepcopy(metrics)
        if qualifies_raw_conditioning(metrics, cfg.beta_threshold):
            if metric_better(metrics, best_raw_conditioning, key="gamma_eff_op"):
                best_raw_conditioning = copy.deepcopy(metrics)

    def record(step: int, phase: str) -> None:
        metrics = compute_metrics(model, X, y, cfg.lambda_B)
        metrics["step"] = step
        metrics["phase"] = phase
        metrics["grad_norm"] = math.sqrt(grad_norm_sq(model))
        history.append(metrics)
        maybe_update_best(metrics)
        print(
            f"[{cfg.name}][{phase}] step={step:4d} "
            f"loss={metrics['loss_total']:.4e} "
            f"beta={metrics['beta_fit']:.4e} "
            f"gamma_tilde={metrics['gamma_tilde_eff_op']:.4e} "
            f"gamma_raw={metrics['gamma_eff_op']:.4e}"
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.sgd_lr)
    for step in range(cfg.sgd_steps + 1):
        optimizer.zero_grad()
        loss = objective(model, X, y, cfg.lambda_B)
        loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        if step % cfg.checkpoint_every == 0 or step == cfg.sgd_steps:
            record(step, phase="sgd")
        optimizer.step()

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

    annotate_phase_metrics(history)

    crossover = None
    for h in history:
        if metric_better(h, crossover, key="crossover_score"):
            crossover = copy.deepcopy(h)

    save_history(out_dir, cfg, history, best_stationarity, best_weighted, best_raw_conditioning, crossover)
    plot_phase_diagram(out_dir, cfg, history)

    return {
        "config": asdict(cfg),
        "history": history,
        "final_metrics": history[-1],
        "best_metrics_by_stationarity": best_stationarity,
        "best_metrics_by_weighted_law": best_weighted,
        "best_metrics_by_raw_conditioning": best_raw_conditioning,
        "crossover_metrics": crossover,
        "output_dir": str(out_dir),
    }


def save_history(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]],
                 best_stationarity: Dict[str, float] | None,
                 best_weighted: Dict[str, float] | None,
                 best_raw_conditioning: Dict[str, float] | None,
                 crossover: Dict[str, float] | None) -> None:
    payload = {
        "config": asdict(cfg),
        "best_metrics_by_stationarity": best_stationarity,
        "best_metrics_by_weighted_law": best_weighted,
        "best_metrics_by_raw_conditioning": best_raw_conditioning,
        "crossover_metrics": crossover,
        "history": history,
    }
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_phase_diagram(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    xs = [abs(h["beta_fit"]) if math.isfinite(h["beta_fit"]) else float("nan") for h in history]
    ys_raw = [h["raw_quality"] if math.isfinite(h.get("raw_quality", float("nan"))) else float("nan") for h in history]
    ys_weighted = [h["weighted_quality"] if math.isfinite(h.get("weighted_quality", float("nan"))) else float("nan") for h in history]

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys_raw, marker="o", markersize=3, label="raw_quality = gamma_eff / gamma_fit")
    plt.plot(xs, ys_weighted, marker="x", markersize=3, label="weighted_quality = gamma_tilde / best_gamma_tilde")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("|beta_fit|")
    plt.ylabel("quality ratio")
    plt.title(cfg.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_phase_diagram.png", dpi=180, bbox_inches="tight")
    plt.close()


def default_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(name="isotropic_noise_0p0", seed=0, anisotropic=False, noise_std=0.0, beta_threshold=1e-5),
        ExperimentConfig(name="anisotropic_noise_0p0", seed=0, anisotropic=True, spectrum_min=1e-2, noise_std=0.0, beta_threshold=1e-5),
        ExperimentConfig(name="isotropic_noise_0p001", seed=0, anisotropic=False, noise_std=1e-3, beta_threshold=1e-5),
        ExperimentConfig(name="anisotropic_noise_0p001", seed=0, anisotropic=True, spectrum_min=1e-2, noise_std=1e-3, beta_threshold=1e-5),
        ExperimentConfig(name="isotropic_noise_0p003", seed=0, anisotropic=False, noise_std=3e-3, beta_threshold=1e-5),
        ExperimentConfig(name="anisotropic_noise_0p003", seed=0, anisotropic=True, spectrum_min=1e-2, noise_std=3e-3, beta_threshold=1e-5),
    ]


def main() -> None:
    configs = default_configs()
    all_results = []
    for cfg in configs:
        results = train_one(cfg)
        all_results.append({
            "name": cfg.name,
            "output_dir": results["output_dir"],
            "config": results["config"],
            "final_metrics": results["final_metrics"],
            "best_metrics_by_stationarity": results["best_metrics_by_stationarity"],
            "best_metrics_by_weighted_law": results["best_metrics_by_weighted_law"],
            "best_metrics_by_raw_conditioning": results["best_metrics_by_raw_conditioning"],
            "crossover_metrics": results["crossover_metrics"],
        })

    summary_path = Path(configs[0].output_root) / "summary.json"
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved summary to:", summary_path)
    for row in all_results:
        cross = row["crossover_metrics"]
        print(
            f"- {row['name']}: crossover at step={cross['step']} phase={cross['phase']} "
            f"| beta={cross['beta_fit']:.3e}, raw_q={cross['raw_quality']:.3e}, weighted_q={cross['weighted_quality']:.3e}"
        )


if __name__ == "__main__":
    main()
