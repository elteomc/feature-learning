from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.weighted_metrics import compute_weighted_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float64": torch.float64}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return mapping[name]


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 0
    data_family: str = "gaussian"
    n: int = 128
    d: int = 256
    signal_rank: int = 8
    signal_noise_std: float = 0.5
    m_teacher: int = 4
    m_student: int = 16
    teacher_B_scale: float = 1.0
    teacher_a_scale: float = 1.0
    student_B_scale: float = 0.5
    student_a_scale: float = 0.5
    sgd_steps: int = 3000
    sgd_lr: float = 5e-2
    checkpoint_every: int = 20
    clip_grad_norm: float | None = 10.0
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
    output_root: str = "results/runs/pair_isotropy"


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


def sample_low_rank_inputs(
    cfg: ExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    basis_raw = torch.randn(cfg.d, cfg.signal_rank, dtype=dtype, device=device)
    basis, _ = torch.linalg.qr(basis_raw, mode="reduced")
    z = torch.randn(cfg.n, cfg.signal_rank, dtype=dtype, device=device)
    ambient_noise = torch.randn(cfg.n, cfg.d, dtype=dtype, device=device)
    X = z @ basis.T + cfg.signal_noise_std * ambient_noise
    eigvals = torch.full((cfg.d,), cfg.signal_noise_std ** 2, dtype=dtype, device=device)
    eigvals[: cfg.signal_rank] = eigvals[: cfg.signal_rank] + 1.0
    return X, eigvals, basis


def softplus_hidden(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return F.softplus(X @ B.T)


def sample_teacher(
    cfg: ExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
    signal_basis: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if signal_basis is None:
        B_star = cfg.teacher_B_scale * torch.randn(cfg.m_teacher, cfg.d, dtype=dtype, device=device) / math.sqrt(cfg.d)
    else:
        coeff = torch.randn(cfg.m_teacher, cfg.signal_rank, dtype=dtype, device=device) / math.sqrt(cfg.signal_rank)
        B_star = cfg.teacher_B_scale * (coeff @ signal_basis.T)
    a_star = cfg.teacher_a_scale * torch.randn(cfg.m_teacher, dtype=dtype, device=device) / math.sqrt(cfg.m_teacher)
    return B_star, a_star


def generate_teacher_student_dataset(cfg: ExperimentConfig, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor]:
    signal_basis = None
    if cfg.data_family == "gaussian":
        X, eigvals = sample_inputs(cfg, dtype=dtype, device=device)
    elif cfg.data_family == "low_rank_signal":
        X, eigvals, signal_basis = sample_low_rank_inputs(cfg, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported data_family {cfg.data_family!r}.")
    B_star, a_star = sample_teacher(cfg, dtype=dtype, device=device, signal_basis=signal_basis)
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


def objective(model: TwoLayerSoftplus, X: torch.Tensor, y: torch.Tensor, lambda_B: float) -> torch.Tensor:
    preds = model(X)
    mse = 0.5 * torch.mean((preds - y) ** 2)
    reg = 0.5 * lambda_B * torch.sum(model.B * model.B)
    return mse + reg


def grad_norm_sq(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None and torch.isfinite(param.grad).all():
            total += float(torch.sum(param.grad.detach() ** 2).item())
        elif param.grad is not None:
            return float("inf")
    return total


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
        cand_tie = candidate.get(tie_key, float("inf"))
        inc_tie = incumbent.get(tie_key, float("inf"))
        if math.isfinite(cand_tie) and (not math.isfinite(inc_tie) or cand_tie < inc_tie):
            return True
    return False


def qualifies_raw_conditioning(metrics: Dict[str, float], beta_threshold: float) -> bool:
    beta = metrics.get("beta_fit", float("nan"))
    return math.isfinite(beta) and abs(beta) >= beta_threshold


def annotate_phase_metrics(history: List[Dict[str, float]]) -> None:
    finite_weighted = [
        h["gamma_tilde_eff_op"]
        for h in history
        if math.isfinite(h["gamma_tilde_eff_op"]) and h["gamma_tilde_eff_op"] > 0
    ]
    best_weighted = min(finite_weighted) if finite_weighted else float("nan")
    for h in history:
        ge = h.get("gamma_eff_op", float("nan"))
        gf = h.get("gamma_fit_op", float("nan"))
        gt = h.get("gamma_tilde_eff_op", float("nan"))

        raw_quality = float("nan")
        if math.isfinite(ge) and math.isfinite(gf) and gf > 0:
            raw_quality = ge / gf

        weighted_quality = float("nan")
        if math.isfinite(gt) and math.isfinite(best_weighted) and best_weighted > 0:
            weighted_quality = gt / best_weighted

        crossover_score = float("nan")
        if math.isfinite(raw_quality) and raw_quality > 0 and math.isfinite(weighted_quality) and weighted_quality > 0:
            crossover_score = abs(math.log(raw_quality) - math.log(weighted_quality))

        h["raw_quality"] = raw_quality
        h["weighted_quality"] = weighted_quality
        h["crossover_score"] = crossover_score


def save_history(
    out_dir: Path,
    cfg: ExperimentConfig,
    history: List[Dict[str, float]],
    best_stationarity: Dict[str, float] | None,
    best_weighted: Dict[str, float] | None,
    best_raw_conditioning: Dict[str, float] | None,
    crossover: Dict[str, float] | None,
) -> None:
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


def _finite_positive_series(history: List[Dict[str, float]], key: str) -> Tuple[List[int], List[float]]:
    steps = []
    values = []
    for h in history:
        value = h.get(key, float("nan"))
        step = h.get("step", len(steps))
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            steps.append(int(step))
            values.append(float(value))
    return steps, values


def plot_beta_collapse(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    beta_steps, beta_values = _finite_positive_series(history, "beta_fit")
    mean_steps, mean_values = _finite_positive_series(history, "resid_mean_sq")
    max_steps, max_values = _finite_positive_series(history, "resid_max_sq")

    if not beta_values:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(beta_steps, beta_values, marker="o", markersize=3, label="beta_fit")
    if mean_values:
        plt.plot(mean_steps, mean_values, marker="x", markersize=3, label="mean residual squared")
    if max_values:
        plt.plot(max_steps, max_values, marker=".", markersize=3, label="max residual squared")
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("scale")
    plt.title(f"Beta collapse: {cfg.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "beta_collapse.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_two_regime_trajectory(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    series = [
        ("beta_fit", "beta_fit"),
        ("resid_mean_sq", "mean residual squared"),
        ("gamma_tilde_eff_op", "weighted law error"),
        ("gamma_eff_op", "raw law error"),
    ]
    plt.figure(figsize=(8, 6))
    plotted = False
    for key, label in series:
        steps, values = _finite_positive_series(history, key)
        if values:
            plt.plot(steps, values, marker="o", markersize=2, label=label)
            plotted = True
    if not plotted:
        plt.close()
        return
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("scale")
    plt.title(f"Two-regime trajectory: {cfg.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "two_regime_trajectory.png", dpi=180, bbox_inches="tight")
    plt.close()


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
        with torch.no_grad():
            preds = model(X)
            residuals = preds - y
            loss = objective(model, X, y, cfg.lambda_B)
            Q = compute_q(model, X)
            metrics = compute_weighted_metrics(
                B=model.B.detach(),
                X=X.detach(),
                Q=Q.detach(),
                residuals=residuals.detach(),
                lambda_B=cfg.lambda_B,
                loss_total=float(loss.item()),
            )
        metrics["step"] = step
        metrics["phase"] = phase
        metrics["grad_norm"] = math.sqrt(grad_norm_sq(model))
        history.append(metrics)
        maybe_update_best(metrics)
        print(
            f"[{cfg.name}][{phase}] step={step:4d} "
            f"loss={metrics['loss_total']:.4e} "
            f"A_pair={metrics['A_pair_op']:.4e} "
            f"gamma_tilde={metrics['gamma_tilde_eff_op']:.4e} "
            f"bound_ratio={metrics['theorem_bound_ratio']:.4e}"
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
            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                loss_value = objective(model, X, y, cfg.lambda_B)
                loss_value.backward()
                return loss_value

            try:
                lbfgs.step(closure)
            except Exception as exc:
                print(f"[{cfg.name}] LBFGS stopped early at outer step {outer} due to: {exc}")
                break
            if outer % cfg.lbfgs_log_every == 0 or outer == cfg.lbfgs_outer_steps:
                record(cfg.sgd_steps + outer, phase="lbfgs")

    annotate_phase_metrics(history)
    crossover = None
    for h in history:
        if metric_better(h, crossover, key="crossover_score"):
            crossover = copy.deepcopy(h)

    save_history(out_dir, cfg, history, best_stationarity, best_weighted, best_raw_conditioning, crossover)
    plot_beta_collapse(out_dir, cfg, history)
    plot_two_regime_trajectory(out_dir, cfg, history)

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
