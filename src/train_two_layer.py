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

from src.pair_isotropy import compute_pair_direction_diagnostics
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
    rare_fraction: float = 0.1
    rare_shift: float = 4.0
    rare_label_scale: float = 3.0
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


def sample_anisotropic_low_rank_inputs(
    cfg: ExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-rank signal whose within-subspace covariance is itself steeply
    anisotropic, plus small isotropic ambient noise.

    Unlike ``low_rank_signal`` (isotropic inside the signal subspace), here the
    signal directions carry variances ``lambda_j^2`` decaying geometrically from
    1 down to ``spectrum_min^2``. The teacher is supported on the same subspace,
    so the network learns those directions; the sample Gram ``X^T X`` is then
    strongly non-scalar precisely on the subspace the network uses. This is the
    intended stress test for high-gain pair closure (a large ``F_X`` aligned with
    the high-gain directions of ``G_stat``), while residuals stay roughly uniform
    so the beta link is unaffected.
    """
    basis_raw = torch.randn(cfg.d, cfg.signal_rank, dtype=dtype, device=device)
    basis, _ = torch.linalg.qr(basis_raw, mode="reduced")  # d x k, orthonormal columns
    log_min = math.log(cfg.spectrum_min)
    exponents = torch.linspace(0.0, 1.0, cfg.signal_rank, dtype=dtype, device=device)
    lam = torch.exp(log_min * exponents)  # length k, geometric from 1 down to spectrum_min
    z = torch.randn(cfg.n, cfg.signal_rank, dtype=dtype, device=device) * lam.unsqueeze(0)
    ambient_noise = torch.randn(cfg.n, cfg.d, dtype=dtype, device=device)
    X = z @ basis.T + cfg.signal_noise_std * ambient_noise
    eigvals = torch.full((cfg.d,), cfg.signal_noise_std ** 2, dtype=dtype, device=device)
    eigvals[: cfg.signal_rank] = eigvals[: cfg.signal_rank] + lam ** 2
    return X, eigvals, basis


def sample_clustered_inputs(
    cfg: ExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_clusters = max(2, cfg.signal_rank)
    centers = torch.randn(n_clusters, cfg.d, dtype=dtype, device=device) / math.sqrt(cfg.d)
    assignments = torch.arange(cfg.n, device=device) % n_clusters
    assignments = assignments[torch.randperm(cfg.n, device=device)]
    X = centers[assignments] + cfg.signal_noise_std * torch.randn(cfg.n, cfg.d, dtype=dtype, device=device)
    eigvals = torch.var(X, dim=0, unbiased=False).clamp_min(1e-12)
    return X, eigvals


def sample_mixture_subspace_inputs(
    cfg: ExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_subspaces = 2
    rank = max(1, cfg.signal_rank)
    bases = []
    for _ in range(n_subspaces):
        raw = torch.randn(cfg.d, rank, dtype=dtype, device=device)
        basis, _ = torch.linalg.qr(raw, mode="reduced")
        bases.append(basis)

    X_parts = []
    for i in range(cfg.n):
        basis = bases[i % n_subspaces]
        coeff = torch.randn(rank, dtype=dtype, device=device) / math.sqrt(rank)
        noise = cfg.signal_noise_std * torch.randn(cfg.d, dtype=dtype, device=device)
        X_parts.append(coeff @ basis.T + noise)

    X = torch.stack(X_parts, dim=0)
    eigvals = torch.var(X, dim=0, unbiased=False).clamp_min(1e-12)
    return X, eigvals, bases[0]


def sample_rare_region_inputs(
    cfg: ExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.randn(cfg.n, cfg.d, dtype=dtype, device=device)
    rare_count = max(1, int(round(cfg.rare_fraction * cfg.n)))
    direction = torch.randn(cfg.d, dtype=dtype, device=device)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    rare_mask = torch.zeros(cfg.n, dtype=torch.bool, device=device)
    rare_mask[:rare_count] = True
    rare_mask = rare_mask[torch.randperm(cfg.n, device=device)]
    X[rare_mask] = X[rare_mask] + cfg.rare_shift * direction
    eigvals = torch.var(X, dim=0, unbiased=False).clamp_min(1e-12)
    return X, eigvals, rare_mask


def sample_xor_inputs(
    cfg: ExperimentConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(cfg.n, cfg.d, dtype=dtype, device=device)
    eigvals = torch.ones(cfg.d, dtype=dtype, device=device)
    return X, eigvals


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
    rare_mask = None
    if cfg.data_family == "gaussian":
        X, eigvals = sample_inputs(cfg, dtype=dtype, device=device)
    elif cfg.data_family == "low_rank_signal":
        X, eigvals, signal_basis = sample_low_rank_inputs(cfg, dtype=dtype, device=device)
    elif cfg.data_family == "anisotropic_low_rank":
        X, eigvals, signal_basis = sample_anisotropic_low_rank_inputs(cfg, dtype=dtype, device=device)
    elif cfg.data_family == "clustered_gaussian":
        X, eigvals = sample_clustered_inputs(cfg, dtype=dtype, device=device)
    elif cfg.data_family == "mixture_subspaces":
        X, eigvals, signal_basis = sample_mixture_subspace_inputs(cfg, dtype=dtype, device=device)
    elif cfg.data_family == "rare_region_outliers":
        X, eigvals, rare_mask = sample_rare_region_inputs(cfg, dtype=dtype, device=device)
    elif cfg.data_family in {"two_region_gating", "xor_feature"}:
        X, eigvals = sample_xor_inputs(cfg, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported data_family {cfg.data_family!r}.")
    if cfg.data_family == "two_region_gating":
        B_left, a_left = sample_teacher(cfg, dtype=dtype, device=device)
        B_right, a_right = sample_teacher(cfg, dtype=dtype, device=device)
        left_values = softplus_hidden(X, B_left) @ a_left
        right_values = softplus_hidden(X, B_right) @ a_right
        y_clean = torch.where(X[:, 0] < 0, left_values, right_values)
        B_star, a_star = B_left, a_left
    elif cfg.data_family == "xor_feature":
        y_clean = cfg.teacher_a_scale * torch.tanh(2.0 * X[:, 0] * X[:, 1])
        B_star, a_star = sample_teacher(cfg, dtype=dtype, device=device, signal_basis=signal_basis)
    else:
        B_star, a_star = sample_teacher(cfg, dtype=dtype, device=device, signal_basis=signal_basis)
        y_clean = softplus_hidden(X, B_star) @ a_star
        if cfg.data_family == "rare_region_outliers" and rare_mask is not None:
            y_clean = y_clean + cfg.rare_label_scale * rare_mask.to(dtype)
    y = y_clean + cfg.noise_std * torch.randn_like(y_clean)
    payload = {"X": X, "y": y, "B_star": B_star, "a_star": a_star, "cov_eigvals": eigvals}
    if rare_mask is not None:
        payload["rare_mask"] = rare_mask
    return payload


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


def _corr_from_lists(xs: List[float], ys: List[float], eps: float = 1e-12) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if isinstance(x, (int, float))
        and isinstance(y, (int, float))
        and math.isfinite(float(x))
        and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return float("nan")
    x_values = torch.tensor([x for x, _ in pairs], dtype=torch.float64)
    y_values = torch.tensor([y for _, y in pairs], dtype=torch.float64)
    x_centered = x_values - torch.mean(x_values)
    y_centered = y_values - torch.mean(y_values)
    denom = torch.sqrt(torch.mean(x_centered.square())) * torch.sqrt(torch.mean(y_centered.square()))
    denom_value = float(denom.item())
    if not math.isfinite(denom_value) or denom_value <= eps:
        return float("nan")
    return float((torch.mean(x_centered * y_centered) / denom).item())


def annotate_damping_metrics(history: List[Dict[str, float]], sample_history: List[Dict[str, object]]) -> None:
    if len(history) != len(sample_history):
        return
    if history:
        history[0]["leverage_damping_corr"] = float("nan")
        history[0]["mean_log_resid_sq_decay"] = float("nan")
    for idx in range(1, len(sample_history)):
        previous = sample_history[idx - 1]
        current = sample_history[idx]
        prev_resid = previous.get("resid_sq", [])
        curr_resid = current.get("resid_sq", [])
        leverage = previous.get("leverage", [])
        if not isinstance(prev_resid, list) or not isinstance(curr_resid, list) or not isinstance(leverage, list):
            continue
        damping = [
            math.log(max(float(prev), 1e-30)) - math.log(max(float(curr), 1e-30))
            for prev, curr in zip(prev_resid, curr_resid)
            if isinstance(prev, (int, float)) and isinstance(curr, (int, float))
        ]
        corr = _corr_from_lists(leverage, damping)
        mean_decay = float(sum(damping) / len(damping)) if damping else float("nan")
        history[idx]["leverage_damping_corr"] = corr
        history[idx]["mean_log_resid_sq_decay"] = mean_decay


def save_history(
    out_dir: Path,
    cfg: ExperimentConfig,
    history: List[Dict[str, float]],
    best_stationarity: Dict[str, float] | None,
    best_weighted: Dict[str, float] | None,
    best_raw_conditioning: Dict[str, float] | None,
    crossover: Dict[str, float] | None,
    pair_direction_history: List[Dict[str, object]] | None = None,
    sample_history: List[Dict[str, object]] | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "best_metrics_by_stationarity": best_stationarity,
        "best_metrics_by_weighted_law": best_weighted,
        "best_metrics_by_raw_conditioning": best_raw_conditioning,
        "crossover_metrics": crossover,
        "history": history,
    }
    if pair_direction_history is not None:
        payload["pair_direction_history"] = pair_direction_history
    if sample_history is not None:
        payload["sample_history"] = sample_history
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


def plot_beta_fit_diagnostics(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    series = [
        ("beta_rel_mean_sq_error", "actual beta relative error"),
        ("beta_cv_bound", "CV leverage times CV residual"),
        ("leverage_cv", "leverage CV"),
        ("leverage_resid_sq_corr", "absolute leverage-residual correlation"),
    ]
    plt.figure(figsize=(8, 6))
    plotted = False
    for key, label in series:
        steps = []
        values = []
        for h in history:
            value = h.get(key, float("nan"))
            if key == "leverage_resid_sq_corr" and isinstance(value, (int, float)) and math.isfinite(value):
                value = abs(value)
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                steps.append(int(h.get("step", len(steps))))
                values.append(float(value))
        if values:
            plt.plot(steps, values, marker="o", markersize=2, label=label)
            plotted = True
    if not plotted:
        plt.close()
        return
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("diagnostic scale")
    plt.title(f"Beta-fit diagnostics: {cfg.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "beta_fit_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_beta_decomposition_identity(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    plt.figure(figsize=(8, 6))
    plotted = False
    for key, label in [
        ("beta_rel_mean_sq_error", "absolute beta relative error"),
        ("beta_corr_cv_product_abs", "absolute corr times CV product"),
    ]:
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
    plt.title(f"Beta decomposition identity: {cfg.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "beta_decomposition_identity.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_pair_gain_diagnostics(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    series = [
        ("A_pair_op", "support worst direction"),
        ("pair_top_abs_defect", "top pair defect"),
        ("pair_top_defect_gain", "gain on top defect"),
        ("pair_high_gain_closure_op", "high-gain scalar closure"),
        ("pair_weighted_contribution_max", "max gain-weighted contribution"),
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
    plt.ylabel("diagnostic scale")
    plt.title(f"Pair gain diagnostics: {cfg.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pair_gain_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_high_gain_closure(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    series = [
        ("pair_high_gain_closure_op", "high-gain closure"),
        ("pair_low_gain_op", "low-gain operator norm"),
        ("pair_gain_op", "full gain operator norm"),
        ("pair_damping_bound_proxy", "damping bound proxy"),
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
    plt.ylabel("diagnostic scale")
    plt.title(f"High-gain scalar closure: {cfg.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "high_gain_closure.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_pair_direction_scatter(out_dir: Path, cfg: ExperimentConfig, pair_direction_history: List[Dict[str, object]]) -> None:
    if not pair_direction_history:
        return
    checkpoint = pair_direction_history[-1]
    abs_defects = checkpoint.get("abs_defects", [])
    gains = checkpoint.get("gains", [])
    weighted = checkpoint.get("weighted_contributions", [])
    if not isinstance(abs_defects, list) or not isinstance(gains, list) or not isinstance(weighted, list):
        return
    if not abs_defects or not gains:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(abs_defects, gains, c=weighted, cmap="viridis", s=42)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("absolute pair defect")
    plt.ylabel("stationarity-induced gain")
    plt.title(f"Pair defect versus gain: {cfg.name}")
    plt.colorbar(label="gain-weighted contribution")
    plt.tight_layout()
    plt.savefig(out_dir / "pair_defect_gain_scatter.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_pair_direction_cumulative(out_dir: Path, cfg: ExperimentConfig, pair_direction_history: List[Dict[str, object]]) -> None:
    if not pair_direction_history:
        return
    checkpoint = pair_direction_history[-1]
    defect_share = checkpoint.get("cumulative_defect_share", [])
    weighted_share = checkpoint.get("cumulative_weighted_share", [])
    if not isinstance(defect_share, list) or not isinstance(weighted_share, list):
        return
    if not defect_share or not weighted_share:
        return
    xs = list(range(1, min(len(defect_share), len(weighted_share)) + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(xs, defect_share[: len(xs)], marker="o", markersize=3, label="pair defect share")
    plt.plot(xs, weighted_share[: len(xs)], marker="x", markersize=3, label="gain-weighted share")
    plt.xlabel("top pair-defect directions")
    plt.ylabel("cumulative share")
    plt.title(f"Cumulative pair contributions: {cfg.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pair_direction_cumulative.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_residual_damping_vs_leverage(out_dir: Path, cfg: ExperimentConfig, sample_history: List[Dict[str, object]]) -> None:
    if len(sample_history) < 2:
        return
    xs = []
    ys = []
    for previous, current in zip(sample_history[:-1], sample_history[1:]):
        prev_resid = previous.get("resid_sq", [])
        curr_resid = current.get("resid_sq", [])
        leverage = previous.get("leverage", [])
        if not isinstance(prev_resid, list) or not isinstance(curr_resid, list) or not isinstance(leverage, list):
            continue
        for lev, prev, curr in zip(leverage, prev_resid, curr_resid):
            if isinstance(lev, (int, float)) and isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
                xs.append(float(lev))
                ys.append(math.log(max(float(prev), 1e-30)) - math.log(max(float(curr), 1e-30)))
    if not xs:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=12, alpha=0.45)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    plt.xlabel("hidden-gradient leverage")
    plt.ylabel("log residual-energy decay")
    plt.title(f"Residual damping versus leverage: {cfg.name}")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_damping_vs_leverage.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_phase_diagram(out_dir: Path, cfg: ExperimentConfig, history: List[Dict[str, float]]) -> None:
    xs = []
    ys = []
    colors = []
    for h in history:
        beta = h.get("beta_fit", float("nan"))
        raw_quality = h.get("raw_quality", float("nan"))
        weighted_quality = h.get("weighted_quality", float("nan"))
        step = h.get("step", len(xs))
        if (
            isinstance(beta, (int, float))
            and isinstance(raw_quality, (int, float))
            and isinstance(weighted_quality, (int, float))
            and math.isfinite(beta)
            and math.isfinite(raw_quality)
            and math.isfinite(weighted_quality)
            and abs(beta) > 0
            and raw_quality > 0
            and weighted_quality > 0
        ):
            xs.append(abs(float(beta)))
            ys.append(float(raw_quality / weighted_quality))
            colors.append(float(step))

    if not xs:
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xs, ys, c=colors, cmap="viridis", s=35)
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(1.0, color="black", linewidth=1, alpha=0.4)
    plt.xlabel("|beta_fit|")
    plt.ylabel("raw quality / weighted quality")
    plt.title(f"Two-regime phase diagram: {cfg.name}")
    plt.colorbar(scatter, label="training step")
    plt.tight_layout()
    plt.savefig(out_dir / "phase_diagram.png", dpi=180, bbox_inches="tight")
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
    pair_direction_history: List[Dict[str, object]] = []
    sample_history: List[Dict[str, object]] = []
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
            hidden_kernel_sq = (Q @ Q.T).square()
            leverage_scores = hidden_kernel_sq.sum(dim=1)
            leverage_mean = torch.mean(leverage_scores)
            if float(torch.abs(leverage_mean).item()) > 1e-12:
                leverage_normalized = leverage_scores / leverage_mean
            else:
                leverage_normalized = torch.full_like(leverage_scores, float("nan"))
            pair_payload = compute_pair_direction_diagnostics(
                C=Q.detach().T * residuals.detach().unsqueeze(0),
                X=X.detach(),
                d_eff=metrics.get("d_eff", float("nan")),
            )
        metrics["step"] = step
        metrics["phase"] = phase
        metrics["grad_norm"] = math.sqrt(grad_norm_sq(model))
        history.append(metrics)
        pair_payload["step"] = step
        pair_payload["phase"] = phase
        pair_direction_history.append(pair_payload)
        sample_history.append(
            {
                "step": step,
                "phase": phase,
                "resid_sq": [float(value) for value in residuals.detach().square().cpu().tolist()],
                "leverage": [float(value) for value in leverage_normalized.detach().cpu().tolist()],
            }
        )
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
    annotate_damping_metrics(history, sample_history)
    crossover = None
    for h in history:
        if metric_better(h, crossover, key="crossover_score"):
            crossover = copy.deepcopy(h)

    save_history(
        out_dir,
        cfg,
        history,
        best_stationarity,
        best_weighted,
        best_raw_conditioning,
        crossover,
        pair_direction_history,
        sample_history,
    )
    plot_beta_collapse(out_dir, cfg, history)
    plot_beta_fit_diagnostics(out_dir, cfg, history)
    plot_beta_decomposition_identity(out_dir, cfg, history)
    plot_pair_gain_diagnostics(out_dir, cfg, history)
    plot_high_gain_closure(out_dir, cfg, history)
    plot_pair_direction_scatter(out_dir, cfg, pair_direction_history)
    plot_pair_direction_cumulative(out_dir, cfg, pair_direction_history)
    plot_residual_damping_vs_leverage(out_dir, cfg, sample_history)
    plot_phase_diagram(out_dir, cfg, history)
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
