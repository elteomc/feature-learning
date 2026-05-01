from __future__ import annotations

import math
from typing import Dict, List

import torch


def symmetrize(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.T)


def is_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


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
    return float(numer / denom)


def _nan_pair_spectral_metrics() -> Dict[str, float]:
    return {
        "pair_spectral_rank": 0,
        "pair_spectral_defect_op": float("nan"),
        "pair_top_abs_defect": float("nan"),
        "pair_top_defect_gain": float("nan"),
        "pair_top_defect_weighted": float("nan"),
        "pair_top_gain": float("nan"),
        "pair_top_gain_abs_defect": float("nan"),
        "pair_top_gain_weighted": float("nan"),
        "pair_gain_defect_corr_abs": float("nan"),
        "pair_weighted_contribution_max": float("nan"),
        "pair_weighted_contribution_sum": float("nan"),
        "pair_high_gain_rank": 0,
        "pair_high_gain_closure_op": float("nan"),
        "pair_low_gain_op": float("nan"),
        "pair_gain_op": float("nan"),
        "pair_damping_bound_proxy": float("nan"),
    }


def _safe_corr_abs(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    denom = torch.sqrt(torch.mean(x_centered.square())) * torch.sqrt(torch.mean(y_centered.square()))
    denom_value = float(denom.item())
    if not math.isfinite(denom_value) or denom_value <= eps:
        return float("nan")
    corr = torch.mean(x_centered * y_centered) / denom
    return float(torch.abs(corr).item())


def compute_pair_spectral_gain_metrics(
    *,
    C: torch.Tensor,
    X: torch.Tensor,
    d_eff: float,
    eps_rel: float = 1e-10,
) -> Dict[str, float]:
    """Measure whether large pair defects have low stationarity-induced gain."""

    if (not math.isfinite(d_eff)) or (not is_finite_tensor(C)) or (not is_finite_tensor(X)):
        return _nan_pair_spectral_metrics()

    try:
        _, singular_values, Vh = torch.linalg.svd(C, full_matrices=False)
    except Exception:
        return _nan_pair_spectral_metrics()

    if singular_values.numel() == 0:
        return _nan_pair_spectral_metrics()

    max_singular = torch.max(singular_values).clamp_min(torch.tensor(1.0, dtype=C.dtype, device=C.device))
    keep = singular_values > eps_rel * max_singular
    if not torch.any(keep):
        return _nan_pair_spectral_metrics()

    sigma_sq = singular_values[keep].square()
    Vh_pos = Vh[keep, :]
    rank = int(keep.sum().item())
    H_X = symmetrize(Vh_pos @ X @ X.T @ Vh_pos.T)
    defect = symmetrize(H_X - d_eff * torch.eye(rank, dtype=X.dtype, device=X.device))

    try:
        defect_evals, defect_evecs = torch.linalg.eigh(defect)
    except Exception:
        return _nan_pair_spectral_metrics()

    abs_defects = torch.abs(defect_evals)
    gain_inputs = (sigma_sq.unsqueeze(1) * defect_evecs)
    gain_vectors = X.T @ Vh_pos.T @ gain_inputs
    gains = torch.linalg.vector_norm(gain_vectors, dim=0)
    weighted = abs_defects * gains.square()
    gain_operator = sigma_sq.unsqueeze(1) * (Vh_pos @ X)
    gain_op = float(torch.linalg.matrix_norm(gain_operator, ord=2).item())
    top_gain_count = max(1, int(math.ceil(0.25 * rank)))

    try:
        gain_left, _, _ = torch.linalg.svd(gain_operator, full_matrices=False)
        high_basis = gain_left[:, :top_gain_count]
        high_core = symmetrize(high_basis.T @ defect @ high_basis)
        high_gain_closure_op = safe_op_norm_symmetric(high_core)
        projector_hi = high_basis @ high_basis.T
        low_gain_operator = (torch.eye(rank, dtype=X.dtype, device=X.device) - projector_hi) @ gain_operator
        low_gain_op = float(torch.linalg.matrix_norm(low_gain_operator, ord=2).item())
    except Exception:
        high_gain_closure_op = float("nan")
        low_gain_op = float("nan")

    pair_spectral_defect_op = float(torch.max(abs_defects).item())
    damping_bound_proxy = float("nan")
    if math.isfinite(high_gain_closure_op) and math.isfinite(low_gain_op) and math.isfinite(gain_op):
        damping_bound_proxy = (
            high_gain_closure_op * gain_op ** 2
            + 2.0 * pair_spectral_defect_op * gain_op * low_gain_op
            + pair_spectral_defect_op * low_gain_op ** 2
        )

    top_defect_idx = int(torch.argmax(abs_defects).item())
    top_gain_idx = int(torch.argmax(gains).item())
    return {
        "pair_spectral_rank": rank,
        "pair_spectral_defect_op": pair_spectral_defect_op,
        "pair_top_abs_defect": float(abs_defects[top_defect_idx].item()),
        "pair_top_defect_gain": float(gains[top_defect_idx].item()),
        "pair_top_defect_weighted": float(weighted[top_defect_idx].item()),
        "pair_top_gain": float(gains[top_gain_idx].item()),
        "pair_top_gain_abs_defect": float(abs_defects[top_gain_idx].item()),
        "pair_top_gain_weighted": float(weighted[top_gain_idx].item()),
        "pair_gain_defect_corr_abs": _safe_corr_abs(abs_defects, gains),
        "pair_weighted_contribution_max": float(torch.max(weighted).item()),
        "pair_weighted_contribution_sum": float(torch.sum(weighted).item()),
        "pair_high_gain_rank": top_gain_count,
        "pair_high_gain_closure_op": float(high_gain_closure_op),
        "pair_low_gain_op": float(low_gain_op),
        "pair_gain_op": gain_op,
        "pair_damping_bound_proxy": float(damping_bound_proxy),
    }


def _tensor_to_floats(values: torch.Tensor) -> List[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


def compute_pair_direction_diagnostics(
    *,
    C: torch.Tensor,
    X: torch.Tensor,
    d_eff: float,
    eps_rel: float = 1e-10,
) -> Dict[str, object]:
    if (not math.isfinite(d_eff)) or (not is_finite_tensor(C)) or (not is_finite_tensor(X)):
        return {}

    try:
        _, singular_values, Vh = torch.linalg.svd(C, full_matrices=False)
    except Exception:
        return {}

    if singular_values.numel() == 0:
        return {}

    max_singular = torch.max(singular_values).clamp_min(torch.tensor(1.0, dtype=C.dtype, device=C.device))
    keep = singular_values > eps_rel * max_singular
    if not torch.any(keep):
        return {}

    sigma_sq = singular_values[keep].square()
    Vh_pos = Vh[keep, :]
    rank = int(keep.sum().item())
    H_X = symmetrize(Vh_pos @ X @ X.T @ Vh_pos.T)
    defect = symmetrize(H_X - d_eff * torch.eye(rank, dtype=X.dtype, device=X.device))

    try:
        defect_evals, defect_evecs = torch.linalg.eigh(defect)
    except Exception:
        return {}

    abs_defects = torch.abs(defect_evals)
    gain_inputs = sigma_sq.unsqueeze(1) * defect_evecs
    gain_vectors = X.T @ Vh_pos.T @ gain_inputs
    gains = torch.linalg.vector_norm(gain_vectors, dim=0)
    weighted = abs_defects * gains.square()

    order = torch.argsort(abs_defects, descending=True)
    sorted_abs_defects = abs_defects[order]
    sorted_weighted = weighted[order]
    defect_total = torch.sum(sorted_abs_defects).clamp_min(torch.tensor(1e-30, dtype=X.dtype, device=X.device))
    weighted_total = torch.sum(sorted_weighted).clamp_min(torch.tensor(1e-30, dtype=X.dtype, device=X.device))

    return {
        "rank": rank,
        "abs_defects": _tensor_to_floats(abs_defects),
        "gains": _tensor_to_floats(gains),
        "weighted_contributions": _tensor_to_floats(weighted),
        "cumulative_defect_share": _tensor_to_floats(torch.cumsum(sorted_abs_defects, dim=0) / defect_total),
        "cumulative_weighted_share": _tensor_to_floats(torch.cumsum(sorted_weighted, dim=0) / weighted_total),
    }


def compute_pair_isotropy_metrics(
    *,
    S: torch.Tensor,
    T: torch.Tensor,
    B: torch.Tensor,
    lambda_B: float,
    n: int,
    d_eff: float,
    E_stat_op: float,
    gamma_tilde_eff_op: float,
    eps_rel: float = 1e-10,
) -> Dict[str, float]:
    """Compute the support-aware pair-isotropy defect and theorem bound.

    Here S and T use the unnormalized theorem convention:
    S = Q^T R X X^T R Q and T = Q^T R^2 Q.
    """

    T_sym = symmetrize(T)
    S_sym = symmetrize(S)
    evals, evecs = torch.linalg.eigh(T_sym)
    max_eval = torch.max(torch.abs(evals)).clamp_min(torch.tensor(1.0, dtype=T.dtype, device=T.device))
    keep = evals > eps_rel * max_eval

    if torch.any(keep) and math.isfinite(d_eff):
        evecs_pos = evecs[:, keep]
        evals_pos = evals[keep]
        T_dag_half = evecs_pos @ torch.diag(evals_pos.rsqrt()) @ evecs_pos.T
        projector_T = evecs_pos @ evecs_pos.T
        pair_core = symmetrize(T_dag_half @ S_sym @ T_dag_half - d_eff * projector_T)
        A_pair_op = safe_op_norm_symmetric(pair_core)
        pair_core_frob = float(torch.linalg.matrix_norm(pair_core, ord="fro").item())
        rank_T = int(keep.sum().item())
    else:
        A_pair_op = float("nan")
        pair_core_frob = float("nan")
        rank_T = 0

    pair_fit_op = float("nan")
    pair_fit_rel_frob = float("nan")
    if math.isfinite(d_eff):
        pair_fit = S_sym - d_eff * T_sym
        pair_fit_op = safe_op_norm_symmetric(pair_fit)
        denom = torch.linalg.matrix_norm(S_sym, ord="fro").item()
        if denom > 0:
            pair_fit_rel_frob = float(torch.linalg.matrix_norm(pair_fit, ord="fro").item() / denom)

    B_op = float(torch.linalg.matrix_norm(B, ord=2).item())
    T_op = safe_op_norm_symmetric(T_sym)
    pair_bound_component_op = float("nan")
    theorem_bound_op = float("nan")
    theorem_bound_ratio = float("nan")

    if math.isfinite(A_pair_op) and math.isfinite(T_op) and math.isfinite(E_stat_op):
        pair_bound_component_op = (T_op / (lambda_B ** 2 * n ** 2)) * A_pair_op
        theorem_bound_op = (B_op ** 2) * (pair_bound_component_op + E_stat_op)
        if theorem_bound_op > 0 and math.isfinite(gamma_tilde_eff_op):
            theorem_bound_ratio = gamma_tilde_eff_op / theorem_bound_op

    return {
        "A_pair_op": float(A_pair_op),
        "pair_core_frob": float(pair_core_frob),
        "pair_fit_op": float(pair_fit_op),
        "pair_fit_rel_frob": float(pair_fit_rel_frob),
        "pair_bound_component_op": float(pair_bound_component_op),
        "theorem_bound_op": float(theorem_bound_op),
        "theorem_bound_ratio": float(theorem_bound_ratio),
        "B_op": float(B_op),
        "T_op": float(T_op),
        "rank_T": rank_T,
    }
