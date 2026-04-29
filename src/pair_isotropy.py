from __future__ import annotations

import math
from typing import Dict

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
