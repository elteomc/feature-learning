from __future__ import annotations

import math
from typing import Dict

import torch

from src.pair_isotropy import (
    best_scalar_fit,
    compute_pair_isotropy_metrics,
    compute_pair_spectral_gain_metrics,
    safe_op_norm_symmetric,
    symmetrize,
)


def safe_psd_sqrt(mat: torch.Tensor) -> torch.Tensor:
    mat = symmetrize(mat)
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


def _relative_residual(numer: float, left: float, right_scale: float, right: float, eps: float = 1e-12) -> float:
    denom = left + abs(right_scale) * right + eps
    if not math.isfinite(numer) or not math.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(numer / denom)


def _safe_ratio(numer: float, denom: float, eps: float = 1e-12) -> float:
    if not math.isfinite(numer) or not math.isfinite(denom) or abs(denom) <= eps:
        return float("nan")
    return float(numer / denom)


def _safe_cv(values: torch.Tensor, mean_value: float, eps: float = 1e-12) -> float:
    if not math.isfinite(mean_value) or abs(mean_value) <= eps:
        return float("nan")
    centered = values - mean_value
    std = torch.sqrt(torch.mean(centered * centered))
    return float((std / abs(mean_value)).item())


def _safe_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    x_std = torch.sqrt(torch.mean(x_centered * x_centered))
    y_std = torch.sqrt(torch.mean(y_centered * y_centered))
    denom = float((x_std * y_std).item())
    if not math.isfinite(denom) or denom <= eps:
        return float("nan")
    numer = torch.mean(x_centered * y_centered)
    return float((numer / (x_std * y_std)).item())


@torch.no_grad()
def compute_weighted_metrics(
    *,
    B: torch.Tensor,
    X: torch.Tensor,
    Q: torch.Tensor,
    residuals: torch.Tensor,
    lambda_B: float,
    loss_total: float | None = None,
    eps_rel: float = 1e-10,
) -> Dict[str, float]:
    """Compute weighted-law, raw-law, stationarity, and pair-isotropy metrics.

    Conventions:
    - X has shape (n, d)
    - Q has shape (n, m)
    - B has shape (m, d)
    - residuals has shape (n,)
    """

    n = X.shape[0]
    r = residuals

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

    Z = lambda_B * n * B + A
    E_stat = symmetrize((-A @ Z.T - Z @ A.T + Z @ Z.T) / (lambda_B ** 2 * n ** 2))
    E_stat_op = safe_op_norm_symmetric(E_stat)

    B_stationary = -A / (lambda_B * n)
    delta_stationary_op = safe_op_norm_symmetric(BBt - B_stationary @ B_stationary.T)

    d_eff = best_scalar_fit(S, T)
    kappa_eff = d_eff / (lambda_B ** 2 * n) if math.isfinite(d_eff) else float("nan")
    delta_eff_op = safe_op_norm_symmetric(BBt - kappa_eff * M_tilde) if math.isfinite(kappa_eff) else float("nan")
    gamma_tilde_eff_op = safe_op_norm_symmetric(H2 - kappa_eff * G_tilde) if math.isfinite(kappa_eff) else float("nan")

    pair_error = S - d_eff * T if math.isfinite(d_eff) else torch.full_like(S, float("nan"))
    pair_push_op = safe_op_norm_symmetric(B.T @ pair_error @ B)
    pair_push_scaled_op = pair_push_op / (lambda_B ** 2 * n ** 2) if math.isfinite(pair_push_op) else float("nan")

    beta_fit = best_scalar_fit(M_tilde, M)
    M_bridge = M_tilde - beta_fit * M if math.isfinite(beta_fit) else torch.full_like(M, float("nan"))
    M_bridge_op = safe_op_norm_symmetric(M_bridge)
    M_bridge_frob = (
        float(torch.linalg.matrix_norm(M_bridge, ord="fro").item())
        if bool(torch.isfinite(M_bridge).all().item())
        else float("nan")
    )
    M_tilde_frob = (
        float(torch.linalg.matrix_norm(M_tilde, ord="fro").item())
        if bool(torch.isfinite(M_tilde).all().item())
        else float("nan")
    )
    M_bridge_rel_frob = _safe_ratio(M_bridge_frob, M_tilde_frob)
    resid_sq = r.square()
    resid_mean_sq = float(torch.mean(resid_sq).item())
    resid_max_sq = float(torch.max(resid_sq).item())
    beta_over_resid_mean_sq = _safe_ratio(beta_fit, resid_mean_sq)
    beta_over_resid_max_sq = _safe_ratio(beta_fit, resid_max_sq)
    beta_rel_mean_sq_error = abs(beta_over_resid_mean_sq - 1.0) if math.isfinite(beta_over_resid_mean_sq) else float("nan")

    hidden_kernel_sq = (Q @ Q.T).square()
    leverage_scores = hidden_kernel_sq.sum(dim=1)
    leverage_mean = float(torch.mean(leverage_scores).item())
    leverage_normalized = leverage_scores / leverage_mean if abs(leverage_mean) > 1e-12 else torch.full_like(leverage_scores, float("nan"))
    leverage_cv = _safe_cv(leverage_normalized, 1.0)
    resid_sq_cv = _safe_cv(resid_sq, resid_mean_sq)
    leverage_resid_sq_corr = _safe_corr(leverage_normalized, resid_sq)
    beta_cv_bound = leverage_cv * resid_sq_cv if math.isfinite(leverage_cv) and math.isfinite(resid_sq_cv) else float("nan")
    beta_corr_cv_product = (
        leverage_resid_sq_corr * leverage_cv * resid_sq_cv
        if math.isfinite(leverage_resid_sq_corr) and math.isfinite(leverage_cv) and math.isfinite(resid_sq_cv)
        else float("nan")
    )
    beta_corr_cv_product_abs = abs(beta_corr_cv_product) if math.isfinite(beta_corr_cv_product) else float("nan")

    c_eff = float("nan")
    gamma_eff_op = float("nan")
    if math.isfinite(kappa_eff) and math.isfinite(beta_fit) and abs(kappa_eff * beta_fit) > 1e-16:
        c_eff = 1.0 / (kappa_eff * beta_fit)
        gamma_eff_op = safe_op_norm_symmetric(G - c_eff * H2)

    c_fit = best_scalar_fit(G, H2)
    gamma_fit_op = safe_op_norm_symmetric(G - c_fit * H2) if math.isfinite(c_fit) else float("nan")
    c_tilde_fit = best_scalar_fit(H2, G_tilde)
    gamma_tilde_fit_op = (
        safe_op_norm_symmetric(H2 - c_tilde_fit * G_tilde)
        if math.isfinite(c_tilde_fit)
        else float("nan")
    )

    H2_op = safe_op_norm_symmetric(H2)
    G_op = safe_op_norm_symmetric(G)
    G_tilde_op = safe_op_norm_symmetric(G_tilde)
    BBt_op = safe_op_norm_symmetric(BBt)
    M_tilde_op = safe_op_norm_symmetric(M_tilde)

    G_sqrt = safe_psd_sqrt(G)
    s_sqrt_fit = best_scalar_fit(H, G_sqrt)
    sqrt_law_op = safe_op_norm_symmetric(H - s_sqrt_fit * G_sqrt) if math.isfinite(s_sqrt_fit) else float("nan")
    sqrt_law_rel = _relative_residual(sqrt_law_op, safe_op_norm_symmetric(H), s_sqrt_fit, safe_op_norm_symmetric(G_sqrt))

    weighted_residual_rel = _relative_residual(gamma_tilde_eff_op, H2_op, kappa_eff, G_tilde_op)
    raw_residual_rel = _relative_residual(gamma_eff_op, G_op, c_eff, H2_op)
    raw_fit_residual_rel = _relative_residual(gamma_fit_op, G_op, c_fit, H2_op)
    weighted_residual_rel_h2 = _safe_ratio(gamma_tilde_eff_op, H2_op)
    weighted_fit_residual_rel_h2 = _safe_ratio(gamma_tilde_fit_op, H2_op)
    hidden_eff_rel = _relative_residual(delta_eff_op, BBt_op, kappa_eff, M_tilde_op)
    stationarity_rel = _relative_residual(E_stat_op, BBt_op, 1.0, safe_op_norm_symmetric(B_stationary @ B_stationary.T))

    metrics = {
        "loss_total": float(loss_total) if loss_total is not None else float("nan"),
        "resid_rms": float(torch.sqrt(torch.mean(r.square())).item()),
        "resid_mean_sq": float(resid_mean_sq),
        "resid_max_sq": float(resid_max_sq),
        "E_stat_op": float(E_stat_op),
        "delta_stationary_op": float(delta_stationary_op),
        "stationarity_rel": float(stationarity_rel),
        "d_eff": float(d_eff),
        "kappa_eff": float(kappa_eff),
        "delta_eff_op": float(delta_eff_op),
        "hidden_eff_rel": float(hidden_eff_rel),
        "pair_push_op": float(pair_push_op),
        "pair_push_scaled_op": float(pair_push_scaled_op),
        "beta_fit": float(beta_fit),
        "M_bridge_op": float(M_bridge_op),
        "M_bridge_frob": float(M_bridge_frob),
        "M_bridge_rel_frob": float(M_bridge_rel_frob),
        "beta_over_resid_mean_sq": float(beta_over_resid_mean_sq),
        "beta_over_resid_max_sq": float(beta_over_resid_max_sq),
        "beta_rel_mean_sq_error": float(beta_rel_mean_sq_error),
        "leverage_mean": float(leverage_mean),
        "leverage_cv": float(leverage_cv),
        "resid_sq_cv": float(resid_sq_cv),
        "leverage_resid_sq_corr": float(leverage_resid_sq_corr),
        "beta_cv_bound": float(beta_cv_bound),
        "beta_corr_cv_product": float(beta_corr_cv_product),
        "beta_corr_cv_product_abs": float(beta_corr_cv_product_abs),
        "c_eff": float(c_eff),
        "gamma_eff_op": float(gamma_eff_op),
        "gamma_eff_rel": float(raw_residual_rel),
        "gamma_tilde_eff_op": float(gamma_tilde_eff_op),
        "gamma_tilde_eff_rel": float(weighted_residual_rel),
        "gamma_tilde_eff_rel_h2": float(weighted_residual_rel_h2),
        "c_tilde_fit": float(c_tilde_fit),
        "gamma_tilde_fit_op": float(gamma_tilde_fit_op),
        "gamma_tilde_fit_rel_h2": float(weighted_fit_residual_rel_h2),
        "c_fit": float(c_fit),
        "gamma_fit_op": float(gamma_fit_op),
        "gamma_fit_rel": float(raw_fit_residual_rel),
        "s_sqrt_fit": float(s_sqrt_fit),
        "sqrt_law_op": float(sqrt_law_op),
        "sqrt_law_rel": float(sqrt_law_rel),
        "H2_op": float(H2_op),
        "G_op": float(G_op),
        "G_tilde_op": float(G_tilde_op),
        "BBt_op": float(BBt_op),
        "M_tilde_op": float(M_tilde_op),
    }

    metrics.update(
        compute_pair_isotropy_metrics(
            S=S,
            T=T,
            B=B,
            lambda_B=lambda_B,
            n=n,
            d_eff=d_eff,
            E_stat_op=E_stat_op,
            gamma_tilde_eff_op=gamma_tilde_eff_op,
            eps_rel=eps_rel,
        )
    )
    metrics.update(
        compute_pair_spectral_gain_metrics(
            C=Q.T * r.unsqueeze(0),
            X=X,
            d_eff=d_eff,
            eps_rel=eps_rel,
        )
    )
    return metrics
