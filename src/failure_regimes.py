from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import math


@dataclass(frozen=True)
class BetaToyResult:
    name: str
    leverage: List[float]
    residual_sq: List[float]
    beta_fit: float
    mean_residual_sq: float
    beta_over_mean_residual_sq: float
    covariance: float
    correlation: float
    leverage_cv: float
    residual_sq_cv: float
    predicted_relative_error: float
    interpretation: str


@dataclass(frozen=True)
class PairToyResult:
    name: str
    hx_eigenvalues: List[float]
    sigma_sq_eigenvalues: List[float]
    d_eff: float
    defects: List[float]
    global_a_pair: float
    pushed_contributions: List[float]
    pushed_pair_proxy: float
    high_gain_defect: float
    interpretation: str


@dataclass(frozen=True)
class RegimePoint:
    name: str
    beta_covariance_abs: float
    high_gain_pair_defect: float
    beta_size: float
    expected_weighted_law: str
    expected_raw_law: str
    interpretation: str


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _variance(values: List[float]) -> float:
    mean_value = _mean(values)
    return _mean([(value - mean_value) ** 2 for value in values])


def _cv(values: List[float]) -> float:
    mean_value = _mean(values)
    if mean_value == 0:
        return float("nan")
    return math.sqrt(_variance(values)) / abs(mean_value)


def _correlation(left: List[float], right: List[float]) -> float:
    left_var = _variance(left)
    right_var = _variance(right)
    if left_var == 0 or right_var == 0:
        return float("nan")
    left_mean = _mean(left)
    right_mean = _mean(right)
    covariance = _mean([
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    ])
    return covariance / math.sqrt(left_var * right_var)


def beta_toy_result(
    name: str,
    leverage: List[float],
    residual_sq: List[float],
    interpretation: str,
) -> BetaToyResult:
    if len(leverage) != len(residual_sq):
        raise ValueError("leverage and residual_sq must have the same length")
    if not leverage:
        raise ValueError("toy beta examples need at least one sample")
    leverage_mean = _mean(leverage)
    if leverage_mean <= 0:
        raise ValueError("mean leverage must be positive")

    normalized_leverage = [value / leverage_mean for value in leverage]
    mean_residual_sq = _mean(residual_sq)
    beta_fit = _mean([
        ell_i * residual_i for ell_i, residual_i in zip(normalized_leverage, residual_sq)
    ])
    covariance = beta_fit - mean_residual_sq
    ratio = beta_fit / mean_residual_sq if mean_residual_sq > 0 else float("nan")
    corr = _correlation(normalized_leverage, residual_sq)
    leverage_cv = _cv(normalized_leverage)
    residual_cv = _cv(residual_sq)
    predicted = corr * leverage_cv * residual_cv if math.isfinite(corr) else float("nan")

    return BetaToyResult(
        name=name,
        leverage=normalized_leverage,
        residual_sq=residual_sq,
        beta_fit=beta_fit,
        mean_residual_sq=mean_residual_sq,
        beta_over_mean_residual_sq=ratio,
        covariance=covariance,
        correlation=corr,
        leverage_cv=leverage_cv,
        residual_sq_cv=residual_cv,
        predicted_relative_error=predicted,
        interpretation=interpretation,
    )


def high_leverage_hard_sample(n: int = 20, leverage_spike: float = 100.0) -> BetaToyResult:
    leverage = [leverage_spike] + [1.0 for _ in range(n - 1)]
    residual_sq = [1.0] + [0.0 for _ in range(n - 1)]
    return beta_toy_result(
        name="high_leverage_hard_sample",
        leverage=leverage,
        residual_sq=residual_sq,
        interpretation=(
            "A high-leverage sample keeps high residual energy, so beta_fit "
            "overestimates ordinary mean residual energy."
        ),
    )


def low_leverage_hard_sample(n: int = 20, leverage_spike: float = 0.01) -> BetaToyResult:
    leverage = [leverage_spike] + [1.0 for _ in range(n - 1)]
    residual_sq = [1.0] + [0.0 for _ in range(n - 1)]
    return beta_toy_result(
        name="low_leverage_hard_sample",
        leverage=leverage,
        residual_sq=residual_sq,
        interpretation=(
            "A high-residual sample has very low leverage, so beta_fit "
            "underestimates ordinary mean residual energy."
        ),
    )


def beta_success_reference(n: int = 20) -> BetaToyResult:
    leverage = [1.0 + 0.1 * math.sin(index) for index in range(n)]
    residual_sq = [0.2 + 0.05 * math.cos(2 * index) for index in range(n)]
    return beta_toy_result(
        name="weakly_correlated_reference",
        leverage=leverage,
        residual_sq=residual_sq,
        interpretation=(
            "Leverage and residual energy vary but have weak coupling, so "
            "beta_fit remains close to mean residual energy."
        ),
    )


def pair_toy_result(
    name: str,
    hx_eigenvalues: List[float],
    sigma_sq_eigenvalues: List[float],
    interpretation: str,
) -> PairToyResult:
    if len(hx_eigenvalues) != len(sigma_sq_eigenvalues):
        raise ValueError("hx_eigenvalues and sigma_sq_eigenvalues must match")
    weights = [sigma_sq ** 2 for sigma_sq in sigma_sq_eigenvalues]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("at least one sigma_sq eigenvalue must be nonzero")

    d_eff = sum(weight * hx for weight, hx in zip(weights, hx_eigenvalues)) / weight_sum
    defects = [hx - d_eff for hx in hx_eigenvalues]
    global_a_pair = max(abs(defect) for defect in defects)
    pushed = [
        hx * (sigma_sq ** 2) * abs(defect)
        for hx, sigma_sq, defect in zip(hx_eigenvalues, sigma_sq_eigenvalues, defects)
    ]
    pushed_pair_proxy = max(pushed)
    max_gain = max(sigma_sq_eigenvalues)
    high_gain_defect = max(
        abs(defect)
        for defect, sigma_sq in zip(defects, sigma_sq_eigenvalues)
        if sigma_sq >= 0.5 * max_gain
    )

    return PairToyResult(
        name=name,
        hx_eigenvalues=hx_eigenvalues,
        sigma_sq_eigenvalues=sigma_sq_eigenvalues,
        d_eff=d_eff,
        defects=defects,
        global_a_pair=global_a_pair,
        pushed_contributions=pushed,
        pushed_pair_proxy=pushed_pair_proxy,
        high_gain_defect=high_gain_defect,
        interpretation=interpretation,
    )


def high_gain_bad_pair_direction() -> PairToyResult:
    return pair_toy_result(
        name="high_gain_bad_pair_direction",
        hx_eigenvalues=[0.0, 2.0],
        sigma_sq_eigenvalues=[1.0, 1.0],
        interpretation=(
            "The bad scalar-fit direction has high stationarity gain, so the "
            "pushed pair error is large."
        ),
    )


def low_gain_bad_pair_direction(epsilon: float = 0.05) -> PairToyResult:
    return pair_toy_result(
        name="low_gain_bad_pair_direction",
        hx_eigenvalues=[0.0, 1.0],
        sigma_sq_eigenvalues=[epsilon, 1.0],
        interpretation=(
            "The global pair defect is large, but the worst bad direction has "
            "low stationarity gain, so the pushed pair proxy is small."
        ),
    )


def pair_success_reference() -> PairToyResult:
    return pair_toy_result(
        name="near_scalar_high_gain_reference",
        hx_eigenvalues=[0.9, 1.0, 1.1],
        sigma_sq_eigenvalues=[1.0, 0.8, 0.7],
        interpretation=(
            "The high-gain input geometry is close to scalar, so pair closure "
            "is benign."
        ),
    )


def regime_points() -> List[RegimePoint]:
    return [
        RegimePoint(
            name="fully_benign",
            beta_covariance_abs=0.03,
            high_gain_pair_defect=0.05,
            beta_size=0.4,
            expected_weighted_law="holds",
            expected_raw_law="holds",
            interpretation="Both bridge diagnostics are small and beta is not tiny.",
        ),
        RegimePoint(
            name="late_weighted_only",
            beta_covariance_abs=0.03,
            high_gain_pair_defect=0.05,
            beta_size=1e-4,
            expected_weighted_law="holds",
            expected_raw_law="ill-conditioned",
            interpretation="Weighted law remains stable, but raw de-weighting divides by tiny beta.",
        ),
        RegimePoint(
            name="beta_failure",
            beta_covariance_abs=1.5,
            high_gain_pair_defect=0.05,
            beta_size=0.5,
            expected_weighted_law="may hold",
            expected_raw_law="biased",
            interpretation="Leverage and residual energy are strongly coupled.",
        ),
        RegimePoint(
            name="pair_failure",
            beta_covariance_abs=0.04,
            high_gain_pair_defect=1.2,
            beta_size=0.5,
            expected_weighted_law="degrades",
            expected_raw_law="fails",
            interpretation="Bad pair directions are visible to stationarity gain.",
        ),
        RegimePoint(
            name="conservative_pair_diagnostic",
            beta_covariance_abs=0.05,
            high_gain_pair_defect=0.08,
            beta_size=0.4,
            expected_weighted_law="holds",
            expected_raw_law="depends on beta",
            interpretation="Global pair defect can be large while high-gain pair defect is small.",
        ),
    ]


def run_all_toy_regimes() -> Dict[str, object]:
    beta_examples = [
        beta_success_reference(),
        high_leverage_hard_sample(),
        low_leverage_hard_sample(),
    ]
    pair_examples = [
        pair_success_reference(),
        high_gain_bad_pair_direction(),
        low_gain_bad_pair_direction(),
    ]
    return {
        "beta_examples": [asdict(example) for example in beta_examples],
        "pair_examples": [asdict(example) for example in pair_examples],
        "regime_points": [asdict(point) for point in regime_points()],
    }
