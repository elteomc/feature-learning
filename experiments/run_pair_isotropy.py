from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from src.train_two_layer import ExperimentConfig, ensure_dir, train_one


def parse_seeds(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_configs(
    seeds: Iterable[int],
    output_root: str,
    fast: bool,
    include_low_rank: bool,
    include_structured: bool,
    include_failure_modes: bool,
) -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    for seed in seeds:
        for anisotropic in [False, True]:
            prefix = "anisotropic" if anisotropic else "isotropic"
            cfg = ExperimentConfig(
                name=f"{prefix}_seed_{seed}",
                seed=seed,
                data_family="gaussian",
                anisotropic=anisotropic,
                noise_std=0.0,
                spectrum_min=1e-2,
                beta_threshold=1e-5,
                output_root=output_root,
            )
            if fast:
                cfg.sgd_steps = 400
                cfg.checkpoint_every = 100
                cfg.lbfgs_outer_steps = 10
                cfg.lbfgs_log_every = 5
            configs.append(cfg)
        if include_low_rank:
            cfg = ExperimentConfig(
                name=f"low_rank_signal_seed_{seed}",
                seed=seed,
                data_family="low_rank_signal",
                anisotropic=False,
                signal_rank=8,
                signal_noise_std=0.5,
                noise_std=0.0,
                beta_threshold=1e-5,
                output_root=output_root,
            )
            if fast:
                cfg.sgd_steps = 400
                cfg.checkpoint_every = 100
                cfg.lbfgs_outer_steps = 10
                cfg.lbfgs_log_every = 5
            configs.append(cfg)
        if include_structured:
            for data_family in ["clustered_gaussian", "mixture_subspaces"]:
                cfg = ExperimentConfig(
                    name=f"{data_family}_seed_{seed}",
                    seed=seed,
                    data_family=data_family,
                    anisotropic=False,
                    signal_rank=8,
                    signal_noise_std=0.35,
                    noise_std=0.0,
                    beta_threshold=1e-5,
                    output_root=output_root,
                )
                if fast:
                    cfg.sgd_steps = 400
                    cfg.checkpoint_every = 100
                    cfg.lbfgs_outer_steps = 10
                    cfg.lbfgs_log_every = 5
                configs.append(cfg)
        if include_failure_modes:
            for data_family in ["rare_region_outliers", "two_region_gating", "xor_feature"]:
                cfg = ExperimentConfig(
                    name=f"{data_family}_seed_{seed}",
                    seed=seed,
                    data_family=data_family,
                    anisotropic=False,
                    signal_rank=8,
                    signal_noise_std=0.35,
                    rare_fraction=0.1,
                    rare_shift=4.0,
                    rare_label_scale=3.0,
                    noise_std=0.0,
                    beta_threshold=1e-5,
                    output_root=output_root,
                )
                if fast:
                    cfg.sgd_steps = 400
                    cfg.checkpoint_every = 100
                    cfg.lbfgs_outer_steps = 10
                    cfg.lbfgs_log_every = 5
                configs.append(cfg)
    return configs


def summarize_metric(rows: List[Dict[str, object]], block: str, metric: str) -> Dict[str, float]:
    values = []
    for row in rows:
        metrics = row.get(block)
        if isinstance(metrics, dict):
            value = metrics.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_group(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        "all_checkpoints": {
            "beta_over_resid_mean_sq": summarize_history_metric(rows, "beta_over_resid_mean_sq"),
            "beta_rel_mean_sq_error": summarize_history_metric(rows, "beta_rel_mean_sq_error"),
            "beta_cv_bound": summarize_history_metric(rows, "beta_cv_bound"),
            "leverage_cv": summarize_history_metric(rows, "leverage_cv"),
            "resid_sq_cv": summarize_history_metric(rows, "resid_sq_cv"),
            "leverage_resid_sq_corr_abs": summarize_history_metric_abs(rows, "leverage_resid_sq_corr"),
            "beta_corr_cv_product": summarize_history_metric(rows, "beta_corr_cv_product"),
            "beta_corr_cv_product_abs": summarize_history_metric(rows, "beta_corr_cv_product_abs"),
            "beta_over_resid_max_sq": summarize_history_metric(rows, "beta_over_resid_max_sq"),
            "leverage_damping_corr": summarize_history_metric(rows, "leverage_damping_corr"),
            "mean_log_resid_sq_decay": summarize_history_metric(rows, "mean_log_resid_sq_decay"),
            "M_bridge_op": summarize_history_metric(rows, "M_bridge_op"),
            "M_bridge_rel_frob": summarize_history_metric(rows, "M_bridge_rel_frob"),
            "theorem_bound_ratio": summarize_history_metric(rows, "theorem_bound_ratio"),
            "gamma_tilde_eff_rel": summarize_history_metric(rows, "gamma_tilde_eff_rel"),
            "gamma_tilde_eff_rel_h2": summarize_history_metric(rows, "gamma_tilde_eff_rel_h2"),
            "pair_push_scaled_op": summarize_history_metric(rows, "pair_push_scaled_op"),
            "pair_top_defect_gain": summarize_history_metric(rows, "pair_top_defect_gain"),
            "pair_gain_defect_corr_abs": summarize_history_metric(rows, "pair_gain_defect_corr_abs"),
            "pair_weighted_contribution_max": summarize_history_metric(rows, "pair_weighted_contribution_max"),
            "pair_high_gain_closure_op": summarize_history_metric(rows, "pair_high_gain_closure_op"),
            "pair_low_gain_op": summarize_history_metric(rows, "pair_low_gain_op"),
            "pair_damping_bound_proxy": summarize_history_metric(rows, "pair_damping_bound_proxy"),
        },
        "best_stationarity": {
            "gamma_tilde_eff_op": summarize_metric(rows, "best_metrics_by_stationarity", "gamma_tilde_eff_op"),
            "gamma_tilde_eff_rel": summarize_metric(rows, "best_metrics_by_stationarity", "gamma_tilde_eff_rel"),
            "gamma_tilde_eff_rel_h2": summarize_metric(rows, "best_metrics_by_stationarity", "gamma_tilde_eff_rel_h2"),
            "gamma_tilde_fit_rel_h2": summarize_metric(rows, "best_metrics_by_stationarity", "gamma_tilde_fit_rel_h2"),
            "A_pair_op": summarize_metric(rows, "best_metrics_by_stationarity", "A_pair_op"),
            "pair_fit_op": summarize_metric(rows, "best_metrics_by_stationarity", "pair_fit_op"),
            "pair_push_scaled_op": summarize_metric(rows, "best_metrics_by_stationarity", "pair_push_scaled_op"),
            "pair_top_abs_defect": summarize_metric(rows, "best_metrics_by_stationarity", "pair_top_abs_defect"),
            "pair_top_defect_gain": summarize_metric(rows, "best_metrics_by_stationarity", "pair_top_defect_gain"),
            "pair_weighted_contribution_max": summarize_metric(rows, "best_metrics_by_stationarity", "pair_weighted_contribution_max"),
            "pair_gain_defect_corr_abs": summarize_metric(rows, "best_metrics_by_stationarity", "pair_gain_defect_corr_abs"),
            "pair_high_gain_closure_op": summarize_metric(rows, "best_metrics_by_stationarity", "pair_high_gain_closure_op"),
            "pair_low_gain_op": summarize_metric(rows, "best_metrics_by_stationarity", "pair_low_gain_op"),
            "pair_damping_bound_proxy": summarize_metric(rows, "best_metrics_by_stationarity", "pair_damping_bound_proxy"),
            "theorem_bound_ratio": summarize_metric(rows, "best_metrics_by_stationarity", "theorem_bound_ratio"),
            "delta_stationary_op": summarize_metric(rows, "best_metrics_by_stationarity", "delta_stationary_op"),
            "beta_over_resid_mean_sq": summarize_metric(rows, "best_metrics_by_stationarity", "beta_over_resid_mean_sq"),
            "beta_rel_mean_sq_error": summarize_metric(rows, "best_metrics_by_stationarity", "beta_rel_mean_sq_error"),
            "beta_cv_bound": summarize_metric(rows, "best_metrics_by_stationarity", "beta_cv_bound"),
            "leverage_cv": summarize_metric(rows, "best_metrics_by_stationarity", "leverage_cv"),
            "resid_sq_cv": summarize_metric(rows, "best_metrics_by_stationarity", "resid_sq_cv"),
            "leverage_resid_sq_corr": summarize_metric(rows, "best_metrics_by_stationarity", "leverage_resid_sq_corr"),
            "beta_corr_cv_product": summarize_metric(rows, "best_metrics_by_stationarity", "beta_corr_cv_product"),
            "beta_corr_cv_product_abs": summarize_metric(rows, "best_metrics_by_stationarity", "beta_corr_cv_product_abs"),
            "beta_over_resid_max_sq": summarize_metric(rows, "best_metrics_by_stationarity", "beta_over_resid_max_sq"),
            "M_bridge_op": summarize_metric(rows, "best_metrics_by_stationarity", "M_bridge_op"),
            "M_bridge_rel_frob": summarize_metric(rows, "best_metrics_by_stationarity", "M_bridge_rel_frob"),
        },
        "best_raw_conditioning": {
            "beta_fit": summarize_metric(rows, "best_metrics_by_raw_conditioning", "beta_fit"),
            "resid_mean_sq": summarize_metric(rows, "best_metrics_by_raw_conditioning", "resid_mean_sq"),
            "resid_max_sq": summarize_metric(rows, "best_metrics_by_raw_conditioning", "resid_max_sq"),
            "beta_over_resid_mean_sq": summarize_metric(rows, "best_metrics_by_raw_conditioning", "beta_over_resid_mean_sq"),
            "beta_rel_mean_sq_error": summarize_metric(rows, "best_metrics_by_raw_conditioning", "beta_rel_mean_sq_error"),
            "beta_cv_bound": summarize_metric(rows, "best_metrics_by_raw_conditioning", "beta_cv_bound"),
            "leverage_cv": summarize_metric(rows, "best_metrics_by_raw_conditioning", "leverage_cv"),
            "resid_sq_cv": summarize_metric(rows, "best_metrics_by_raw_conditioning", "resid_sq_cv"),
            "leverage_resid_sq_corr": summarize_metric(rows, "best_metrics_by_raw_conditioning", "leverage_resid_sq_corr"),
            "beta_corr_cv_product": summarize_metric(rows, "best_metrics_by_raw_conditioning", "beta_corr_cv_product"),
            "beta_corr_cv_product_abs": summarize_metric(rows, "best_metrics_by_raw_conditioning", "beta_corr_cv_product_abs"),
            "beta_over_resid_max_sq": summarize_metric(rows, "best_metrics_by_raw_conditioning", "beta_over_resid_max_sq"),
            "M_bridge_op": summarize_metric(rows, "best_metrics_by_raw_conditioning", "M_bridge_op"),
            "M_bridge_rel_frob": summarize_metric(rows, "best_metrics_by_raw_conditioning", "M_bridge_rel_frob"),
            "gamma_eff_op": summarize_metric(rows, "best_metrics_by_raw_conditioning", "gamma_eff_op"),
            "gamma_fit_op": summarize_metric(rows, "best_metrics_by_raw_conditioning", "gamma_fit_op"),
            "gamma_eff_rel": summarize_metric(rows, "best_metrics_by_raw_conditioning", "gamma_eff_rel"),
        },
        "crossover": {
            "beta_fit": summarize_metric(rows, "crossover_metrics", "beta_fit"),
            "raw_quality": summarize_metric(rows, "crossover_metrics", "raw_quality"),
            "weighted_quality": summarize_metric(rows, "crossover_metrics", "weighted_quality"),
        },
    }


def summarize_history_metric(rows: List[Dict[str, object]], metric: str) -> Dict[str, float]:
    values = []
    for row in rows:
        history = row.get("_history", [])
        if isinstance(history, list):
            for checkpoint in history:
                if isinstance(checkpoint, dict):
                    value = checkpoint.get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        values.append(float(value))
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_history_metric_abs(rows: List[Dict[str, object]], metric: str) -> Dict[str, float]:
    values = []
    for row in rows:
        history = row.get("_history", [])
        if isinstance(history, list):
            for checkpoint in history:
                if isinstance(checkpoint, dict):
                    value = checkpoint.get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        values.append(abs(float(value)))
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_summary(results: List[Dict[str, object]]) -> Dict[str, object]:
    groups = {
        "isotropic": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "gaussian" and not row["config"]["anisotropic"]
        ],
        "anisotropic": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "gaussian" and row["config"]["anisotropic"]
        ],
        "low_rank_signal": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "low_rank_signal"
        ],
        "anisotropic_low_rank": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "anisotropic_low_rank"
        ],
        "clustered_gaussian": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "clustered_gaussian"
        ],
        "mixture_subspaces": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "mixture_subspaces"
        ],
        "rare_region_outliers": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "rare_region_outliers"
        ],
        "two_region_gating": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "two_region_gating"
        ],
        "xor_feature": [
            row for row in results
            if row["config"].get("data_family", "gaussian") == "xor_feature"
        ],
    }
    runs = [{key: value for key, value in row.items() if key != "_history"} for row in results]
    return {
        "runs": runs,
        "groups": {name: summarize_group(rows) for name, rows in groups.items() if rows},
    }


def load_existing_results(configs: List[ExperimentConfig]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        history_path = Path(cfg.output_root) / cfg.name / "history.json"
        with open(history_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        history = payload["history"]
        rows.append(
            {
                "name": cfg.name,
                "output_dir": str(history_path.parent),
                "config": payload["config"],
                "final_metrics": history[-1],
                "best_metrics_by_stationarity": payload["best_metrics_by_stationarity"],
                "best_metrics_by_weighted_law": payload["best_metrics_by_weighted_law"],
                "best_metrics_by_raw_conditioning": payload["best_metrics_by_raw_conditioning"],
                "crossover_metrics": payload["crossover_metrics"],
                "_history": history,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pair-isotropy robustness checks.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated random seeds.")
    parser.add_argument("--output-root", default="results/runs/pair_isotropy_multiseed")
    parser.add_argument("--fast", action="store_true", help="Use a short smoke-test schedule.")
    parser.add_argument("--include-low-rank", action="store_true", help="Include low-rank signal plus isotropic noise.")
    parser.add_argument("--include-structured", action="store_true", help="Include clustered and mixture-of-subspaces families.")
    parser.add_argument("--include-failure-modes", action="store_true", help="Include rare-region, two-region, and XOR-style failure families.")
    parser.add_argument("--summary-only", action="store_true", help="Rebuild summaries from existing run histories.")
    args = parser.parse_args()

    configs = build_configs(
        parse_seeds(args.seeds),
        args.output_root,
        args.fast,
        args.include_low_rank,
        args.include_structured,
        args.include_failure_modes,
    )
    if args.summary_only:
        all_results = load_existing_results(configs)
    else:
        all_results: List[Dict[str, object]] = []

        for cfg in configs:
            result = train_one(cfg)
            all_results.append(
                {
                    "name": cfg.name,
                    "output_dir": result["output_dir"],
                    "config": result["config"],
                    "final_metrics": result["final_metrics"],
                    "best_metrics_by_stationarity": result["best_metrics_by_stationarity"],
                    "best_metrics_by_weighted_law": result["best_metrics_by_weighted_law"],
                    "best_metrics_by_raw_conditioning": result["best_metrics_by_raw_conditioning"],
                    "crossover_metrics": result["crossover_metrics"],
                    "_history": result["history"],
                }
            )

    summary = build_summary(all_results)
    summary_path = Path(args.output_root) / "summary.json"
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    compact_path = Path(args.output_root) / "compact_summary.json"
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(summary["groups"], f, indent=2)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved compact summary to: {compact_path}")
    for group, values in summary["groups"].items():
        weighted = values["best_stationarity"]["gamma_tilde_eff_op"]["mean"]
        bound_ratio = values["best_stationarity"]["theorem_bound_ratio"]["mean"]
        print(f"- {group}: mean gamma_tilde={weighted:.3e}, mean bound_ratio={bound_ratio:.3e}")


if __name__ == "__main__":
    main()
