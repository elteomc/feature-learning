# Experiments

This folder contains the runnable experiment entry points for the weighted AGOP
two-regime project.

## Main Scripts

`run_pair_isotropy.py` trains two-layer models, records checkpoint diagnostics,
and writes per-run plots plus JSON summaries.

`make_final_figures.py` reads a compact summary and copies or creates figure
candidates for the paper and demo.

## Fast Smoke Run

Use this when checking that all families, metrics, and plots are wired together:

```bash
python -m experiments.run_pair_isotropy --seeds 0 --fast --include-low-rank --include-structured --include-failure-modes --output-root results/tmp/smoke_fragment1
```

Then build figures:

```bash
python -m experiments.make_final_figures --results-dir results/tmp/smoke_fragment1 --outdir results/tmp/smoke_fragment1_figures
```

The smoke run is not the final source of reported numbers. It is meant to verify
that the full diagnostic pipeline runs end to end.

## Data Families

The default run includes:

- `isotropic`: Gaussian inputs with identity covariance
- `anisotropic`: Gaussian inputs with a decaying covariance spectrum

`--include-low-rank` adds:

- `low_rank_signal`: low-rank signal plus isotropic noise

`--include-structured` adds:

- `clustered_gaussian`: clustered Gaussian inputs
- `mixture_subspaces`: samples drawn from multiple low-dimensional subspaces

`--include-failure-modes` adds:

- `rare_region_outliers`: rare shifted inputs with amplified labels
- `two_region_gating`: a two-region task designed to stress residual weighting
- `xor_feature`: an XOR-style feature task

## Outputs

Each run directory contains:

- `history.json`: checkpoint metrics over training
- `summary.json`: selected best checkpoints and summary metrics
- per-run plots such as `phase_diagram.png`, `beta_collapse.png`,
  `beta_decomposition_identity.png`, `high_gain_closure.png`,
  `pair_defect_gain_scatter.png`, `pair_direction_cumulative.png`, and
  `residual_damping_vs_leverage.png`

The parent output directory also contains:

- `all_summaries.json`: all run summaries
- `compact_summary.json`: grouped metrics by data family

## Diagnostics To Read First

For the weighted law:

- `gamma_tilde_eff_op`
- `gamma_tilde_eff_rel`
- `theorem_bound_ratio`

For beta tracking:

- `beta_over_resid_mean_sq`
- `beta_corr_cv_product_abs`
- `leverage_cv`
- `resid_sq_cv`
- `leverage_resid_sq_corr`

For pair compression:

- `A_pair_op`
- `pair_push_scaled_op`
- `pair_high_gain_closure_op`
- `pair_low_gain_op`
- `pair_damping_bound_proxy`
- `pair_gain_defect_corr_abs`

## Interpreting The Runs

The core expected pattern is:

- the residual-weighted law is most stable near stationarity
- `beta_fit` often tracks mean residual energy
- the raw AGOP bridge becomes ill-conditioned as `beta_fit` approaches zero
- global pair error can be conservative
- high-gain pair diagnostics are closer to the error that enters the weighted law

The structured and failure-mode families are included to test where these
patterns weaken.
