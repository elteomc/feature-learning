# Experiment log

## Current summary

The current experiment suite tests the weighted AGOP law on three synthetic
teacher-student families:

1. isotropic Gaussian inputs
2. anisotropic Gaussian inputs
3. low-rank signal plus isotropic noise

The main run is stored in:

```text
results/runs/pair_isotropy_with_low_rank/
```

The compact summary is:

```text
results/runs/pair_isotropy_with_low_rank/compact_summary.json
```

The final figure export script is:

```text
experiments/make_final_figures.py
```

The selected figures are written to:

```text
paper/figures/
```

## Main claims supported by the run

The results support four presentation-level claims:

1. The stable late-training object is the residual-weighted AGOP law.
2. The raw AGOP law is an intermediate-regime relation because the beta bridge
   collapses near interpolation.
3. The theorem bound is conservative across isotropic, anisotropic, and
   low-rank signal data.
4. The pushed-forward pair error is the useful diagnostic for the final law,
   while `A_pair` is a conservative support-normalized quantity.

### Weighted law

At best-stationarity checkpoints, the residual-weighted law

```text
H^2 approx kappa_eff G_tilde
```

is stable across all three data families. The observed weighted residual stays
below the deterministic theorem bound on average.

| family | mean weighted residual | mean theorem bound ratio |
| --- | ---: | ---: |
| isotropic | 3.58e-03 | 0.147 |
| anisotropic | 1.90e-03 | 0.0896 |
| low_rank_signal | 6.36e-04 | 0.0987 |

### Beta collapse

Across all logged checkpoints, `beta_fit` closely tracks the empirical residual
energy `mean(r^2)`:

| family | mean beta / mean(r^2) | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| isotropic | 0.993 | 0.042 | 0.847 | 1.059 |
| anisotropic | 1.002 | 0.015 | 0.968 | 1.050 |
| low_rank_signal | 1.015 | 0.050 | 0.893 | 1.181 |

This supports the simplified empirical statement:

```text
beta_fit is essentially the training residual energy in these experiments.
```

The lemma in `notes/theorem_sketch.md` explains why `beta_fit` must collapse
near interpolation. The new empirical point is that the hidden-gradient kernel
does not strongly bias the weighted average away from ordinary mean residual
energy in these runs.

### Pair diagnostics

The raw support-normalized `A_pair` can be large, but the pushed-forward pair
error is small at best-stationarity checkpoints:

| family | mean A_pair | mean pair_push_scaled_op | mean theorem bound ratio |
| --- | ---: | ---: | ---: |
| isotropic | 260.947 | 5.59e-05 | 0.147 |
| anisotropic | 76.103 | 1.11e-05 | 0.0896 |
| low_rank_signal | 97.771 | 2.51e-06 | 0.0987 |

This supports treating `A_pair` as a conservative worst-direction diagnostic,
while using pushed pair error as the more direct diagnostic for the weighted law.

### Relative errors

The symmetric relative weighted error is stable enough to report as a secondary
diagnostic:

| family | best-stationarity symmetric relative mean | std |
| --- | ---: | ---: |
| isotropic | 0.288 | 0.132 |
| anisotropic | 0.273 | 0.177 |
| low_rank_signal | 0.340 | 0.199 |

The `H^2`-only normalization can be unstable across all checkpoints, so it
should not be central in the writeup.

## Final figure set

### Main figures

Use these as the main figures for a report or presentation:

1. `weighted_residual_by_family.png`
   Shows that the weighted residual is small at best-stationarity checkpoints.

2. `theorem_bound_ratio_by_family.png`
   Shows that the deterministic theorem bound is conservative on average.

3. `beta_over_residual_energy_by_family.png`
   Shows that `beta_fit` tracks mean residual squared across all checkpoints.

4. `pushed_pair_error_by_family.png`
   Shows that the pushed-forward pair error is small even when `A_pair` itself
   can be large.

5. One representative `two_regime_trajectory` plot.
   This is the best explanatory figure for the two-regime story.

### Backup figures

Keep these as appendix or demo material:

1. `symmetric_relative_error_by_family.png`
   Useful for honest reporting of relative errors, but not the core claim.

2. Per-family `beta_collapse` plots.
   Useful if someone asks whether beta collapse is visible run by run.

3. The remaining per-family `two_regime_trajectory` plots.
   Useful for robustness, but one representative trajectory is enough in the
   main narrative.

## Caveat language

Use careful wording for two points:

- Do not say `A_pair` is small. Say it is a conservative support-normalized
  diagnostic, and that the pushed-forward pair error is small in the observed
  runs.
- Do not make the `H^2`-only relative normalization central. Prefer the
  symmetric relative normalization, and describe relative errors as a secondary
  diagnostic.

## Next polish target

The remaining writeup task is to decide exactly which of the main figures should
appear in the final report, then reference them from `paper/section_weighted_law.tex`.
