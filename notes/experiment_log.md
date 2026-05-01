# Experiment log

## Current summary

There are two kinds of runs in the repo right now:

1. the tracked three-family run used for the current paper and demo figures
2. the broader eight-family smoke run used to verify new diagnostics and
   failure-mode wiring

The three-family run is the current source of reported numbers. The eight-family
run is a development smoke test until repeated with final seeds and plotting
choices.

## Current reported run

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

## Main claims supported by the run

### Weighted law

At best-stationarity checkpoints, the residual-weighted law

```text
H^2 approx kappa_eff G_tilde
```

is stable across all three data families. The observed weighted residual stays
below the deterministic theorem bound on average.

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

The current final figure candidates are:

1. weighted law residual and theorem bound ratio by family
2. two-regime trajectory for one representative seed per family
3. pair diagnostic comparison using `A_pair`, pushed pair error, and residual
4. optional matrix visualization for `H^2` and scaled weighted AGOP

The first three are enough for the project narrative.

## Fragment one diagnostics

A fast smoke run now covers the original families plus the structured and
failure-mode families:

1. isotropic Gaussian inputs
2. anisotropic Gaussian inputs
3. low-rank signal plus isotropic noise
4. clustered Gaussian inputs
5. mixture of subspaces
6. rare-region outliers
7. two-region gating
8. XOR-style feature task

The smoke command was:

```text
python -m experiments.run_pair_isotropy --seeds 0 --fast --include-low-rank --include-structured --include-failure-modes --output-root results/tmp/smoke_fragment1
```

The companion figure command was:

```text
python -m experiments.make_final_figures --results-dir results/tmp/smoke_fragment1 --outdir results/tmp/smoke_fragment1_figures
```

This generated ten per-run plots, including beta decomposition, pair
defect-versus-gain, cumulative pair contributions, high-gain closure, residual
damping versus leverage, phase diagram, and the existing beta and trajectory
plots. The final smoke figure folder contained 92 PNG files.

These runs are smoke tests, not final reported numbers. Their purpose is to
verify that the new diagnostics and failure-mode families are wired into the
pipeline.

Before promoting this suite to final evidence, run more seeds, choose the main
figure subset, and copy the selected plots into `paper/figures/` with an updated
manifest and figure guide.
