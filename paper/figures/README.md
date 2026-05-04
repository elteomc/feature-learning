# Figure Guide

This folder contains tracked figure candidates for the paper and static demo.
The manifest is in `manifest.txt`.

## Summary Figures

`weighted_residual_by_family.png`

Shows the residual of the weighted law at best-stationarity checkpoints. This is
the main evidence plot for the late-training relation
`H^2 approx kappa_eff * G_tilde`.

`theorem_bound_ratio_by_family.png`

Shows the observed weighted-law residual divided by the deterministic theorem
bound. Values below one mean the theorem bound covers the observed residual.
The bound is expected to be conservative.

`beta_over_residual_energy_by_family.png`

Shows whether `beta_fit` tracks mean residual squared across checkpoints. Values
near one support the interpretation that the bridge from weighted AGOP to raw
AGOP is controlled by training residual energy.

`pushed_pair_error_by_family.png`

Shows the pair error after pushing it through the matrices that actually enter
the weighted law. This is more directly relevant than the raw support-normalized
pair diagnostic.

`symmetric_relative_error_by_family.png`

Shows the symmetric relative weighted-law error. This is a secondary
normalization that is less unstable than normalizing only by `H^2`.

## Representative Trajectories

`isotropic_two_regime_trajectory.png`

Representative trajectory for the isotropic family. Use this to show the
separation between the late weighted law and the intermediate raw-conditioned
law.

`anisotropic_two_regime_trajectory.png`

Representative trajectory for anisotropic inputs. Use this to show that the
effective-dimension correction still gives a coherent weighted-law story outside
the isotropic baseline.

`low_rank_signal_two_regime_trajectory.png`

Representative trajectory for low-rank signal plus isotropic noise. Use this to
show the same diagnostics in a structured signal family.

## Beta Figures

`isotropic_beta_collapse.png`

Shows beta collapse for the isotropic representative run.

`anisotropic_beta_collapse.png`

Shows beta collapse for the anisotropic representative run.

`low_rank_signal_beta_collapse.png`

Shows beta collapse for the low-rank signal representative run.

The beta-collapse figures support the point that converting the stable weighted
law into a raw AGOP law becomes ill-conditioned near interpolation.

## Current Figure Status

The tracked figures in this folder are suitable for the current paper and demo
story. The larger smoke-run figure set is useful for development, but it should
not be treated as final evidence until those runs are repeated with broader
seeds and final plotting choices.

Failure-mode toy figures live in `failure_modes/`. They are deterministic
algebraic checks for the taxonomy, not trained-network evidence.

For the latest smoke commands and run status, see `notes/experiment_log.md`.
