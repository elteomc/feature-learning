# Feature Learning Two Regimes

Project on weighted AGOP laws, effective-dimension corrections, and the two-regime picture:
- late / near-stationary weighted law
- intermediate / raw-conditioned law

## Main idea

The project studies when a two-layer network learns a feature matrix

```text
H = B^T B
```

that is related to AGOP-style gradient geometry.

The main conclusion so far is that the most stable late-training relation is
not the raw AGOP law. It is the residual-weighted law

```text
H^2 approx kappa_eff * G_tilde
```

where `G_tilde` is the residual-weighted AGOP. The raw AGOP relation is better
viewed as an intermediate-regime relation that becomes ill-conditioned when the
residual-weight bridge scalar `beta_fit` collapses near interpolation.

## Structure

- `legacy/`: old single-file scripts v1-v7 (when I was testing my proposal)
- `src/`: reusable training + diagnostics code
- `experiments/`: runnable scripts
- `configs/`: experiment configs
- `notes/`: theorem sketches, todo, experiment log
- `paper/`: writeup and figures
- `apps/`: peer-facing demos

## Current focus

1. weighted deterministic proposition
2. pair-isotropy diagnostic
3. multi-seed confirmation
4. presentation demo

## What is proved versus tested

The deterministic part is in `notes/theorem_sketch.md` and
`paper/section_weighted_law.tex`.

The proved algebra includes:

- exact AGOP and residual-weighted AGOP identities
- the stationarity identity for `B`
- the effective-dimension reduction from pair closure to the weighted square law
- the beta identity showing `beta_fit` is a hidden-gradient-leverage-weighted
  residual average
- high-gain pair closure as a conditional theorem

The open mathematical part is training geometry:

- why trained networks should satisfy pair closure on high-gain directions
- why hidden-gradient leverage should decorrelate from residual energy
- when the NTK actually gives leverage-sensitive residual damping

The experiments test these bridge assumptions and diagnostics on synthetic
families.

## Quickstart

Run a fast smoke experiment:

```bash
python -m experiments.run_pair_isotropy --seeds 0 --fast --include-low-rank --include-structured --include-failure-modes --output-root results/tmp/smoke_fragment1
```

Build figures from a result directory:

```bash
python -m experiments.make_final_figures --results-dir results/tmp/smoke_fragment1 --outdir results/tmp/smoke_fragment1_figures
```

Run the static demo:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/apps/weighted-law-explorer/
```

## Reading Order

For the math story, start with `paper/section_weighted_law.tex`, then read
`notes/theorem_sketch.md` for the fuller theorem sketches.

For the experiment story, start with `notes/experiment_log.md`, then
`experiments/README.md`, then `paper/figures/README.md`.

For a visual walkthrough, open `apps/weighted-law-explorer/`.
