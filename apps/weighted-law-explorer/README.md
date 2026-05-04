# Weighted AGOP Law Explorer

This is a small static demo for the weighted AGOP two-regime story. The main
view is a live dashboard with generated sweeps, plots, tables, and explanatory
cards. The reported-figures view keeps the tracked paper figures available
without making static PNGs the center of the interaction.

## Run locally

From the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/apps/weighted-law-explorer/
```

The app has no build step and no JavaScript dependencies. The live lab generates
its plots in the browser with SVG. The reported-figures view reads tracked
figures from `paper/figures/`.

## What it shows

- The stable late-training law is the residual-weighted AGOP relation.
- `beta_fit` tracks mean residual squared across the logged checkpoints.
- The deterministic bound is conservative across the tested synthetic families.
- `A_pair` is a conservative support-normalized diagnostic, while pushed pair
  error is closer to the observed weighted law.

## Real data files

The dashboard reads two precomputed JSON files at startup. Both live under
`data/` and are small enough to ship statically.

`data/trajectories.json` is a compact slice of the per-checkpoint history
files in `results/runs/...`. It contains `step`, `loss_total`, `resid_rms`,
`beta_fit`, `gamma_tilde_eff_rel_h2`, `pair_push_scaled_op`,
`theorem_bound_ratio`, and `var_y` for one representative seed per family.
The `var_y` field is what lets the demo plot R-squared as
`1 - resid_mean_sq / var_y` without storing R-squared per checkpoint.

```bash
python apps/weighted-law-explorer/build_trajectories.py
```

`data/matrix_snapshots.json` reruns SGD for one seed per family and captures
three snapshots (init, mid, late). Each snapshot stores the student feature
gram matrix `B B^T`, the alignment matrix `B B_*^T`, and the top eigenvalues
of `H squared` and `kappa_eff times G_tilde`. These power the feature
alignment heatmap and the spectrum overlay.

```bash
python apps/weighted-law-explorer/build_matrix_snapshots.py
```

Both scripts depend on the project source under `src/`, so run them from the
repository root with `PYTHONPATH=.` if needed.

## Headless smoke test

A small JSDOM-based smoke test exercises every control and asserts that the
page renders without console errors:

```bash
npm install --no-save jsdom@22
node apps/weighted-law-explorer/__test__/smoke.mjs
```

The test loads the page, fires `input`/`change` events on every slider for
every sweep, switches the family, the trajectory metric, the theme, and the
view, and verifies that the live plot, trajectory plot, regime locator, and
reported figure all update.

## Views

`Live lab` is the default. It uses the left control bar to generate qualitative
sweeps for residual energy, beta correlation, and pair gain.

`Reported figures` shows static experiment artifacts from the repo. This is the
place for paper-style benchmark plots and archived evidence.
