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

## Views

`Live lab` is the default. It uses the left control bar to generate qualitative
sweeps for residual energy, beta correlation, and pair gain.

`Reported figures` shows static experiment artifacts from the repo. This is the
place for paper-style benchmark plots and archived evidence.
