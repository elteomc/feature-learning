# Weighted AGOP Law Explorer

This is a small static demo for the weighted AGOP two-regime story.

## Run locally

From the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/apps/weighted-law-explorer/
```

The app has no build step and no JavaScript dependencies. It reads the tracked
figures from `paper/figures/`.

## What it shows

- The stable late-training law is the residual-weighted AGOP relation.
- `beta_fit` tracks mean residual squared across the logged checkpoints.
- The deterministic bound is conservative across the tested synthetic families.
- `A_pair` is a conservative support-normalized diagnostic, while pushed pair
  error is closer to the observed weighted law.
