# Presentation slides

`slides.tex` is the talk deck for the weighted AGOP two-regime project (14
frames, `beamer`, no external theme dependencies beyond `tcolorbox` and
`tikz`).

## Build

From this directory:

```bash
python ../../paper/slides/make_slide_figures.py --seeds 0,1,2,3,4   # writes figures/*.pdf
pdflatex slides.tex
```

(Run `make_slide_figures.py` from the repository root, or pass
`--positive-root` / `--adversarial-root` / `--seeds` if your run directories
differ.)

## Figures

`make_slide_figures.py` reads per-run `history.json` files and writes vector
PDFs into `figures/`:

- `two_regime_isotropic_seed_0.pdf`  weighted vs raw law error, beta, and mean
  residual energy along training (the "phenomenon" slide).
- `weighted_law_residual.pdf`  observed weighted residual vs the deterministic
  theorem bound, by positive family.
- `beta_tracking.pdf`  `beta_fit / mean(r^2)` by positive family.
- `pushed_pair_error.pdf`  global `A_pair` vs pushed pair error, by positive
  family (shows the global diagnostic is conservative).
- `phase_diagram.pdf`  every family (positive plus adversarial) on the bridge
  map. `x` is the beta-link error `|beta_fit / mean(r^2) - 1|` at the best
  raw-AGOP checkpoint; `y` is the pushed pair error (the weighted-law residual
  the theorem controls) at the best-stationarity checkpoint. Markers are family
  medians over the seeds, whiskers span them.

Default inputs (5 seeds each):

- positive families (isotropic, anisotropic, low-rank signal):
  `results/runs/phase_diagram_positive/` produced by
  `python -m experiments.run_pair_isotropy --seeds 0,1,2,3,4 --include-low-rank --output-root results/runs/phase_diagram_positive`
- adversarial families (rare hard cluster, rare easy cluster, two-region gating,
  mixture of subspaces, strongly anisotropic):
  `results/runs/phase_diagram_adversarial/trained/` produced by
  `python -m experiments.run_failure_modes --trained --seeds 0,1,2,3,4 --output-root results/runs/phase_diagram_adversarial --figure-dir results/runs/phase_diagram_adversarial/fig`

The adversarial configs live in `experiments/run_failure_modes.py`
(`trained_configs`). `strong_anisotropic` uses a Gaussian input covariance
spectrum from `1` down to `1e-4`; `mixture_subspaces_stress` uses rank 3 and
ambient noise std 0.05 so the inputs concentrate on a small union of subspaces.

## What the runs show (so you do not have to re-read the plots)

- The weighted law `H^2 ~ kappa_eff * G_tilde` holds (residual roughly `1e-4` to
  `1e-3`) for the positive families, and also for strongly anisotropic inputs
  and mixture-of-subspaces. It degrades by orders of magnitude only for the rare
  hard cluster, and somewhat for two-region gating.
- The global pair diagnostic `A_pair` stays large (roughly 50 to 360)
  essentially everywhere while the pushed pair error stays tiny everywhere
  except the rare hard cluster: the conservative-diagnostic pattern, broadly.
- The leverage-residual coupling (the thing that bends the beta link) is
  elevated for the rare-cluster, gating, and mixture families.

## Caveats baked into the deck

- The trained phase diagram (slide 13) is a guiding map, not a final phase
  diagram: 5 seeds per family with wide per-seed spread, and the adversarial
  constructions are not yet a deliberate sweep over leverage-residual
  correlation and high-gain anisotropy. The deck says this explicitly.
- The failure-mode taxonomy table (slide 12) is deterministic algebra plus a
  closed-form `n=40` sanity check. Slide 13 is the trained companion.
