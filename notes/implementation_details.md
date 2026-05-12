# Implementation details

This document explains what the code actually does: the model and training
loop, every diagnostic that is logged, all of the synthetic data families
(including the adversarial ones used in the trained phase diagram), the figure
pipeline, and why the methodology is sound rather than cherry-picked. It is
meant to be read alongside `notes/theorem_sketch.md` (the math),
`paper/section_weighted_law.tex` and `paper/section_failure_modes.tex` (the
writeup), and `experiments/README.md` (the run commands).

Notation here follows the math notes: a two-layer network `f(x) = a^T phi(Bx)`
with `B in R^{m x d}`, residuals `r_i = f(x_i) - y_i`, hidden-derivative
features `q_i = a (.) phi'(Bx_i)`, and the stacked matrices `X = [x_1..x_n] in R^{d x n}`,
`Q = [q_1..q_n] in R^{m x n}`, `R = diag(r_1..r_n)`. The code uses transposed
shapes internally (`X` is `(n, d)`, `Q` is `(n, m)`). The formulas below are in
the math convention, and the mapping is noted where it matters.

## 1. Where things live

- `src/train_two_layer.py`  the model, the teacher, all data families, the
  training loop, checkpoint selection, and the per-run plotting.
- `src/weighted_metrics.py`  the per-checkpoint diagnostics (weighted law, raw
  law, beta link, stationarity).
- `src/pair_isotropy.py`  the pair-closure diagnostics (global `A_pair`, the
  singular-coordinate `F_X` / `G_stat` decomposition, high-gain closure, the
  theorem bound).
- `src/failure_regimes.py`  the deterministic algebraic toy regimes (used by
  `run_failure_modes.py --toy`).
- `experiments/run_pair_isotropy.py`  trains the positive and structured
  families across seeds, writes per-run histories and a grouped summary.
- `experiments/run_failure_modes.py`  the `--toy` algebraic checks and the
  `--trained` adversarial families used for the phase diagram.
- `experiments/make_final_figures.py`  paper and demo figures from a
  `compact_summary.json`.
- `paper/slides/make_slide_figures.py`  the five figures the slide deck uses,
  including the trained phase diagram.

## 2. Model and training pipeline

### Model
`TwoLayerSoftplus`: `f(x) = a^T softplus(Bx)`, with `B in R^{m_student x d}`,
`a in R^{m_student}`. The activation is softplus, so `phi'(z) = sigmoid(z)` and
`q_i = a (.) sigmoid(Bx_i)` (`compute_q`). Softplus is used because the
deterministic theorem statements assume a smooth activation, and the same
algebra applies to other smooth activations.

Initialization: `B0 = student_B_scale * randn(m_student, d) / sqrt(d)`,
`a0 = student_a_scale * randn(m_student) / sqrt(m_student)` (defaults
`student_B_scale = student_a_scale = 0.5`).

### Loss
`L(B, a) = (1 / 2n) sum_i r_i^2 + (lambda / 2) ||B||_F^2`. The L2 penalty is on
`B` only. This matters: it is exactly this objective whose `B`-stationarity
condition gives the identity `lambda n B = -Q R X^T + Z` that the whole project
rests on (`grad_B L = (1/n) Q R X^T + lambda B`, so `Z := n grad_B L` and
`Z = 0` at exact stationarity). Default `lambda = lambda_B = 1e-2`.

### Optimization: two phases
1. **SGD** for `sgd_steps` full-batch steps (`n = 128`, so this is gradient
   descent on the full objective, not stochastic minibatching), learning rate
   `sgd_lr = 5e-2`, gradient-norm clipping at `clip_grad_norm = 10`.
2. **L-BFGS** for `lbfgs_outer_steps` outer iterations (`torch.optim.LBFGS`,
   history size 50, learning rate 0.5, tight gradient and parameter tolerances).
   Each "outer step" is itself a quasi-Newton solve over the full objective.

Why two phases: SGD reliably gets into the right neighborhood but plateaus at a
noise floor (residuals around `1e-4`). L-BFGS then drives the iterate much
closer to interpolation and to a near-stationary point, which is the regime the
deterministic theorem describes. In the trajectory plots this shows up as a
sharp drop ("cliff") at training step `sgd_steps`: the L-BFGS phase is recorded
on the same step axis at steps `sgd_steps + 1, sgd_steps + 2, ...`, even though
an L-BFGS outer iteration is not comparable in magnitude to an SGD step. The
drop is a real regime change, not a plotting artifact beyond the axis stitching:
residuals go to zero, so the bridge scalar `beta_fit` (a weighted average of the
`r_i^2`) collapses with them, the raw-AGOP conversion becomes ill-conditioned,
and the weighted law stays accurate.

### Checkpoint protocol
A checkpoint is recorded every `checkpoint_every` SGD steps (and at the final
SGD step), then every `lbfgs_log_every` L-BFGS outer steps (and at the final
one). For the full schedule that is roughly 171 checkpoints per run. At each
checkpoint, `compute_weighted_metrics` (and the pair-isotropy metrics it calls)
are evaluated under `torch.no_grad()`.

### Hyperparameters
Full ("reported") schedule: `n = 128`, `d = 256`, `m_teacher = 4`,
`m_student = 16`, `lambda = 1e-2`, `sgd_steps = 3000`, `sgd_lr = 5e-2`,
`lbfgs_outer_steps = 100`, `dtype = float64`, `device = cpu`. The double
precision is deliberate: most of the diagnostics are differences of near-equal
matrices (`H^2 - kappa_eff G_tilde`, `S - d_eff T`, `M_tilde - beta_fit M`), so
single precision would swamp the signal near stationarity.

`--fast` schedule (`sgd_steps` around 300-400, `lbfgs_outer_steps` around 8-10):
smoke tests only, used to check that families, metrics, and plots are wired
together. Never the source of reported numbers.

## 3. The teacher and the data families

### Teacher
`f_*(x) = a_*^T softplus(B_* x)`. By default `B_* = teacher_B_scale * randn(m_teacher, d) / sqrt(d)`.
For families that carry a signal subspace, `B_*` is instead supported on that
subspace (`B_* = teacher_B_scale * (coeff @ basis^T)`, `coeff` random
`m_teacher x signal_rank` / sqrt(signal_rank)). `a_* = teacher_a_scale * randn(m_teacher) / sqrt(m_teacher)`.
Targets are `y = f_*(x) + noise_std * randn`, where `noise_std` defaults to 0,
and the families below use clean labels unless noted.

### Families (`data_family` switch in `generate_teacher_student_dataset`)

| `data_family` | inputs | notes |
| --- | --- | --- |
| `gaussian`, `anisotropic = False` | `x ~ N(0, I_d)` | the clean theory regime |
| `gaussian`, `anisotropic = True` | `x ~ N(0, Sigma)`, `Sigma = diag` with eigenvalues geometric from `1` down to `spectrum_min` (default `1e-2`) | mild covariance stress |
| `low_rank_signal` | `x = z basis^T + signal_noise_std * xi`, `z ~ N(0, I_k)`, `basis in R^{d x k}` random orthonormal (`k = signal_rank`), `xi ~ N(0, I_d)` | signal near a `k`-dim subspace, **isotropic inside it**, teacher supported on `basis` |
| `anisotropic_low_rank` | `x = (z (.) Lambda) basis^T + signal_noise_std * xi`, `Lambda = diag(lambda_1..lambda_k)` geometric from `1` down to `spectrum_min` | low-rank signal with a **steeply anisotropic within-subspace** covariance, teacher supported on `basis` |
| `clustered_gaussian` | `max(2, signal_rank)` centers `~ randn(d) / sqrt(d)`, samples assigned round-robin (shuffled), plus `signal_noise_std * N(0, I_d)` | clusters with similar center norms |
| `mixture_subspaces` | two random orthonormal subspaces of rank `signal_rank`, each sample drawn from one (alternating), coeffs `~ N(0, I / rank)`, plus `signal_noise_std * N(0, I_d)` | teacher supported on the first subspace's basis |
| `rare_region_outliers` | `x ~ N(0, I_d)`, then a random `rare_fraction` of samples shifted by `rare_shift * (unit direction)`, and for those samples `rare_label_scale` is **added** to the label | returns a `rare_mask` |
| `two_region_gating` | `x ~ N(0, I_d)`, label `= where(x_0 < 0, teacher_left(x), teacher_right(x))` with two independent teachers | a gated, piecewise task |
| `xor_feature` | `x ~ N(0, I_d)`, label `= teacher_a_scale * tanh(2 x_0 x_1)` | an XOR-style two-feature task |

The `cov_eigvals` returned alongside each dataset records the input-covariance
spectrum (or, for the low-rank families, `lambda_j^2` on the signal directions
and `signal_noise_std^2` elsewhere). It is for inspection, not used in training.

### Why these families

The positive families (`isotropic`, `anisotropic`, `low_rank_signal`) are
controls of increasing realism: a clean rotationally-symmetric setting, a mild
covariance perturbation, and a structured signal-plus-noise setting. They test
whether the weighted law survives beyond the cleanest case and whether the
relevant pair diagnostic is the global one or the pushed one.

The remaining families are deliberately adversarial, each aimed at one bridge:

- **Beta link.** `rare_region_outliers` with amplified labels puts a small
  cluster of samples that have both high hidden-gradient leverage and high
  residual energy, which makes `Cov_n(leverage, r^2)` large and bends
  `beta_fit` away from the mean residual energy. With de-amplified labels it
  bends it the other way. `clustered_gaussian` and `mixture_subspaces` also
  perturb leverage.
- **High-gain pair closure.** This turns out to be hard to break, because the
  deterministic obstruction is not "anisotropic inputs" but "`F_X` large on the
  high-gain directions of `G_stat`". A full-rank anisotropic Gaussian does
  **not** do it: the `n x n` sample Gram `X^T X` of high-dimensional Gaussian
  data concentrates around `tr(Sigma) I_n`, so `H_X = V^T (X^T X) V` is close to
  scalar regardless of how anisotropic `Sigma` is, and `F_X` stays small (this
  is exactly what is observed for `strong_anisotropic`). `anisotropic_low_rank`
  was built to fix that: the signal lives near a `k`-dim subspace, the
  within-subspace covariance is steeply anisotropic, the teacher uses the same
  subspace, so `X^T X` is genuinely non-scalar on the directions the network
  relies on (the sample Gram of one such draw has eigenvalues `130, 66, 38, 21, ...`
  then a tail at `~0.06`). In practice it still does not break the *weighted*
  law: near stationarity `B -> -Q R X^T / (lambda n)`, so the only directions
  that enter the pushed error are the ones the residual-weighted hidden
  gradients span, and the network self-aligns those with the input geometry, so
  `S ~ d_eff T` holds on exactly the directions `B` sees even when the whitened
  global defect `A_pair` (and even `pair_high_gain_closure_op`) is large. The
  upshot is a robustness statement: the only family that degrades the weighted
  law in these runs is `rare_hard_cluster`, where amplified outlier labels keep
  residuals large persistently and so sustain a misaligned `Q R` mass the
  network cannot iron out, and that same mechanism also bends the beta link
  (a *combined* failure, not a pure pair one). See the trained-family list in
  section 6.
- **Raw-law conditioning.** Near interpolation `beta_fit -> 0`, so the
  conversion from weighted to raw AGOP is ill-conditioned even when the weighted
  law is perfect. This is a regime, not a separate family. It appears at the
  late checkpoints of every run.

## 4. The diagnostics

These are computed by `compute_weighted_metrics` plus the
`compute_pair_isotropy_metrics` and `compute_pair_spectral_gain_metrics` it
calls.

Hidden objects: `q_i = a (.) sigmoid(Bx_i)`, `r_i = f(x_i) - y_i`,
`M = (1/n) sum_i q_i q_i^T`, `M_tilde = (1/n) sum_i r_i^2 q_i q_i^T`,
`T = n M_tilde = Q R^2 Q^T`, `A = Q R X^T`, `S = A A^T = Q R X^T X R Q^T`,
`H = B^T B`, `G = B^T M B`, `G_tilde = B^T M_tilde B`.

### Stationarity
- `Z = lambda n B + Q R X^T` (the defect), `E_stat = (1/lambda^2 n^2)(-A Z^T - Z A^T + Z Z^T)`,
  `E_stat_op = ||E_stat||_op`. At exact stationarity `Z = 0` and `BB^T = S / (lambda^2 n^2)`.
- `delta_stationary_op = ||BB^T - B_stat B_stat^T||_op` with `B_stat = -A / (lambda n)`
  (the value `B` would take at exact stationarity). This is the quantity
  minimized to pick the "best stationarity" checkpoint.
- `stationarity_rel = E_stat_op / (||BB^T||_op + ||B_stat B_stat^T||_op)`, a
  relative version also tracked through training.

### Weighted square law
- `d_eff = <S, T>_F / ||T||_F^2` (the best Frobenius scalar fitting `S ~ d T`),
  `kappa_eff = d_eff / (lambda^2 n)`.
- `gamma_tilde_eff_op = ||H^2 - kappa_eff G_tilde||_op`, the central quantity:
  "how well does the residual-weighted square law hold?"
- `pair_error = S - d_eff T`, `pair_push_op = ||B^T (S - d_eff T) B||_op`,
  `pair_push_scaled_op = pair_push_op / (lambda^2 n^2)`. By the theorem, at
  exact stationarity `H^2 - kappa_eff G_tilde = (1/lambda^2 n^2) B^T (S - d_eff T) B`,
  so `pair_push_scaled_op = gamma_tilde_eff_op` up to the stationarity defect.
  This is the bridge-relevant pair error, and it is what the phase diagram puts
  on its vertical axis.
- `gamma_tilde_eff_rel = gamma_tilde_eff_op / (||H^2||_op + |kappa_eff| ||G_tilde||_op)`,
  the symmetric, "theorem-natural" relative error (equal to the relative pushed
  pair-closure error). `gamma_tilde_eff_rel_h2 = gamma_tilde_eff_op / ||H^2||_op`
  is the alternative normalization. It is logged but is *not* used as a headline
  metric because it blows up near interpolation when `||H^2||_op` is small. The
  notes call this out explicitly. Nothing about the unstable normalization is
  hidden.

### Pair-closure and high-gain decomposition (`src/pair_isotropy.py`)
- Global, support-normalized defect: with `T = T_sym`, project onto `range(T)`
  (eigenvalues above `eps_rel * max_eval`), form `pair_core = T^{+/2} S T^{+/2} - d_eff P_T`,
  and `A_pair_op = ||pair_core||_op`. This is the worst-direction defect on the
  whole support of `T`. It can be large while the pushed error is tiny ("the
  conservative diagnostic").
- Singular coordinates: with the thin SVD `Q R = U Sigma V^T` (in code,
  `C = Q R`, then `svd(C) = U Sigma Vh`, `V = Vh^T`), define `H_X = V^T (X^T X) V`,
  `F_X = H_X - d_eff I`, `G_stat = Sigma^2 V^T X^T`. Then at exact stationarity
  `B^T (S - d_eff T) B = (1/lambda^2 n^2) G_stat^T F_X G_stat`. Logged: the
  spectral defect `pair_spectral_defect_op = ||F_X||_op` (equals `A_pair_op`),
  per-direction `abs_defects` and stationarity `gains` (`||X^T V^T Sigma^2 e_k||`),
  their product `weighted = abs_defects * gains^2` (the per-direction
  contribution to the pushed error), and `pair_gain_defect_corr_abs` (do the big
  defects coincide with high gain?).
- High-gain closure: `high_basis` = top `ceil(0.25 * rank)` left-singular
  vectors of `G_stat`, `pair_high_gain_closure_op = ||high_basis^T F_X high_basis||_op`,
  `pair_low_gain_op = ||(I - P_hi) G_stat||_op`. `pair_damping_bound_proxy`
  combines these into the high-gain-closure bound from the theorem sketch.
- Theorem bound: `theorem_bound_op = ||B||_op^2 ( (||T||_op / lambda^2 n^2) A_pair_op + E_stat_op )`,
  `theorem_bound_ratio = gamma_tilde_eff_op / theorem_bound_op` (a value below 1
  means the deterministic bound covers the observed weighted residual, and the
  bound is expected to be loose because it uses the conservative `A_pair`).

### Beta link (weighted to raw AGOP)
- `beta_fit = <M_tilde, M>_F / ||M||_F^2`. Using `<q_i q_i^T, q_j q_j^T>_F = (q_i^T q_j)^2`,
  this equals `(sum_{ij} r_i^2 (q_i^T q_j)^2) / (sum_{ij} (q_i^T q_j)^2)`.
- Leverage: `K_ij = (q_i^T q_j)^2` (`hidden_kernel_sq`), `s_i = sum_j K_ij`
  (`leverage_scores`), `ell_i = s_i / (mean_k s_k)` (`leverage_normalized`, mean
  1). Then `beta_fit = (1/n) sum_i ell_i r_i^2`, so `beta_fit - r_bar^2 = Cov_n(ell, r^2)`.
- `beta_over_resid_mean_sq = beta_fit / r_bar^2`, `beta_rel_mean_sq_error = |beta_fit / r_bar^2 - 1|`,
  `beta_over_resid_max_sq = beta_fit / max_i r_i^2`.
- `leverage_resid_sq_corr = Corr_n(ell, r^2)`, `leverage_cv = CV_n(ell)`,
  `resid_sq_cv = CV_n(r^2)`, `beta_cv_bound = CV_n(ell) CV_n(r^2)` (the
  Cauchy-Schwarz bound), `beta_corr_cv_product = Corr_n(ell, r^2) CV_n(ell) CV_n(r^2)`
  (which equals `beta_fit / r_bar^2 - 1` exactly), `beta_corr_cv_product_abs` its
  absolute value. The identity `|beta_corr_cv_product| == beta_rel_mean_sq_error`
  holds numerically up to floating point, and the `--toy` checks verify the beta
  identity to `< 1e-10`.
- `M_bridge = M_tilde - beta_fit M`, `M_bridge_op`, `M_bridge_rel_frob = ||M_bridge||_F / ||M_tilde||_F`
  (the operator and Frobenius residuals of the `M_tilde ~ beta M` fit).

### Raw AGOP law
- `c_eff = 1 / (kappa_eff beta_fit)` (when `|kappa_eff beta_fit| > 1e-16`),
  `gamma_eff_op = ||G - c_eff H^2||_op` (the raw-law residual, written in the
  "solve for `G`" direction, with `c_eff -> infinity` as `beta_fit -> 0`, which
  is the conditioning failure). `gamma_eff_rel` is the symmetric relative
  version.
- `c_fit = best scalar fitting G ~ c H^2`, `gamma_fit_op = ||G - c_fit H^2||_op`
  (the best-case raw fit, independent of `kappa_eff` and `beta_fit`).
- `sqrt_law_op` and `sqrt_law_rel` track the alternative `H ~ s sqrt(G)` form
  for comparison.

### Trajectory and dynamics diagnostics
`annotate_phase_metrics` adds, between consecutive checkpoints, per-sample
log-residual decay rates and `leverage_damping_corr` (`Corr_n(ell_i, -Delta log r_i^2)`,
the empirical test of the leverage-sensitive damping conjecture),
`mean_log_resid_sq_decay`, `raw_quality` and `weighted_quality` (relative
scores), and a `crossover_score`. Per-direction pair histories are also recorded
for the diagnostic plots.

## 5. Checkpoint selection and summaries

Each run picks three named checkpoints (`metric_better` does the comparison,
with the gradient norm as a tie-break):

- `best_metrics_by_stationarity`  minimal `delta_stationary_op`. This is the
  checkpoint where the theorem's hypothesis (`Z ~ 0`) holds best, so it is where
  the weighted law and its bound are most meaningful. The phase diagram reads
  most of its quantities here.
- `best_metrics_by_weighted_law`  minimal `gamma_tilde_eff_op`.
- `best_metrics_by_raw_conditioning`  minimal `gamma_eff_op` **among checkpoints
  with `|beta_fit| >= beta_threshold`** (default `1e-5`). The threshold exists
  to avoid the degenerate "the raw law looks great here only because `H^2 ~ 0`
  and everything is tiny" trap. This is the intermediate-regime checkpoint where
  raw AGOP gets its best honest shot.
- `crossover_metrics`  the first checkpoint where raw quality starts degrading
  relative to weighted quality.

`compute_weighted_metrics` plus the pair metrics are computed once per
checkpoint and stored verbatim in `history.json`. The three "best" blocks are
just the rows that minimize the corresponding key. `build_summary`
(`run_pair_isotropy.py`) groups runs by `data_family` (with the `anisotropic`
flag splitting `gaussian` into `isotropic` and `anisotropic`) and aggregates
mean/std/min/max over seeds, both at the best-stationarity checkpoint and over
all checkpoints, into `summary.json` and `compact_summary.json`.

## 6. The trained adversarial families (the phase diagram)

`run_failure_modes.py --trained --seeds ...` runs, per seed, the configs in
`trained_configs` (all with `noise_std = 0`, full schedule unless `--fast`):

| name | `data_family` | key parameters | intended target | observed (5 seeds) |
| --- | --- | --- | --- | --- |
| `rare_hard_cluster` | `rare_region_outliers` | `rare_fraction = 0.1`, `rare_shift = 4.0`, `rare_label_scale = 4.0` | beta link (positive `Cov(ell, r^2)`), also dominates `Q R X^T` | beta error elevated **and** weighted-law error up by orders of magnitude (a *combined* failure) |
| `rare_easy_cluster` | `rare_region_outliers` | `rare_label_scale = 0.4` | beta link (negative `Cov(ell, r^2)`) | beta error elevated, weighted law fine |
| `two_region_gating_stress` | `two_region_gating` | `signal_noise_std = 0.25` | nonlinear gating stress | mild on both axes, weighted law somewhat degraded |
| `mixture_subspaces_stress` | `mixture_subspaces` | `signal_rank = 3`, `signal_noise_std = 0.05` | non-scalar `X^T X` via subspace structure | stresses the beta link more than pair closure (the two subspaces are symmetric, so `X^T X` on the union is close to scalar) |
| `strong_anisotropic` | `gaussian` (anisotropic) | `spectrum_min = 1e-4` | high-gain pair closure | does **not** break the weighted law: a full-rank anisotropic Gaussian has `X^T X ~ tr(Sigma) I_n`. Lands bottom-left with the positives. |
| `anisotropic_low_rank` | `anisotropic_low_rank` | `signal_rank = 10`, `spectrum_min = 0.08`, `signal_noise_std = 0.05` | a *pure* high-gain pair-closure failure | `X^T X` is strongly non-scalar on the subspace the network uses, but the *pushed* pair error stays tiny because the network self-aligns near stationarity. Also lands bottom-left, alongside `strong_anisotropic`: the weighted law is robust to anisotropy attacks. |

So in these runs the weighted law degrades by orders of magnitude only for
`rare_hard_cluster`, and somewhat for `two_region_gating_stress`. Both have a
persistent-residual mechanism that sustains a misaligned `Q R` near
stationarity, which is also what bends their beta link. The three anisotropy and
mixture constructions all sit in the healthy bottom-left cluster.

The phase diagram (slide 13, `paper/slides/figures/phase_diagram.pdf`) plots
each family at `x = |beta_fit / r_bar^2 - 1|` at the best raw-AGOP checkpoint
(the beta-link error, evaluated where the raw law is most relevant rather than
at interpolation where it is degenerate) against `y = ||B^T (S - d_eff T) B|| / (lambda^2 n^2)`
at the best-stationarity checkpoint (the weighted-law residual). Markers are
per-family medians over the seeds, whiskers span the seeds, and the faint clouds
are the individual seeds.

## 7. The figure pipeline

- `experiments/make_final_figures.py --results-dir <runs> --outdir <figs>`:
  bar-by-family figures (weighted residual, theorem-bound ratio, beta tracking,
  pushed pair error, symmetric relative error, and several pair and beta
  diagnostics) from `<runs>/compact_summary.json`, plus copies of representative
  per-run plots and a `manifest.txt`. The committed `paper/figures/` set comes
  from this.
- `paper/slides/make_slide_figures.py [--positive-root ...] [--adversarial-root ...] [--seeds ...]`:
  the five deck figures. `two_regime_isotropic_seed_0.pdf` (weighted vs raw law
  error, `beta_fit`, mean `r^2` along training, with the best-stationarity
  checkpoint marked), `weighted_law_residual.pdf` (observed residual vs theorem
  bound by positive family), `beta_tracking.pdf` (`beta_fit / r_bar^2` by
  positive family), `pushed_pair_error.pdf` (global `A_pair` vs pushed pair
  error by positive family), and `phase_diagram.pdf` (section 6).

## 8. Why the methodology is sound, not cherry-picked

- **The plotted quantities are the theorem's objects, not generic accuracy.**
  Everything reported is one of the terms in the deterministic identities
  `H^2 - kappa_eff G_tilde = (1/lambda^2 n^2) B^T (S - d_eff T) B + B^T E_stat B`
  and `beta_fit = (1/n) sum_i ell_i r_i^2`, or a ratio of such terms. The
  experiments test the *bridge assumptions* (pair closure, beta stability,
  near-stationarity), not whether NFA "works" as a black box.
- **The identities are checked, not assumed.** `beta_corr_cv_product_abs` equals
  `|beta_fit / r_bar^2 - 1|` to floating point at every checkpoint, and the
  `run_failure_modes.py --toy` constructions verify `beta_fit = (1/n) sum_i ell_i r_i^2`
  to `< 1e-10`. The whole pipeline runs in float64 because the diagnostics are
  differences of near-equal matrices.
- **Checkpoint selection is by explicit, fixed rules** (minimal stationarity
  defect, minimal weighted residual, minimal raw residual subject to
  `|beta_fit| >= beta_threshold`), not hand-picked per run. The `beta_threshold`
  is there precisely to block the "raw law looks perfect because everything is
  zero" artifact.
- **Conventions are stated and the unstable ones are flagged.** Normalized vs
  unnormalized weighted AGOP (`kappa_eff` vs `kappa_u`), operator norm vs
  Frobenius, and the symmetric "theorem-natural" relative error vs the
  `||H^2||`-only normalization (which is logged but deliberately not used as a
  headline metric because it diverges near interpolation). None of this is
  hidden.
- **The "cliff" at step `sgd_steps` is the SGD to L-BFGS handoff**, not a bug.
  The post-handoff behavior is exactly what the theory predicts: `r_i^2 -> 0`,
  `beta_fit -> 0` with them, raw conditioning fails, weighted law stays
  accurate. The deck and `notes/experiment_log.md` say so.
- **Seeds are run and the spread is shown.** Five seeds per family in the current
  phase diagram, and the figure shows per-seed clouds and whiskers, not just
  medians. The per-seed variation of `pair_push_scaled_op` near interpolation is
  genuinely large (it is a near-zero quantity, and the exact best-stationarity
  checkpoint differs by seed). This is stated rather than smoothed away.
- **Negative results are kept, not dropped, and they became the finding.**
  `mixture_subspaces`, `strong_anisotropic`, and the purpose-built
  `anisotropic_low_rank` were all *intended* to break high-gain pair closure,
  and none of them did. Rather than dropping them, we report them in their
  actual positions on the map, because "the weighted law is robust to these
  attacks, and the only family that degrades it (`rare_hard_cluster`) does so
  through a persistent-residual mechanism that also bends the beta link" is a
  more honest and more useful conclusion than "we broke it cleanly". The deck's
  slide 13 is framed around this.

### Limitations (so a reader can calibrate the claims)

- Small scale: `n = 128`, `d = 256`, `m_student = 16`, one architecture
  (two-layer softplus, square loss, L2 on `B` only). The families are synthetic.
- The deterministic theorem assumes *exact* `B`-stationarity. We report at the
  best-stationarity checkpoint, where `Z` is small but nonzero, and `E_stat` is
  logged so its contribution is visible. `theorem_bound_ratio` uses the
  conservative `A_pair`, so it is expected to be loose, not tight.
- The conditional pair-closure theorem assumes a frozen or `X`-independent
  high-gain subspace. The trained `V` is neither, so that result is a sufficient
  mechanism, not a proof for trained networks.
- The trained phase diagram is a guiding map, not a deliberate grid sweep over
  `Corr_n(ell, r^2)` times high-gain anisotropy with samplewise diagnostics. The
  FL-PLAN notes describe the fuller version as future work.
- The dynamics-bridge claims (leverage-sensitive residual damping, adaptive
  high-gain concentration) remain conjectural. `leverage_damping_corr` is logged
  as the relevant empirical probe but no theorem-level statement is claimed.

## 9. Reproducing the runs

```bash
# positive families, 5 seeds
python -m experiments.run_pair_isotropy --seeds 0,1,2,3,4 --include-low-rank \
  --output-root results/runs/phase_diagram_positive

# adversarial families, 5 seeds
python -m experiments.run_failure_modes --trained --seeds 0,1,2,3,4 \
  --output-root results/runs/phase_diagram_adversarial \
  --figure-dir results/runs/phase_diagram_adversarial/fig

# deterministic algebraic toy checks
python -m experiments.run_failure_modes --toy

# slide figures
python paper/slides/make_slide_figures.py --seeds 0,1,2,3,4

# build the deck
cd paper/slides && pdflatex slides.tex
```

`results/runs/` and `results/tmp/` are gitignored. The committed evidence is the
figures under `paper/figures/`, the slide figures under `paper/slides/figures/`,
and the summaries quoted in `notes/experiment_log.md`.
