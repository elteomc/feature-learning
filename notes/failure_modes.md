# Failure-Mode Taxonomy

This note classifies how AGOP-style laws can hold or fail in the two-layer
feature-learning setting. The goal is not to prove a universal NFA theorem. The
goal is to identify which algebraic bridge fails in each regime.

The status is:

- The decompositions below are deterministic algebra.
- The toy counterexamples are finite-dimensional algebraic examples.
- Claims about trained nonlinear networks are empirical or conditional unless
  explicitly stated as algebra.

## Setup

We use the two-layer model

$$
f(x)=a^\top \phi(Bx),
\qquad
H:=B^\top B.
$$

For samples \((x_i,y_i)_{i=1}^n\), define

$$
r_i:=f(x_i)-y_i,
\qquad
q_i:=a\odot \phi'(Bx_i).
$$

Then

$$
\nabla f(x_i)=B^\top q_i.
$$

The raw hidden covariance and residual-weighted hidden covariance are

$$
M:=\frac1n\sum_i q_iq_i^\top,
\qquad
\widetilde M:=\frac1n\sum_i r_i^2q_iq_i^\top.
$$

The raw and weighted AGOP matrices are

$$
G:=B^\top MB,
\qquad
\widetilde G:=B^\top \widetilde M B.
$$

The stationarity-derived late law is

$$
H^2\approx \kappa_{\mathrm{eff}}\widetilde G.
$$

The raw AGOP law is a derived relation. It needs both a good beta bridge

$$
\widetilde M\approx \beta_{\mathrm{fit}}M
$$

and a beta value that is not too small.

## Axis 1: beta bridge

The Frobenius best-fit scalar is

$$
\beta_{\mathrm{fit}}
:=
\frac{\langle \widetilde M,M\rangle_F}{\|M\|_F^2}.
$$

Let

$$
K_{ij}:=(q_i^\top q_j)^2,
\qquad
s_i:=\sum_jK_{ij},
\qquad
\ell_i:=\frac{s_i}{\frac1n\sum_k s_k},
\qquad
u_i:=r_i^2.
$$

Then

$$
\frac1n\sum_i\ell_i=1
$$

and

$$
\beta_{\mathrm{fit}}
=
\frac1n\sum_i\ell_i u_i.
$$

Therefore

$$
\beta_{\mathrm{fit}}-\bar u
=
\frac1n\sum_i(\ell_i-1)(u_i-\bar u)
=
\operatorname{Cov}_n(\ell,u),
\qquad
\bar u:=\frac1n\sum_i u_i.
$$

When empirical variances are nonzero,

$$
\frac{\beta_{\mathrm{fit}}}{\bar u}-1
=
\operatorname{Corr}_n(\ell,u)
\operatorname{CV}_n(\ell)
\operatorname{CV}_n(u).
$$

So the beta bridge tracks mean residual energy exactly when the leverage-weighted
residual average is close to the ordinary residual average.

### Beta-failure criterion

Beta tracking fails when

$$
|\operatorname{Cov}_n(\ell,r^2)|
$$

is large. This can happen even if the weighted law itself holds.

## Axis 2: high-gain pair closure

Let

$$
Q=[q_1,\ldots,q_n],
\qquad
R=\operatorname{diag}(r_1,\ldots,r_n),
\qquad
X=[x_1,\ldots,x_n].
$$

Define

$$
A:=QR,
\qquad
T:=AA^\top=QR^2Q^\top,
\qquad
S:=AX^\top XA^\top=QRX^\top XRQ^\top.
$$

The best scalar pair fit is

$$
d_{\mathrm{eff}}
:=
\frac{\langle S,T\rangle_F}{\|T\|_F^2}.
$$

Let

$$
E_{\mathrm{pair}}:=S-d_{\mathrm{eff}}T.
$$

The global support-normalized pair diagnostic is

$$
\mathcal A_{\mathrm{pair}}
:=
\|T^{\dagger/2}E_{\mathrm{pair}}T^{\dagger/2}\|_{\mathrm{op}}.
$$

This diagnostic can be conservative. The weighted law only sees

$$
B^\top E_{\mathrm{pair}}B.
$$

Let the thin SVD of \(A\) be

$$
A=U\Sigma V^\top.
$$

Define

$$
H_X:=V^\top X^\top XV,
\qquad
F_X:=H_X-d_{\mathrm{eff}}I.
$$

Then

$$
\mathcal A_{\mathrm{pair}}=\|F_X\|_{\mathrm{op}}.
$$

At exact \(B\)-stationarity,

$$
B=-\frac1{\lambda n}U\Sigma V^\top X^\top.
$$

Therefore

$$
B^\top E_{\mathrm{pair}}B
=
\frac1{\lambda^2n^2}
G_{\mathrm{stat}}^\top F_XG_{\mathrm{stat}},
\qquad
G_{\mathrm{stat}}:=\Sigma^2V^\top X^\top.
$$

### Pair-failure criterion

Weighted pair closure fails when \(F_X\) is large on high-gain directions of
\(G_{\mathrm{stat}}\). A large global \(\mathcal A_{\mathrm{pair}}\) is not by
itself enough. The defect must also be visible to the stationarity-induced gain
map.

## Axis 3: raw-law conditioning

Suppose the weighted law and beta bridge both hold in absolute error:

$$
\|H^2-\kappa_{\mathrm{eff}}\widetilde G\|_{\mathrm{op}}\le E_w,
\qquad
\|\widetilde G-\beta G\|_{\mathrm{op}}\le E_\beta.
$$

Then

$$
\|H^2-\kappa_{\mathrm{eff}}\beta G\|_{\mathrm{op}}
\le
E_w+\kappa_{\mathrm{eff}}E_\beta.
$$

If \(\beta\ne 0\), solving for the raw law gives

$$
\left\|
G-\frac1{\kappa_{\mathrm{eff}}\beta}H^2
\right\|_{\mathrm{op}}
\le
\frac{E_w+\kappa_{\mathrm{eff}}E_\beta}
{\kappa_{\mathrm{eff}}|\beta|}.
$$

So raw AGOP becomes ill-conditioned as \(|\beta|\to 0\), even when the weighted
law remains stable.

## Regime taxonomy

| Regime | Beta bridge | High-gain pair closure | Weighted law | Raw law |
| --- | --- | --- | --- | --- |
| Fully benign | holds | holds | holds | holds if beta is not tiny |
| Late weighted-only | structurally holds | holds | holds | ill-conditioned because beta is tiny |
| Beta failure | fails high or low | may hold | may hold | biased or fails |
| Pair failure | may hold | fails | fails | fails unless accidental cancellation occurs |
| Conservative pair diagnostic | may hold | pushed error small despite large global defect | holds | depends on beta |
| Intermediate raw success | holds | approximately holds | may be improving | can hold before beta collapse |

## Algebraic examples

### Example 1: beta overestimation from a high-leverage hard sample

Let \(n\ge 2\). Suppose one sample has leverage much larger than the rest:

$$
s_1\gg \sum_{i=2}^n s_i.
$$

Then

$$
\beta_{\mathrm{fit}}
=
\frac{\sum_i s_i r_i^2}{\sum_i s_i}
\approx
r_1^2.
$$

If

$$
r_1^2=1,
\qquad
r_i^2=0\quad (i\ge 2),
$$

then

$$
\beta_{\mathrm{fit}}\approx 1,
\qquad
\bar r^2=\frac1n.
$$

Thus

$$
\frac{\beta_{\mathrm{fit}}}{\bar r^2}\approx n.
$$

Beta tracking is not universal. A high-leverage high-residual sample can make
beta much larger than ordinary residual energy.

### Example 2: beta underestimation from a high-residual low-leverage sample

Again let \(n\ge 2\), but suppose

$$
s_1\ll \sum_{i=2}^n s_i.
$$

With the same residual pattern

$$
r_1^2=1,
\qquad
r_i^2=0\quad (i\ge 2),
$$

we get

$$
\beta_{\mathrm{fit}}\approx 0,
\qquad
\bar r^2=\frac1n.
$$

So beta can also underestimate mean residual energy. The beta bridge fails
whenever residual energy concentrates on leverage-extreme samples.

### Example 3: high-gain bad pair direction

Consider an aligned rank-two pair model with

$$
H_X=
\begin{pmatrix}
0 & 0\\
0 & 2
\end{pmatrix},
\qquad
\Sigma^2=I.
$$

Then

$$
d_{\mathrm{eff}}=1,
\qquad
F_X=
\begin{pmatrix}
-1 & 0\\
0 & 1
\end{pmatrix},
\qquad
\mathcal A_{\mathrm{pair}}=1.
$$

In the aligned formula, the direction-wise pushed contribution scales like

$$
\lambda_k\gamma_k^2|\lambda_k-d_{\mathrm{eff}}|.
$$

The second direction has

$$
\lambda_2=2,
\qquad
\gamma_2=1,
$$

so its contribution is \(2\). The bad pair direction is high-gain. The pushed
pair error is large, so the weighted law can fail.

### Example 4: large global pair defect but small pushed error

Let

$$
H_X=
\begin{pmatrix}
0 & 0\\
0 & 1
\end{pmatrix},
\qquad
\Sigma^2=
\begin{pmatrix}
\varepsilon & 0\\
0 & 1
\end{pmatrix},
\qquad
0<\varepsilon\ll 1.
$$

Then

$$
d_{\mathrm{eff}}
=
\frac{1}{1+\varepsilon^2}
$$

and

$$
\mathcal A_{\mathrm{pair}}\approx 1.
$$

The low-input-energy bad direction has very small stationarity gain. Its pushed
contribution is suppressed. The remaining pushed contribution is \(O(\varepsilon^2)\).

Thus global \(\mathcal A_{\mathrm{pair}}\) can be large while the pushed pair
error is small.

## Data-regime interpretation

| Data regime | Expected mechanism | Prediction |
| --- | --- | --- |
| Isotropic exchangeable teacher-student | weak leverage-residual covariance and high-gain scalar closure | weighted law holds late, raw law holds only while beta is not tiny |
| Rare hard cluster | rare samples have high leverage and high residual | beta overestimates residual energy, raw law is biased |
| Rare easy cluster | rare high-leverage samples have low residual | beta underestimates residual energy, raw law is biased |
| High-gain anisotropic pair geometry | high-gain subspace aligns with anisotropic \(H_X\) directions | pushed pair error is large, weighted law degrades |
| Low-gain anisotropic pair geometry | bad global pair directions have low gain | global pair diagnostic looks bad, weighted law can still hold |
| Near interpolation | beta collapses | weighted law remains stable, raw law is ill-conditioned |

## What remains open

The algebra explains what can fail. It does not prove that a trained nonlinear
network enters any particular failure regime.

A trained-network theorem would need assumptions such as:

- leverage-residual covariance is controlled or forced to be large by the data
  construction;
- high-gain subspaces have either scalar or non-scalar input geometry;
- leverage drift and NTK residual damping are controlled well enough to connect
  training dynamics to the static diagnostics.

The experiment plan should therefore report trained failure-mode runs as
empirical probes, not as theorems.
