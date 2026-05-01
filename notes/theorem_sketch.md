# Weighted deterministic proposition

## Setup

We study the two-layer model

$$
  f(x)=a^\top \phi(Bx), \qquad B \in \mathbb{R}^{m \times d},
$$

trained with square loss and $L_2$-regularization on $B$:

$$
  \mathcal{L} (B,a)
  =
  \frac{1}{2n}\sum_{i=1}^n (f(x_i)-y_i)^2
  +
  \frac{\lambda}{2}\|B\|_F^2.
$$

For each sample define

$
r_i := f(x_i)-y_i,
\qquad
q_i := a\odot \phi'(Bx_i)\in\mathbb R^m.
$

Stack

$
X := [x_1,\dots,x_n]\in\mathbb R^{d\times n},
\qquad
Q := [q_1,\dots,q_n]\in\mathbb R^{m\times n},
\qquad
R := \mathrm{diag}(r_1,\dots,r_n)\in\mathbb R^{n\times n}.
$

Define

$
H := B^\top B,
\qquad
T := Q R^2 Q^\top,
\qquad
S := Q R X^\top X R Q^\top.
$

Define the residual-weighted AGOP

$
\widetilde G := B^\top T B
= \sum_{i=1}^n r_i^2 \,\nabla f(x_i)\nabla f(x_i)^\top,
$

since \(\nabla f(x_i)=B^\top q_i\).

Define the stationarity defect

$
Z := \lambda n\,B + Q R X^\top.
$

Define

$
d_{\mathrm{eff}}
:=
\frac{\langle S,T\rangle_F}{\|T\|_F^2},
\qquad
\kappa_{\mathrm{eff}}
:=
\frac{d_{\mathrm{eff}}}{\lambda^2 n^2}.
$

If \(T\) is singular, let \(P_T\) be the orthogonal projector onto \(\mathrm{range}(T)\), and define

$
\mathcal A_{\mathrm{pair}}
:=
\left\|
T^{\dagger/2} S T^{\dagger/2}
-
d_{\mathrm{eff}} P_T
\right\|_{\mathrm{op}}.
$

---

## Proposition

With the notation above,

$
\|H^2-\kappa_{\mathrm{eff}}\widetilde G\|_{\mathrm{op}}
\le
\|B\|_{\mathrm{op}}^2
\left(
\frac{\|T\|_{\mathrm{op}}}{\lambda^2 n^2}\,\mathcal A_{\mathrm{pair}}
+
\|E_{\mathrm{stat}}\|_{\mathrm{op}}
\right),
$

where

$
E_{\mathrm{stat}}
:=
\frac{1}{\lambda^2 n^2}
\left(
-QRX^\top Z^\top
-
ZXRQ^\top
+
ZZ^\top
\right).
$

Moreover,

$
\|E_{\mathrm{stat}}\|_{\mathrm{op}}
\le
\frac{1}{\lambda^2 n^2}
\left(
2\|QRX^\top\|_{\mathrm{op}}\|Z\|_{\mathrm{op}}
+
\|Z\|_{\mathrm{op}}^2
\right).
$

In particular, at exact stationarity (\(Z=0\)),

$
\|H^2-\kappa_{\mathrm{eff}}\widetilde G\|_{\mathrm{op}}
\le
\frac{\|B\|_{\mathrm{op}}^2\,\|T\|_{\mathrm{op}}}{\lambda^2 n^2}
\mathcal A_{\mathrm{pair}}.
$

If additionally

$
T^{\dagger/2} S T^{\dagger/2}=d_{\mathrm{eff}}P_T,
$

then

$
H^2=\kappa_{\mathrm{eff}}\widetilde G.
$

---

## Proof sketch

From

$
\nabla_B \mathcal L
=
\frac1n QRX^\top + \lambda B,
$

we get

$
\lambda n\,B = -QRX^\top + Z.
$

Hence

$
BB^\top
=
\frac{1}{\lambda^2 n^2}
(QRX^\top - Z)(XRQ^\top - Z^\top)
=
\frac{1}{\lambda^2 n^2}S + E_{\mathrm{stat}}.
$

Now write

$
S = d_{\mathrm{eff}}T + E_{\mathrm{pair}},
\qquad
E_{\mathrm{pair}}:=S-d_{\mathrm{eff}}T.
$

Then

$
E_{\mathrm{pair}}
=
T^{1/2}
\Bigl(
T^{\dagger/2}ST^{\dagger/2}-d_{\mathrm{eff}}P_T
\Bigr)
T^{1/2},
$

so

$
\|E_{\mathrm{pair}}\|_{\mathrm{op}}
\le
\|T\|_{\mathrm{op}}\mathcal A_{\mathrm{pair}}.
$

Thus

$
BB^\top
=
\kappa_{\mathrm{eff}}T
+
\frac{1}{\lambda^2 n^2}E_{\mathrm{pair}}
+
E_{\mathrm{stat}}.
$

Applying \(B^\top(\cdot)B\),

$
H^2
=
B^\top(BB^\top)B
=
\kappa_{\mathrm{eff}} B^\top T B
+
B^\top
\left(
\frac{1}{\lambda^2 n^2}E_{\mathrm{pair}}
+
E_{\mathrm{stat}}
\right)B.
$

Since \(B^\top T B=\widetilde G\),

$
H^2-\kappa_{\mathrm{eff}}\widetilde G
=
B^\top
\left(
\frac{1}{\lambda^2 n^2}E_{\mathrm{pair}}
+
E_{\mathrm{stat}}
\right)B.
$

Taking operator norms gives the proposition.

---

## Raw-AGOP bridge collapse

The weighted law above is the stable late-training object. To relate it back to
the unweighted AGOP, define

$$
M := \frac1n\sum_{i=1}^n q_i q_i^\top,
\qquad
\widetilde M := \frac1n\sum_{i=1}^n r_i^2 q_iq_i^\top.
$$

Let

$$
\beta_{\mathrm{fit}}
:=
\frac{\langle \widetilde M,M\rangle_F}{\|M\|_F^2}.
$$

### Lemma

If \(\|M\|_F>0\), then

$$
\beta_{\mathrm{fit}}
=
\frac{
\sum_{i,j} r_i^2 (q_i^\top q_j)^2
}{
\sum_{i,j} (q_i^\top q_j)^2
}.
$$

Consequently, \(\beta_{\mathrm{fit}}\) is a weighted average of the squared
residuals \(r_i^2\), with nonnegative weights depending on the hidden-gradient
kernel \((q_i^\top q_j)^2\). In particular,

$$
\min_i r_i^2
\le
\beta_{\mathrm{fit}}
\le
\max_i r_i^2,
$$

and therefore if \(\max_i |r_i|\to 0\), then
\(\beta_{\mathrm{fit}}\to 0\).

### Leverage diagnostic

Define hidden-gradient leverage scores

$$
s_i
:=
\sum_{j=1}^n (q_i^\top q_j)^2.
$$

Then the previous identity can be written as

$$
\beta_{\mathrm{fit}}
=
\frac{\sum_i s_i r_i^2}{\sum_i s_i}.
$$

Let

$$
\bar r^2 := \frac1n\sum_i r_i^2,
\qquad
\bar s := \frac1n\sum_i s_i.
$$

If \(\bar r^2>0\) and \(\bar s>0\), then

$$
\frac{\beta_{\mathrm{fit}}}{\bar r^2}-1
=
\frac1n\sum_i
\left(
\frac{s_i}{\bar s}-1
\right)
\left(
\frac{r_i^2}{\bar r^2}-1
\right).
$$

Therefore

$$
\left|
\frac{\beta_{\mathrm{fit}}}{\bar r^2}-1
\right|
\le
\operatorname{CV}(s)\operatorname{CV}(r^2),
$$

by Cauchy-Schwarz. More exactly, if \(\rho\) denotes the empirical correlation
between \(s_i\) and \(r_i^2\), then

$$
\frac{\beta_{\mathrm{fit}}}{\bar r^2}-1
=
\rho\,\operatorname{CV}(s)\operatorname{CV}(r^2).
$$

This gives a deterministic diagnostic for the empirical observation
\(\beta_{\mathrm{fit}}\approx \bar r^2\). It can happen because the hidden
leverage scores \(s_i\) are nearly flat, because the squared residuals are nearly
flat, or because leverage and squared residuals are weakly correlated. Proving
the last mechanism from training dynamics is the heavier question and is not
assumed in the diagnostic.

### Interpretation

The raw-AGOP conversion

$$
\widetilde G
\approx
\beta_{\mathrm{fit}}G
$$

becomes ill-conditioned near interpolation. Thus the late-regime law should be
written in terms of the residual-weighted AGOP \(\widetilde G\), while the raw
AGOP \(G\) is expected to be most meaningful in an intermediate regime where
\(\beta_{\mathrm{fit}}\) remains bounded away from zero.

---

## High-gain pair closure

The pair diagnostic above is intentionally global on the support of \(T\). The
weighted law only sees the pair error after stationarity compresses it.

Let

$$
A:=QR=U\Sigma V^\top
$$

be the thin SVD, and define

$$
H_X:=V^\top X^\top XV,
\qquad
F_X:=H_X-d_{\mathrm{eff}}I.
$$

Then

$$
T=U\Sigma^2U^\top,
\qquad
S-d_{\mathrm{eff}}T=U\Sigma F_X\Sigma U^\top.
$$

At exact stationarity,

$$
B=-\frac1{\lambda n}U\Sigma V^\top X^\top.
$$

Therefore

$$
B^\top(S-d_{\mathrm{eff}}T)B
=
\frac1{\lambda^2n^2}
G_{\mathrm{stat}}^\top F_XG_{\mathrm{stat}},
\qquad
G_{\mathrm{stat}}:=\Sigma^2V^\top X^\top.
$$

This shows that global pair-isotropy is sufficient but not necessary. What is
needed is scalar closure in high-gain directions of \(G_{\mathrm{stat}}\).

### Conditional theorem

Let \(P_{\mathrm{hi}}\) be an orthogonal projector. Suppose

$$
\|P_{\mathrm{hi}}F_XP_{\mathrm{hi}}\|_{\mathrm{op}}\le \varepsilon,
\qquad
\|(I-P_{\mathrm{hi}})G_{\mathrm{stat}}\|_{\mathrm{op}}\le \delta.
$$

Then at exact stationarity,

$$
\|B^\top(S-d_{\mathrm{eff}}T)B\|_{\mathrm{op}}
\le
\frac1{\lambda^2n^2}
\left[
\varepsilon\|G_{\mathrm{stat}}\|_{\mathrm{op}}^2
+
2\mathcal A_{\mathrm{pair}}\|G_{\mathrm{stat}}\|_{\mathrm{op}}\delta
+
\mathcal A_{\mathrm{pair}}\delta^2
\right].
$$

The proof expands \(G_{\mathrm{stat}}=P_{\mathrm{hi}}G_{\mathrm{stat}}+
(I-P_{\mathrm{hi}})G_{\mathrm{stat}}\) and bounds the high-high, cross, and
low-low terms separately. This theorem is ready to state. What is not proved is
that actual training always produces the high-gain scalar closure assumption.

### Adaptive concentration route

If the learned high-gain subspace \(W=VP_{\mathrm{hi}}\) were fixed or
independent of Gaussian \(X\), then standard Wishart concentration would imply

$$
W^\top X^\top XW\approx dI.
$$

For learned \(W\), the natural conditional theorem is: if the possible high-gain
subspaces lie in a low-complexity class, a net argument gives uniform
concentration up to the covering complexity. This is a useful route, but it
depends on a real training-geometry assumption about the complexity or stability
of the learned high-gain subspace.

More explicitly, let \(X\in\mathbb R^{d\times n}\) have iid \(N(0,1)\)
entries. Let \(\mathcal W\) be a deterministic class of \(n\times k\) matrices
with orthonormal columns. If

$$
\log N(\mathcal W,\|\cdot\|_{\mathrm{op}},\rho)\le \mathcal C,
$$

then a standard net argument gives, with high probability and uniformly over
\(W\in\mathcal W\),

$$
\left\|
\frac1d W^\top X^\top XW-I_k
\right\|_{\mathrm{op}}
\lesssim
\sqrt{\frac{k+\mathcal C+t}{d}}
+
\frac{k+\mathcal C+t}{d}
+
\rho\frac{\|X\|_{\mathrm{op}}^2}{d}.
$$

Consequently, if the learned high-gain subspace \(W_{\mathrm{hi}}\) belongs to
such a class, then

$$
\|P_{\mathrm{hi}}F_XP_{\mathrm{hi}}\|_{\mathrm{op}}
\lesssim
d\left(
\sqrt{\frac{k+\mathcal C+t}{d}}
+
\frac{k+\mathcal C+t}{d}
+
\rho\frac{\|X\|_{\mathrm{op}}^2}{d}
\right)
+
|d-d_{\mathrm{eff}}|.
$$

The proof route is fixed-subspace Wishart concentration, union bound over a
\(\rho\)-net, and the perturbation inequality

$$
\left\|
W^\top X^\top XW-W_0^\top X^\top XW_0
\right\|_{\mathrm{op}}
\le
2\|X\|_{\mathrm{op}}^2\|W-W_0\|_{\mathrm{op}}
+
\|X\|_{\mathrm{op}}^2\|W-W_0\|_{\mathrm{op}}^2.
$$

This is still conditional. It reduces Claim A to proving that training selects
high-gain subspaces with low effective complexity or enough stability.

---

## Conditional beta tracking and the dynamics bridge

The beta identity can be written with normalized leverage

$$
\ell_i:=\frac{s_i}{\bar s},
\qquad
\bar s:=\frac1n\sum_i s_i,
$$

as

$$
\beta_{\mathrm{fit}}=\frac1n\sum_i \ell_i r_i^2.
$$

Thus

$$
\beta_{\mathrm{fit}}-\bar r^2
=
\operatorname{Cov}_n(\ell,r^2).
$$

### Conditional homogeneity

Let \(\mathcal Q=\sigma(q_1,\ldots,q_n)\). If

$$
\mathbb E[r_i^2\mid \mathcal Q]=\mu
\qquad
\text{for all }i,
$$

then

$$
\mathbb E[\beta_{\mathrm{fit}}\mid\mathcal Q]
=
\mathbb E[\bar r^2\mid\mathcal Q]
=
\mu.
$$

With conditional independence and sub-exponential tails, Bernstein gives

$$
|\beta_{\mathrm{fit}}-\bar r^2|
\le
C\nu
\left[
\frac{\operatorname{CV}(\ell)}{\sqrt n}\sqrt t
+
\frac{\|\ell-1\|_\infty}{n}t
\right]
$$

with high probability, conditional on \(\mathcal Q\). This is a clean sufficient
condition for beta tracking.

### Leverage-sensitive damping

The dynamics bridge asks whether high hidden-gradient leverage makes residuals
decay faster. In the simplified diagonal model

$$
\dot r_i=-(\eta_0+\alpha(\ell_i-1))r_i,
\qquad
\alpha>0,
$$

letting \(u_i=r_i^2\) and \(C(t)=\operatorname{Cov}_n(\ell,u(t))\) gives

$$
C'(t)
=
-2\eta_0C(t)
-
2\alpha\frac1n\sum_i(\ell_i-1)^2u_i(t).
$$

Therefore positive covariance is damped in this model. For real two-layer
training, the hard bridge is to connect \(\ell_i\) to NTK damping. The exact
condition is that

$$
r^\top(D_\ell K^{\mathrm{NTK}}+K^{\mathrm{NTK}}D_\ell)r
$$

is positive along the residual trajectory and dominates the leverage drift term.
This is not automatic, since \(D_\ell\) is indefinite and \(\ell_i(t)\) changes
during training. It is the main remaining dynamics assumption.

The actual two-layer NTK has the form

$$
K^{\mathrm{NTK}}_{ij}
=
\phi(Bx_i)^\top\phi(Bx_j)
+
(q_i^\top q_j)(x_i^\top x_j).
$$

By contrast, the beta leverage is

$$
\ell_i
\propto
\sum_j(q_i^\top q_j)^2.
$$

These two objects are related but not identical. A large NTK diagonal is also
not enough, because residual dynamics are coupled through off-diagonal kernel
terms. In addition, \(\ell_i(t)\) moves during training, so differentiating

$$
C(t):=\frac1n\sum_i(\ell_i(t)-1)r_i(t)^2
$$

gives a leverage-drift term:

$$
C'(t)
=
\frac1n\sum_i\dot\ell_i(t)r_i(t)^2
-
\frac1n r(t)^\top(D_\ell K+KD_\ell)r(t).
$$

Thus the full bridge condition is that the kernel damping term is positive and
dominates leverage drift:

$$
r(t)^\top(D_\ell K^{\mathrm{NTK}}+K^{\mathrm{NTK}}D_\ell)r(t)
\gg
\left|\sum_i\dot\ell_i(t)r_i(t)^2\right|.
$$

This condition is plausible in regimes where high leverage corresponds to
directions the model can fit efficiently, but it is not a consequence of
positive semidefiniteness of the NTK alone.

### Perturbative kernel model

A clean conditional model is

$$
\dot r=-(\eta_0 I+\alpha D_\ell+E)r,
\qquad
\alpha>0.
$$

Then

$$
C'(t)
\le
-2\eta_0C(t)
-
\frac{\alpha}{n}\|D_\ell r(t)\|^2
+
\frac{\|E\|_{\mathrm{op}}^2}{\alpha n}\|r(t)\|^2.
$$

Therefore leverage-residual covariance is damped whenever

$$
\|E\|_{\mathrm{op}}^2\|r(t)\|^2
\ll
\alpha^2\|D_\ell r(t)\|^2.
$$

This is the corrected perturbative version of the dynamics bridge. It says that
the diagonal leverage-damping mechanism survives small enough kernel
perturbations.

---

## Immediate coding targets

1. log \(d_{\mathrm{eff}}\)
2. log \(\kappa_{\mathrm{eff}}\)
3. log \(\mathcal A_{\mathrm{pair}}\)
4. log \(\|E_{\mathrm{stat}}\|_{\mathrm{op}}\)
5. log \(\|H^2-\kappa_{\mathrm{eff}}\widetilde G\|_{\mathrm{op}}\)
6. compare theorem RHS against observed weighted-law residual
7. run isotropic baseline + anisotropic ablation
