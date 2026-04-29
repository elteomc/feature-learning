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

## Immediate coding targets

1. log \(d_{\mathrm{eff}}\)
2. log \(\kappa_{\mathrm{eff}}\)
3. log \(\mathcal A_{\mathrm{pair}}\)
4. log \(\|E_{\mathrm{stat}}\|_{\mathrm{op}}\)
5. log \(\|H^2-\kappa_{\mathrm{eff}}\widetilde G\|_{\mathrm{op}}\)
6. compare theorem RHS against observed weighted-law residual
7. run isotropic baseline + anisotropic ablation
