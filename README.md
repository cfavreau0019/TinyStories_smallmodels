## Representation convention

For sequence sample $s$, layer $\ell$, and token $i$, let

$$
x_{s,\ell,i} \in \mathbb{R}^d
$$

denote the residual-stream representation.

After RMS normalization,

$$
u_{s,\ell,i}
=
\frac{
x_{s,\ell,i}
}{
\sqrt{
\frac{1}{d}
\sum_k x_{s,\ell,i,k}^2
}
}.
$$

Define

$$
\hat u_{s,\ell,i}
=
\frac{u_{s,\ell,i}}{\sqrt d}.
$$

Then

$$
\hat u_{s,\ell,i} \in S^{d-1}.
$$

The working interpretation is:

- **Token** → oscillator
- **Hidden dimensions** → coordinates of the oscillator state
- **RMSNorm** → hyperspherical constraint
- **Attention matrix** → coupling matrix
- **Training** → evolution of learned synchronization structure
- **Layer depth** → sequence of distinct representation geometries

---

## Hyperspherical synchronization spectrum

Let $Y_{n,a}$ denote hyperspherical harmonics of degree $n$.

The empirical distribution of normalized token states can be expanded as

$$
\rho(u)
=
\sum_{n=0}^{\infty}
\sum_a
c_{n,a}Y_{n,a}(u).
$$

The corresponding coefficients are

$$
c_{n,a}
=
\frac{1}{T}
\sum_i
Y_{n,a}^{*}(\hat u_i).
$$

Define the spectral power at degree $n$ as

$$
P_n
=
\sum_a
|c_{n,a}|^2.
$$

Rather than explicitly constructing every hyperspherical harmonic, the power can be estimated using the hyperspherical addition theorem:

$$
P_n
\propto
\frac{1}{T^2}
\sum_{i,j}
C_n^{(\alpha)}
\left(
\hat u_i^\top \hat u_j
\right),
$$

where

$$
\alpha = \frac{d-2}{2}.
$$

The initial experiments use low harmonic degrees,

$$
n = 1,\ldots,8.
$$

---

## Effective synchronization dimension

Normalize the harmonic powers:

$$
p_n
=
\frac{P_n}
{\sum_r P_r}.
$$

Then define

$$
D_{\mathrm{sync}}
=
\frac{1}{\sum_n p_n^2}.
$$

Equivalently,

$$
D_{\mathrm{sync}}
=
\frac{
\left(\sum_n P_n\right)^2
}{
\sum_n P_n^2
}.
$$

A low value of $D_{\mathrm{sync}}$ indicates that synchronization structure is concentrated into a small number of collective modes. A larger value indicates a more distributed synchronization spectrum.

---

## Attention as coupling

For attention head $h$,

$$
A_{ij}^{(\ell,h)}
=
\operatorname{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt{d_h}}
\right).
$$

The working hypothesis is that the attention matrix can be treated as an empirical heterogeneous coupling matrix:

$$
K_{ij} \sim A_{ij}.
$$

A hyperspherical Kuramoto-like interaction would then have the form

$$
\dot u_i
=
\Omega_i u_i
+
\sum_j
K_{ij} P_{u_i}u_j,
$$

where the tangent-space projector is

$$
P_{u_i}
=
I-u_i u_i^\top.
$$

The experiments ask whether the learned attention coupling structure predicts changes in the synchronization spectrum.

---

## RMSNorm and ellipsoidal geometry

Before learned RMSNorm scaling,

$$
u \in S^{d-1}.
$$

After elementwise scaling,

$$
y = \Gamma u,
$$

with

$$
\Gamma
=
\operatorname{diag}
(\gamma_1,\ldots,\gamma_d).
$$

The scaled representation satisfies

$$
y^\top \Gamma^{-2} y
=
\text{constant},
$$

which defines an ellipsoid.

This motivates comparing synchronization measured on the normalized hypersphere with synchronization after learned anisotropic scaling.

---

## Dimensionality-reduction hypothesis

The long-term objective is

$$
H
\rightarrow
\text{synchronization transform}
\rightarrow
\text{top synchronization modes}
\rightarrow
\hat H.
$$

The reconstructed representation can then be compared with PCA, truncated SVD, random projections, and autoencoder baselines.

A central question is whether the synchronization dimension

$$
D_{\mathrm{sync}}^{(\ell)}
$$

predicts the minimum representation dimension required to preserve model behavior,

$$
d_{95}^{(\ell)}.
$$

The central project hypothesis is therefore:

> High-dimensional transformer representations may admit a lower-dimensional description in terms of collective synchronization modes.