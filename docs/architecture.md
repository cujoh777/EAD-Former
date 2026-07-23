# SwinEADFormer Architecture

## Processing stages

SwinEADFormer receives two co-registered RGB images and uses a shared
Swin-Tiny encoder to extract four hierarchical feature stages.

Stage 3 is used for explicit difference construction and cross-temporal
interaction. Stage 4 supplies high-level semantic context. Stage 1 and Stage 2
provide shallow difference skips for spatial recovery.

## Explicit S3 difference

The mid-level difference representation combines temporal features, their
absolute difference, and their element-wise product:

$$
D_m=\phi_d\left(
[F_1^3,F_2^3,|F_1^3-F_2^3|,F_1^3\odot F_2^3]
\right).
$$

## Edge-region-semantic router

The three routing cues are:

$$
U_e=2\operatorname{Norm}(\operatorname{Sobel}(\psi(D_m)))-1,
$$

$$
U_r=\phi_r(D_m),
$$

$$
U_s=\operatorname{Up}(\phi_s(D_h)),
\qquad D_h=|F_1^4-F_2^4|.
$$

They form a single-channel soft gate:

$$
R=\sigma(U_e+U_r+U_s+b).
$$

The edge cue represents local structural transitions. The region cue represents
compact change evidence. The semantic cue supplies high-level context.

## Router-gated interaction

The router gates the output of cross-temporal attention before residual
addition:

$$
\widetilde{X}
=
\operatorname{Attn}
\left(
\operatorname{GN}(X),
\operatorname{GN}(Y)
\right)
\odot R.
$$

All spatial tokens remain in the attention computation. The router changes the
spatial response amplitude.

## Bidirectional discrepancy

The model processes both temporal directions:

$$
O_{12}=\operatorname{EAD}^{(2)}(F_1^3,F_2^3,R),
$$

$$
O_{21}=\operatorname{EAD}^{(2)}(F_2^3,F_1^3,R).
$$

The absolute directional discrepancy is:

$$
D_{\mathrm{int}}=|O_{12}-O_{21}|.
$$

## Phi fusion

The final S3 change representation combines interaction and direct difference
evidence:

$$
Z=\Phi\left(
[|O_{12}-O_{21}|,D_m,|F_1^3-F_2^3|]
\right).
$$

The decoder then fuses \(Z\) with shallow absolute-difference features and
produces the final change logits.
