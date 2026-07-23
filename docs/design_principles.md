# Design Principles

SwinEADFormer is organized around task-specific inductive biases for building
change detection.

## Sparse changes require selective interaction

Unchanged pixels dominate common building change detection datasets. Uniform
cross-temporal interaction can propagate responses caused by shadows,
illumination variation, texture differences, or slight misregistration. The
router therefore modulates interaction responses using evidence derived from
the current image pair.

## Region and boundary evidence are complementary

Building changes require region discrimination and boundary localization. A
single semantic cue may lose local structure. A single edge cue cannot
distinguish structural change from unchanged high-frequency texture. The router
combines edge, region, and semantic cues because they describe different
properties of a potential change.

## Temporal interaction is directional

Query and reference features play different roles in cross-attention. The model
therefore computes both temporal directions and retains their discrepancy as a
separate evidence source.

## Direct and interaction-derived differences should coexist

Attention-derived representations can model context, but they may suppress or
alter direct temporal evidence. Phi fusion combines the directional interaction
discrepancy with learned and direct S3 differences.

## Interaction should occur at a balanced feature level

Shallow features contain spatial detail but respond strongly to local
appearance variation. Deep features provide semantic stability but have low
spatial resolution. S3 provides a practical balance for building-level
interaction. S4 supplies context, while S1 and S2 support spatial recovery.

These principles motivate the architecture. They are empirical design
assumptions rather than formal theoretical guarantees.
