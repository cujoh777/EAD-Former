# SwinEADFormer Model Card

## Model summary

SwinEADFormer is a Siamese hierarchical Transformer architecture for binary
building change detection from two co-registered optical remote sensing images.
It combines explicit temporal differences with router-gated bidirectional
cross-temporal interaction.

## Intended task

- binary building change detection
- co-registered bitemporal RGB imagery
- high-resolution optical remote sensing scenes

## Input and output

The model receives two RGB tensors representing the same geographical area at
two acquisition times. The study configuration uses 256 by 256 image patches.

| Output | Meaning |
|---|---|
| `pred` | main single-channel change logits |
| `aux` | auxiliary S3 prediction logits |
| `edge` | resized soft router map |
| `sparsity_loss` | mean router activation |

## Main inductive biases

- changed pixels are usually sparse;
- structural boundaries are informative for building changes;
- high-level semantics can suppress irrelevant local responses;
- both temporal interaction directions can provide complementary evidence;
- direct and interaction-derived differences should be jointly represented.

## Evaluation scope reported in the manuscript

The manuscript evaluates the method on LEVIR-CD, WHU-CD, and SYSU-CD. The
datasets are not redistributed in this repository. Dataset characteristics,
official sources, citations, and the study protocol are recorded in
[the dataset documentation](docs/datasets.md).

## Limitations

- The architecture assumes co-registered optical image pairs.
- The reported study does not establish performance on SAR, multispectral, or
  cross-sensor image pairs.
- The router provides soft response modulation and does not reduce attention
  FLOPs.
- This public package does not include the training or evaluation pipeline.
- No formal theoretical guarantee is claimed for the routing mechanism.

## Release status

This is an architecture-only release associated with a manuscript in the
publication process.
