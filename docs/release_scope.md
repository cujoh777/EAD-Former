# Public Release Scope

## Included

- SwinEADFormer model definition
- explicit S3 difference construction
- edge-region-semantic router
- router-gated cross-temporal attention
- bidirectional EAD interaction
- Phi fusion
- multi-scale decoder
- architecture documentation
- architecture metadata
- dataset loader
- loss and metric utilities
- boundary metrics
- training and evaluation entry points
- Python dependency list

## Not included

- baseline implementations; obtain them from the official repositories
  maintained by their original authors
- original datasets and derived image patches
- dataset downloading or preprocessing
- dataset split manifests
- internal experiment scheduling
- trained model checkpoints used in the manuscript; these will be added upon
  acceptance
- logs and numerical result files
- visualization and paper-generation scripts
- manuscript and submission materials

This repository provides the SwinEADFormer model, dataset loader, loss and
metric utilities, boundary evaluation code, dependency list, and public
training and evaluation entry points. It does not redistribute the original
datasets, baseline implementations, split manifests, or trained checkpoints.
The trained checkpoints used in the manuscript will be added upon acceptance.
