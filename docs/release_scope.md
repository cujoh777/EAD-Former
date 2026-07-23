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
- sanitized training and evaluation entry-point references

## Not included

- baseline implementations; obtain them from the official repositories
  maintained by their original authors
- dataset downloading or preprocessing
- dataset split manifests
- the internal dataset loader, boundary metrics, and utility package referenced
  by the public entry-point files
- a complete executable training, validation, and evaluation pipeline
- internal experiment scheduling
- environment installation files
- pretrained weights and checkpoints
- logs and numerical result files
- visualization and paper-generation scripts
- manuscript and submission materials

This repository is a limited public release. The entry-point files document
selected interfaces but are not standalone. The repository is not an
end-to-end reproducibility package.
