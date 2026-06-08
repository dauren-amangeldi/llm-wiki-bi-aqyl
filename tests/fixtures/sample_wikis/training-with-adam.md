# Training

This page covers the training setup for our models.

## Optimizers

We use the **Adam optimizer** (Adaptive Moment Estimation) for all training runs.
Adam combines momentum and RMSprop to adapt per-parameter learning rates, which
makes it well-suited for sparse gradients and non-stationary objectives.

Typical hyperparameters:
- Learning rate: 3e-4 (Karpathy constant)
- β₁: 0.9 (momentum)
- β₂: 0.999 (RMS decay)
- ε: 1e-8

## Hardware

Training is run on A100 GPUs with mixed-precision (bf16).
