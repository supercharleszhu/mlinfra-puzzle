# Day 05 - Layer Normalization

**Official topic:** [Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html).

LayerNorm combines row reductions with elementwise affine work. The forward
kernel computes mean and variance while a row is resident on chip. Backward
uses two scalar reductions per row for `dx`, then a second kernel reduces
`dw` and `db` across rows.

## Challenge

1. Fuse row mean, variance, normalization, weight, and bias in the forward
   kernel.
2. Save mean and reciprocal standard deviation for backward.
3. Derive `dx` from `w*dy`, `mean(xhat*w*dy)`, and `mean(w*dy)`.
4. Reduce `dy*xhat` and `dy` across rows for weight and bias gradients.
5. Wrap the three kernels in `torch.autograd.Function`.
6. Compare output and all three gradients with `torch.nn.functional.layer_norm`.

The solution rejects feature rows larger than 64KB because the simple fused
strategy retains a power-of-two row in program-local storage. The official
tutorial uses locked partial buffers before a final reduction; this curriculum
uses independent column blocks for a race-free reduction that is easier to
inspect while preserving both backward stages.
