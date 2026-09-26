# Day 06 - Fused Attention

**Official topic:** [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).

Attention combines a score matrix, stable softmax, and a value projection.
The official lesson develops blockwise FlashAttention v2 with descriptor,
causal, FP8, backward, and warp-specialized paths. This compact challenge keeps
an entire small sequence in one Triton program so the same forward and
backward equations remain visible before introducing online block updates.

## Challenge

1. Compute scaled `Q @ K.T` for one batch-head per program.
2. Apply the causal triangle before stable row softmax.
3. Multiply probabilities by V without storing the probability matrix.
4. Recompute probabilities in backward.
5. Derive `dV`, `dP`, `dS`, `dQ`, and `dK`.
6. Compare both causal modes and all gradients with PyTorch SDPA.

The educational kernel intentionally limits sequence and head dimensions to
64. It demonstrates fusion and recomputation, not production FlashAttention
tiling. Use the official tutorial next to replace full-sequence residency with
online max/sum updates, descriptor loads, and separate dQ/dK/dV tiles for long
contexts.
