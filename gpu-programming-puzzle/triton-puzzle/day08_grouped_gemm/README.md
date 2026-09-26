# Day 08 - Grouped GEMM

**Official topic:** [Grouped GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html).

Mixture-of-experts and ragged workloads contain many independent GEMMs with
different dimensions. Launching one kernel per problem adds overhead and can
underfill the GPU. Grouped GEMM uploads pointer, shape, and leading-dimension
arrays, then lets a fixed number of persistent programs schedule all tiles.

## Challenge

1. Pack matrix addresses into device int64 tensors.
2. Pack `(M,N,K)` and leading dimensions into device metadata.
3. Convert loaded integer addresses to typed Triton pointers.
4. Map a global tile index into the owning problem and local tile.
5. Advance each program by `NUM_SM` until all assigned tiles are complete.
6. Mask ragged M, N, and K boundaries and compare every output with PyTorch.

The kernel uses one clear tile configuration rather than autotuning across
groups. The official extension adds a Hopper TMA descriptor path; after this
challenge, compare pointer loads with descriptor loads while preserving the
single-launch device scheduler.
