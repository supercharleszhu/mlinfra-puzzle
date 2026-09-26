# Day 03 - Matrix Multiplication

**Official topic:** [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).

This challenge turns a one-dimensional program ID into a two-dimensional
output tile, stages K-sized fragments, and accumulates with `tl.dot`. Grouped
ordering visits several neighboring M tiles before advancing N, increasing the
chance that A tiles remain useful in cache.

## Challenge

1. Derive grouped `(pid_m, pid_n)` coordinates from the linear program ID.
2. Build pointer matrices from row/column strides.
3. Loop over K in fixed-size blocks and mask the final partial K block.
4. Accumulate in FP32 with `tl.dot` and cast only in the epilogue.
5. Mask M/N edges and validate rectangular, non-divisible shapes.

The solution deliberately uses one clear configuration before introducing
autotuning. Once it is correct, wrap the kernel with `triton.autotune` and
compare multiple block shapes, warp counts, and pipeline stage counts. Measure
TFLOP/s as `2*M*N*K / time`, but preserve the irregular-shape test so tuning
cannot hide a boundary bug.

This tiled kernel becomes the baseline for grouped and persistent GEMM later.
