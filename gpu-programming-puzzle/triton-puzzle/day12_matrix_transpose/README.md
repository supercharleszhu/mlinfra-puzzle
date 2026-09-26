# Day 12 - Matrix Transpose

**Original extension:** compare with
[`../../cute-dsl/day05_matrix_transpose/`](../../cute-dsl/day05_matrix_transpose/).

Transpose exposes the same address tradeoff in every GPU language: swapping
logical coordinates can make one global-memory direction strided. Triton
expresses a full two-dimensional tile, loads it with row-major offsets, and
stores the same register tile through swapped output strides.

## Challenge

1. Launch a two-dimensional grid over 32-by-32 input tiles.
2. Build row and column offset vectors from the two program IDs.
3. Form a rectangular boundary mask for irregular M and N.
4. Load `x[row, col]` and store it at `y[col, row]`.
5. Validate square, rectangular, and non-divisible shapes.
6. Benchmark effective bandwidth as one full read plus one full write.

Unlike the explicit CUDA/CuTe shared-memory lesson, Triton owns register and
shared-memory lowering decisions. Inspect generated PTX and profile global
sectors to determine whether the compiler realizes efficient transactions;
the compact source alone does not prove coalescing. Compare the result with
CuTe's unpadded, padded, and swizzled kernels to understand which scheduling
details Triton hides and which performance questions remain visible.
