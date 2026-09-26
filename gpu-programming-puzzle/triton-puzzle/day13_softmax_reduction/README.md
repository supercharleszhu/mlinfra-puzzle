# Day 13 - Softmax Reduction Benchmark

**Original extension:** matched comparison with
[`../../cute-dsl/day11_softmax_reduction/`](../../cute-dsl/day11_softmax_reduction/).

Day 02 follows Triton's official fused-softmax tutorial. This new day revisits
the operation as a cross-framework reduction lab: it adds adversarial
correctness tests, a reusable output-buffer launcher, matched shapes, and
benchmarks against PyTorch and CuTe DSL.

## Challenge

Implement stable row-wise softmax for a contiguous FP32 matrix:

```text
row_max = max(x)
numerator = exp(x - row_max)
output = numerator / sum(numerator)
```

1. Assign one Triton program to each row.
2. Round `n_cols` up to a power of two for `tl.arange`.
3. Mask padded columns and load them as negative infinity.
4. Use `tl.max` and `tl.sum` along the program's vector axis.
5. Store only valid columns.
6. Choose `num_warps` from the padded block size.

`launch_softmax(x, y)` writes into a preallocated output so the benchmark does
not repeatedly allocate the Triton output.

## Tests

`test_softmax()` checks:

- `1x1` and rows shorter than a warp;
- 128- and 512-column regular rows;
- irregular width 1003;
- large positive and negative logits;
- elementwise agreement with `torch.softmax`;
- every output row sums to one.

The max subtraction is observable in the extreme-value case. An unstable
implementation will overflow or produce NaNs.

## Benchmark

```bash
python day13_softmax_reduction/solution.py --benchmark
```

The benchmark uses the same FP32 column counts as CuTe Day 11: 128, 512, 1003,
and 1024. It reports Triton and PyTorch time plus effective bandwidth:

```text
GB/s = 2 * rows * columns * sizeof(float) / elapsed_seconds / 1e9
```

The two represents one input read and one output write. This is a useful
traffic-normalized metric, but softmax also performs exponentials and two
reductions, so it is not a pure copy-bandwidth measurement.

## Compare with CuTe

| Concern | Triton | CuTe DSL |
| --- | --- | --- |
| Work mapping | One program owns a logical row vector | One explicit warp owns a row |
| Local values | `tl.arange` vector | Per-lane register tensor |
| Reduction | `tl.max`, `tl.sum` | `TensorSSA.reduce` plus warp shuffles |
| Tail handling | Masked load/store | `-inf` register initialization plus guarded accesses |
| Parallelism | Compiler lowers program reduction | Programmer specifies lane communication |
| Tuning knob | `BLOCK_SIZE`, `num_warps` | Values per lane, warps per row |

Triton's source is shorter because the compiler owns the cross-lane reduction
lowering. CuTe exposes ownership and communication directly, making it easier
to teach exactly where the reduction occurs and to replace the one-warp design
with a custom multi-warp shared-memory tree.

Use the CuTe Day 11 notebook for a side-by-side benchmark under one timing
harness. Compare measured output rather than assuming either abstraction is
faster: row width, register pressure, compiler version, and GPU architecture
all affect the result.
