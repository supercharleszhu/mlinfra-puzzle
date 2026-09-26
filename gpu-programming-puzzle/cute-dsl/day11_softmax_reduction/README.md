# Day 11 - Softmax and Reductions in CuTe DSL

This day implements stable row-wise softmax and compares CuTe's explicit
reduction hierarchy with Triton's program-level reductions.

```text
max_i x_i
e_i = exp(x_i - max)
sum_i e_i
y_i = e_i / sum
```

Subtracting the row maximum is required for numerical stability. Without it,
large positive inputs can overflow `exp`, while very negative inputs can
underflow every numerator to zero.

## Reduction has two levels

The official CuTe TensorSSA notebook introduces:

```python
local_max = values.reduce(
    cute.ReductionOp.MAX,
    -cutlass.Float32.inf,
    reduction_profile=0,
)
```

`TensorSSA.reduce` only combines values already owned by one thread. It lowers
a register-resident vector reduction; it does not communicate with other
lanes. A row distributed across a warp therefore needs a second level:

```text
thread register values
    | TensorSSA.reduce
thread partial
    | shuffle_sync_bfly at offsets 16, 8, 4, 2, 1
warp result replicated to all lanes
```

This lesson uses one 32-thread warp per row. Lane `l` owns columns:

```text
l, l + 32, l + 64, ...
```

At each item index, adjacent lanes access adjacent columns, preserving
coalesced global-memory traffic. An irregular row such as 1003 columns gives
each lane 32 register slots; slots beyond column 1002 are initialized to
negative infinity. They cannot affect the maximum, and `exp(-inf)` contributes
zero to the sum.

## Challenge

Implement three pieces in `puzzle.py`:

1. `warp_reduce_max`: five butterfly shuffles combined with `cute.arch.fmax`.
2. `warp_reduce_sum`: the same shuffle tree combined with addition.
3. `WarpSoftmax.kernel`: load the local register fragment, perform local and
   warp max reductions, exponentiate, perform local and warp sum reductions,
   normalize, and store valid columns.

The supplied tests cover:

- one value and widths smaller than one warp;
- exact powers of two;
- multiple values per lane;
- irregular width 1003;
- large positive and negative logits;
- row sums equal to one.

Run the standalone files:

```bash
python day11_softmax_reduction/puzzle.py
python day11_softmax_reduction/solution.py
python day11_softmax_reduction/solution.py --benchmark
```

## Why one warp per row?

CuTe DSL 4.6 exposes register reductions and warp shuffle primitives, but no
single generic CTA-wide reduction call. A multi-warp row would need:

1. one local reduction per thread;
2. one reduction per warp;
3. lane 0 of each warp writing a shared-memory partial;
4. a CTA barrier;
5. a final warp reducing those partials;
6. shared-memory broadcast back to every warp.

That is a useful follow-up, but it obscures the core reduction semantics. The
one-warp design is correct for widths up to this lesson's 4096-column limit
and makes the hierarchy visible without shared-memory synchronization.

The tradeoff is register pressure: width 1024 means 32 values per lane. Triton
may choose more warps and a different lowering for the same row, so benchmark
results are part of the lesson rather than an assumed conclusion.

## CuTe versus Triton

The notebook contains a matched Triton kernel and benchmarks both
implementations with `triton.testing.do_bench`.

| Concern | CuTe Day 11 | Triton Day 13 |
| --- | --- | --- |
| Thread ownership | Explicit lane-strided register tensor | Logical vector from `tl.arange` |
| Local reduction | `TensorSSA.reduce` | Hidden inside `tl.max` / `tl.sum` lowering |
| Cross-lane reduction | Explicit butterfly shuffles | Compiler-generated |
| Padding identity | Register slots initialized to `-inf` | Masked load uses `other=-inf` |
| Launch mapping | Exactly one warp per row | One program per row, configurable warps |
| Specialization | `WarpSoftmax(n_cols)` | `BLOCK_SIZE: tl.constexpr` |

The comparison uses the same FP32 input tensors and shapes:
128, 512, 1003, and 1024 columns. Effective bandwidth counts one input read
and one output write:

```text
GB/s = 2 * rows * columns * sizeof(float) / elapsed_seconds / 1e9
```

This is a traffic proxy, not a claim that softmax performs only memory work.
The exponential and two reductions also consume compute and synchronization
resources.

## API notes

- `reduction_profile=0` reduces the only mode of the 1-D TensorSSA.
- Identities must match the operation: `-inf` for maximum, zero for addition.
- Butterfly reduction works because the group size is a power of two and all
  32 lanes participate.
- Every lane receives the final value, so scalar subtraction and division
  broadcast naturally across the local TensorSSA.
- Shape is specialized at compile time because register tensor dimensions
  must be static.

The reduction pattern follows NVIDIA CUTLASS's official CuTe DSL TensorSSA
notebook and the warp-shuffle reduction used in the official Ampere Flash
Attention example.
