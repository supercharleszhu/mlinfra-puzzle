# Day 05 - Matrix Transpose: Naive to Tiled

**Goal:** transpose a row-major `(M, N)` matrix into a row-major `(N, M)`
matrix, first with direct global-memory copies and then with a padded
shared-memory tile that keeps both global-memory directions coalesced.

This challenge is an original CuTe DSL adaptation of the progression explained
in Lei Mao's
[CuTe Matrix Transpose](https://leimao.github.io/article/CuTe-Matrix-Transpose/).
The article implements C++ CuTe variants; this lesson recreates the progression
in Python CuTe DSL with two challenges:

1. **Challenge A:** write naive direct kernels with either coalesced reads or
   coalesced writes.
2. **Challenge B:** stage through shared memory using the same partition/copy
   workflow, then remove bank conflicts with padding and swizzling.

Start with [`day05.ipynb`](day05.ipynb). It derives the address mappings,
implements the two direct mappings, models shared-memory banks, and then walks
through the nine shared-kernel TODOs. A benchmark block immediately after
Challenge A compares the two naive kernels before shared-memory optimization.

## Challenge A: naive direct transpose

A direct transpose cannot coalesce both global-memory operations:

| Variant | Adjacent lanes access | Cost |
| --- | --- | --- |
| Coalesced read | `A[row, col + lane]` | writes different rows of `B` |
| Coalesced write | `B[row, col + lane]` | reads different rows of `A` |

Fill five Challenge A TODOs in `puzzle.py`:

1. Convert the linear thread ID into an `(8, 32)` logical layout.
2. Map threads for coalesced input reads and strided output writes.
3. Reverse ownership for strided input reads and coalesced output writes.
4. Launch the coalesced-read kernel over `32 x 32` input tiles.
5. Launch the coalesced-write kernel over the same tile grid.

The notebook's Challenge A cell contains the recovered working implementation
from the Jupyter execution history. Its persistent source is the Challenge A
portion of `solution.py`, so regenerating `day05.ipynb` preserves it.

Both variants use edge predicates and support arbitrary positive matrix shapes.
The reference solutions use `cute.copy` with layout aliases rather than manual
scalar assignments:

- Coalesced read creates `mBT` with logical shape `(M,N)` and stride `(1,M)`,
  so `mBT[m,n]` aliases row-major `mB[n,m]`.
- Coalesced write creates `mAT` with logical shape `(N,M)` and stride `(1,N)`,
  so `mAT[n,m]` aliases row-major `mA[m,n]`.

Each alias shares a logical domain with its copy partner. The same
`local_partition`, coordinate predicate, register fragment, and predicated
`cute.copy` calls therefore work in both directions.

```bash
python puzzle.py --M 1024 --N 1536 --mode read
python solution.py --M 1003 --N 1507 --mode read
python solution.py --M 8192 --N 8192 --mode write --benchmark
python solution.py --M 8192 --N 8192 --mode all --benchmark
```

## Challenge B: shared-memory transpose

The shared kernel decouples the two global mappings:

```text
row-major A tile
    | coalesced global reads
    v
shared tile plus transposed layout alias
    | unpadded, padded, or S<5,0,5> swizzled mapping
    v
row-major B tile
      coalesced global writes
```

A block has 256 threads arranged logically as `(8, 32)`. Each thread handles
four values, covering one `32 x 32` tile. Edge predicates make arbitrary
positive matrix shapes valid.

Challenge B does not calculate global coordinates manually. It follows
Challenge A:

```text
partition -> coordinate predicate -> register fragment -> cute.copy
```

Section 5 of the notebook now pauses after every shared-memory concept:

| Subsection | Immediate challenge |
| --- | --- |
| 5.1 Shared staging | Implement the real unpadded CuTe shared-memory kernel and launch. |
| 5.2 Padding | Derive stride 33 and extend the CuTe kernel with padded layouts. |
| 5.3 Swizzling | Derive `S<5,0,5>` and extend the CuTe kernel with composed layouts. |

The notebook builds one kernel progressively instead of postponing everything
to a monolithic capstone. `solution.py` contains all completed stages plus the
bank, padding, and swizzle model helpers. Every Section 5 challenge cell ends
with validation: the CuTe stages run both a divisible `64 x 96` case and an
irregular `1003 x 1507` case, while the address-model cells assert the expected
bank mapping and allocation size. Each subsection also benchmarks a
`4096 x 4096` transpose and reports effective GB/s plus utilization relative to
the H100 SXM theoretical 3,350 GB/s peak.

`sA` and `sAT` share one allocation but use row-major and transposed layouts.
The recovered Challenge 5.1 implementation completes the first seven steps:

1. Select the CTA's input/output data and coordinate tiles.
2. Allocate `sA` and create the transposed alias `sAT`.
3. Partition and copy global A to shared memory under the input predicate.
4. Synchronize after filling shared memory.
5. Partition and copy `sAT` to global B under the output predicate.
6. Tile all tensors and launch the common shared kernel.
7. Launch the naive unpadded layout.

The recovered completed notebook also adds padded layouts and then swizzled
layouts over the same kernel. These implementations are persisted in
`puzzle.py`, so regenerating the notebook does not erase them.

```bash
python puzzle.py --M 1024 --N 1536
python solution.py --M 1003 --N 1507 --mode all
python solution.py --M 8192 --N 8192 --mode all --benchmark
```

Compare all five variants on the same large shape. The naive kernels establish
why remapping threads alone only moves the uncoalesced access. The three shared
variants then isolate the cost of stride-32 bank conflicts and the two remedies:

- `shared/unpadded`: layouts `(32,1)` and `(1,32)`.
- `shared/padded`: layouts `(33,1)` and `(1,33)`.
- `shared/swizzled`: `S<5,0,5>` composed with both unpadded base layouts.

CuTe layout offsets are measured in elements. For a `32 x 32` Float32 tile,
the low five offset bits select the column and the next five select the row.
`S<5,0,5>` XORs those row bits into the column bits, distributing a transposed
warp access across all 32 banks without adding padding.

`puzzle.py` is the only challenge source and `solution.py` is the only
reference source. Both cover the two naive kernels and all three shared-memory
variants.

Effective bandwidth counts one input read and one output write:

```text
bytes = 2 * M * N * sizeof(Float32)
GB/s  = bytes / (time_us * 1000)
```

The notebook includes reusable timing helpers for all three transpose variants,
reports the percentage of the H100 SXM theoretical 3,350 GB/s peak, and
measures a large PyTorch device-copy baseline as a practical HBM ceiling. The
copy buffers are intentionally larger than L2, and CUDA events exclude Python
setup time.

## Why padding works

For Float32, shared memory has 32 banks and consecutive words map to
consecutive banks. With an unpadded row stride of 32, a warp reading one column
touches addresses:

```text
0, 32, 64, ... -> banks 0, 0, 0, ...
```

That is a 32-way conflict. A stride of 33 changes the bank sequence to:

```text
0, 33, 66, ... -> banks 0, 1, 2, ...
```

The extra physical column is never a logical matrix element. It only changes
the address mapping. Swizzling is another solution and is discussed as an
extension in the notebook.
