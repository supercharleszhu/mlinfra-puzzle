# Day 04 - TV-layout elementwise operations and predication

Day 04 connects the first half of official
[Notebook 08](../notebooks/08_elementwise_add.ipynb) to a reusable CuTe DSL
kernel pattern. Notebook 08 progresses through:

1. one scalar element per thread;
2. a `(1, 8)` vector per thread;
3. CTA tiling plus a Thread-Value (TV) layout;
4. a generic elementwise operator.

Day 03 covers the first two stages. Day 04 starts at stage three, makes the
thread/value ownership explicit, and adds coordinate-based predication so
partial edge tiles are correct. The elementwise operation is still just
`C = A + B`; the challenge is mapping each thread to the right addresses.

## 1. From thread IDs to TV layouts

The Day 03 vectorized kernel calculates a global thread index manually and
uses it to select one `(1, 8)` vector. That works for one simple mapping, but
the index arithmetic becomes hard to reuse for larger thread tiles.

A TV layout packages ownership as a function:

```text
tv_layout(thread_id, value_id) -> (tile_m, tile_n)
```

`thread_id` identifies a thread in the CTA. `value_id` identifies one value
owned by that thread. The result is a logical coordinate inside one CTA tile.
The data tensor layout then maps that logical coordinate to a memory address.

For FP16 in this lesson:

```python
elts_per_vec = 128 // 16  # 8 FP16 values in 128 bits
thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
val_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
```

There are `4 * 32 = 128` threads. Each thread owns `4 * 8 = 32` values.
Together they cover `128 * 32 = 4096` values, arranged as a
`(4 * 4, 32 * 8) = (16, 256)` CTA tile.

The two returned objects answer different questions:

- `tiler_mn` says **how large one CTA tile is**: `(16, 256)`.
- `tv_layout` says **which `(tid, vid)` owns each coordinate in that tile**.

## 2. What `zipped_divide` does

For an input tensor with logical shape `(M, N)`:

```python
gA = cute.zipped_divide(mA, tiler_mn)
```

conceptually changes the domain to:

```text
((TileM, TileN), (RestM, RestN))
```

The first mode is a coordinate inside a tile. The second mode chooses a tile.
`RestM` and `RestN` use ceiling division, so the last tile may extend beyond
the original tensor. No data is padded or allocated by `zipped_divide`; it is
a layout transformation.

For `(M, N) = (33, 513)` and tile `(16, 256)`:

```text
RestM = ceil(33 / 16)  = 3
RestN = ceil(513 / 256) = 3
```

The launch therefore needs nine CTAs. CTA tiles on the bottom and right edges
contain logical coordinates outside the real tensor and must be predicated.

## 3. The mapping chain inside the kernel

Each CTA first selects its tile:

```python
blk_coord = ((None, None), bidx)
blkA = gA[blk_coord]
```

`None` preserves the complete inner tile while `bidx` selects one outer tile.
The resulting `blkA` maps:

```text
(tile_m, tile_n) -> global-memory address
```

Composition connects TV ownership to that address mapping:

```python
tidfrgA = cute.composition(blkA, tv_layout)
```

Mathematically:

```text
(tid, vid) --tv_layout--> (tile_m, tile_n) --blkA--> address
```

so `tidfrgA` directly maps `(tid, vid)` to an address. Slicing with
`(tidx, None)` fixes the thread ID and preserves every value owned by it:

```python
thrA = tidfrgA[(tidx, None)]  # vid -> address
```

The same mapping is applied to A, B, C, and the coordinate tensor. Identical
partitioning is what keeps data and bounds checks aligned.

## 4. Why the identity tensor is the predicate

`cute.make_identity_tensor((M, N))` is a coordinate-valued tensor:

```text
idC(m, n) = (m, n)
```

It does not allocate an `(M, N)` data buffer. After the same
`zipped_divide`, CTA selection, composition, and thread selection as C,
`thrCrd[vid]` gives the global `(m, n)` coordinate associated with
`thrC[vid]`.

The predicate is therefore:

```python
frgPred[vid] = cute.elem_less(thrCrd[vid], shape)
```

Both coordinates must be inside the original shape. Apply the predicate to
loads and stores so invalid edge lanes never access memory.

## 5. Puzzle TODO sequence

Implement one ownership transition at a time in `puzzle.py` or `day04.ipynb`:

1. **CTA tile:** slice A, B, C, and coordinates with `blk_coord`.
2. **TV composition:** compose every CTA-local tensor with `tv_layout`.
3. **Thread slice:** fix `tidx` and preserve the value mode.
4. **Predicate:** compare each coordinate against `(M, N)`.
5. **Elementwise body:** predicated load, `A + B`, predicated store.

At each stage, ask what the current tensor maps **from** and **to**. If that
sentence is unclear, print its type before writing the next line.

## 6. Correctness before performance

Use both a divisible shape and a partial-tile shape:

```bash
python puzzle.py --M 1024 --N 1024
python puzzle.py --M 1023 --N 1025
python solution.py --M 1023 --N 1025
```

The second shape is the important one: it exercises both row and column
predicates. Always synchronize before checking because kernel launches are
asynchronous.

## 7. Benchmarking

Compile once, then benchmark only the compiled callable:

```python
try:
    import cutlass.testing as testing       # CuTe DSL 4.6+
except ModuleNotFoundError as error:
    if error.name != "cutlass.testing":
        raise
    from cutlass.cute import testing        # older Notebook 08 API

compiled = cute.compile(elementwise_add, a_, b_, c_)
avg_time_us = testing.benchmark(
    compiled,
    kernel_arguments=testing.JitArguments(a_, b_, c_),
    warmup_iterations=5,
    iterations=100,
)
```

Do not include JIT compilation or tensor allocation in kernel timing.
Elementwise FP16 add transfers two inputs and one output:

```text
bytes = M * N * (2 reads + 1 write) * 2 bytes = 6 * M * N
GB/s  = bytes / (time_us * 1000)
```

This is **effective algorithmic bandwidth**. It counts the required tensor
traffic, not every physical transaction or cache effect.

```bash
python solution.py --M 16384 --N 8192 --benchmark
python solution.py --M 16384 --N 8192 --benchmark \
  --warmup 10 --iterations 200
```

Use a large shape for stable timing. Keep the odd shape for correctness, not
for peak-bandwidth comparison.

## What you should know afterward

- A tiler defines the CTA tile shape; a TV layout defines ownership inside it.
- `zipped_divide` separates intra-tile and inter-tile coordinates.
- Composition turns `(tid, vid)` into an address without manual index math.
- An identity tensor carries coordinates through the exact same partitioning.
- Elementwise kernels are usually memory-bound, so report time and effective
  bandwidth after proving correctness.
- Notebook 08's generic elementwise operator changes step 5; the tiling,
  ownership, and predicate skeleton stay the same.
