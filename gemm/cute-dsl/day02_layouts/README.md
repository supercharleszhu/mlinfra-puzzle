# Day 02 — CuTe DSL Layout Algebra

**Goal:** internalize the four operations every later DSL kernel will use:
constructing layouts, `zipped_divide`, Thread-Value (TV) layouts, and identity
tensors for predication.

The notebook begins with NVIDIA's comprehensive
[`05_layout_algebra.ipynb`](../notebooks/05_layout_algebra.ipynb) tutorial,
covering composition, complement, divide, and product operations. A transition
then leads into the original Day 02 exercises.

The notebook also vendors all 11 diagrams referenced by NVIDIA's
`02_layout_algebra.md` under `images/layout-algebra/` and places them beside
the corresponding composition, complement, division, and product sections.

## Background

A **Layout** = (Shape, Stride) — a pure function from coordinates → linear
offset. Layouts are the only piece of math you need to write efficient CuTe
kernels. The DSL exposes the same algebra as the C++ side, but at JIT compile
time you can just `print(layout)` and read what you built.

### Building layouts

```python
L = cute.make_layout((4, 8), stride=(8, 1))   # row-major, shape (4, 8)
cute.size(L)    # 32   number of elements
cute.cosize(L)  # 32   one past the largest offset
```

### Zipped divide — splitting a tensor into tiles

```python
A   = cute.make_layout((8, 16), stride=(16, 1))   # (M, N) row-major
gA  = cute.zipped_divide(A, (2, 4))                # ((2,4),(4,4))
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       (TileM, TileN), (NumTilesM, NumTilesN)
```

Inside a kernel you then write `gA[(None, None), bidx]` to grab one tile.

### Thread-Value (TV) layout

`(tid, vid) -> coord-in-tile`. This is *the* CuTe idiom. You build two pieces:

```python
thr_layout = cute.make_layout((4, 32), stride=(32, 1))  # 128 threads
val_layout = cute.make_layout((4, 4),  stride=(4, 1))   # 16 vals/thread
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
```

`tiler_mn` is the per-CTA tile size; `tv_layout` maps `(thread_index,
value_index)` to a logical `(m, n)` coordinate within that tile.

### Identity tensor

`make_identity_tensor(shape)` is a tensor where `T(c) = c`. You'll use it on
Day 04 to build the predicate mask for OOB writes.

## Task

Fill in the four `# TODO`s in `puzzle.py`. Each prints its results and asserts
a known size / shape — if it runs to completion, you got it right.

## Run

```bash
python puzzle.py     # should NotImplementedError
python solution.py   # reference
```

## What you should walk away knowing

- A `Layout` is a function — and `size()` / `cosize()` tell you its domain / range.
- `zipped_divide` is how you go from `(M, N)` to `((TileM, TileN), (m', n'))`,
  so each block can slice with `gA[((None, None), bidx)]`.
- TV layouts decouple **which thread does what** from **which element it
  touches** — the same `(thr, val) -> coord` table is reused by every CuTe copy
  / MMA pattern in the codebase.
- An identity tensor is just a coord-valued companion that follows the same
  partitioning as your data tensor — that's how you build OOB masks without
  writing per-thread index math by hand.
