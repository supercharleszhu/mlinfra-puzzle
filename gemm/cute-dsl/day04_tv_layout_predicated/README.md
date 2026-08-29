# Day 04 — CuTe DSL TV-Layout Elementwise + Predication

**Goal:** rewrite Day 03's elementwise add using an explicit Thread-Value (TV)
layout, plus a coordinate tensor + predicate so the kernel works correctly
when `M` or `N` is *not* a multiple of the CTA tile size.

This is **the** canonical CuTe DSL kernel skeleton. Every later kernel
(layernorm, GEMM mainloop, GEMM epilogue, FlashAttention) is a variation on
the same five steps.

## Background

### The five-step CuTe DSL pattern

```
(host)
  thr_layout, val_layout                          ← who reads what
  tiler_mn, tv_layout = make_layout_tv(thr, val)  ← (tid, vid) -> tile coord
  gA  = zipped_divide(mA, tiler_mn)               ← per-CTA tiling
  idC = make_identity_tensor(shape)               ← coord-valued companion
  cC  = zipped_divide(idC, tiler_mn)              ← same partitioning as gC
  launch kernel(gA, gB, gC, cC, shape, tv_layout)

(device)
  1. blk_coord = ((None, None), bidx)
     blkA = gA[blk_coord]                  CTA-local view  (TileM,TileN)->addr
     blkCrd = cC[blk_coord]                CTA-local view  (TileM,TileN)->coord
  2. tidfrgA = composition(blkA, tv_layout)        (tid, vid) -> addr
     tidfrgCrd = composition(blkCrd, tv_layout)    (tid, vid) -> coord
  3. thrA = tidfrgA[(tidx, None)]                  (vid) -> addr
     thrCrd = tidfrgCrd[(tidx, None)]              (vid) -> coord
  4. frgPred[i] = elem_less(thrCrd[i], shape)
  5. predicated load → compute → predicated store
```

Once you've internalized this, every CUTLASS DSL example becomes legible:
they all just substitute different `tv_layout`s, copy atoms, and inner
operations into the same five-step shell.

### Why predication?

`zipped_divide(mA, (16, 256))` doesn't error when `(M, N)` isn't a multiple
of `(16, 256)`. It just rounds up the outer mode, producing some tiles whose
coordinates run past the end of the real tensor. The *coordinate tensor* `cC`
follows the same partitioning, so each thread can check its own `(m, n)` for
each value it would write — and skip the OOB ones.

## Task

Implement the five steps in `puzzle.py`. The TV layout, tiler, and host launch
are already wired up.

## Run

```bash
python puzzle.py --M 1024 --N 1024     # divisible case
python puzzle.py --M 1023 --N 1025     # exercises the predicate
python solution.py                     # reference
```

## What you should walk away knowing

- The TV-layout / composition / partition pattern is one shape — once you know
  it, *all* CuTe DSL kernels become "fill in the body of step 5".
- An identity tensor is the cleanest way to get per-thread OOB coordinates
  without writing index arithmetic by hand. It also costs zero memory — it's
  a layout-only construct.
- `cutlass.range_constexpr` is unrolled at compile time; the per-thread
  predicate loop has no runtime overhead.
- `make_fragment` / `make_fragment_like` allocate a register tensor matching
  the shape of a thread slice. That's the "register file" of the DSL.
