#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 04 — CuTe DSL TV-layout elementwise add with OOB predication (solution).

Builds on Day 03 by:

  - Using an explicit Thread-Value (TV) layout instead of zipped_divide alone.
  - Adding a coordinate tensor + predicate fragment so we work correctly when
    the input shape is *not* a multiple of the CTA tile.

This is the canonical CuTe pattern you'll see in every later DSL kernel
(elementwise, layernorm, GEMM epilogue, etc.).
"""
import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,           # coordinate tensor, same partitioning as gC
    shape: cute.Shape,         # original (M, N) — for OOB compare
    tv_layout: cute.Layout,    # (tid, vid) -> tile coord
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 1. Pick out this CTA's tile.
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]       # (TileM, TileN) -> addr
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]     # (TileM, TileN) -> (m, n) global coord

    # 2. Compose with TV layout to get (tid, vid) -> addr / coord directly.
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)
    tidfrgCrd = cute.composition(blkCrd, tv_layout)

    # 3. Slice down to this thread's view: (vid) -> addr.
    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]
    thrCrd = tidfrgCrd[thr_coord]

    # 4. Build the predicate fragment from the coord tensor.
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    # 5. Load A, B (predicated), compute, store C (predicated).
    #    Since we don't need predication on the load with no defined-fill, we
    #    can also just load and conditionally store — but the canonical pattern
    #    is to apply pred to both loads and stores via a register fragment.
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    # Load with mask: OOB lanes will read garbage; we just won't write them.
    for i in cutlass.range_constexpr(cute.size(frgA)):
        if frgPred[i]:  # DSL 4.5+: plain `if` is traced as a dynamic branch
            frgA[i] = thrA[i]
            frgB[i] = thrB[i]

    res = frgA.load() + frgB.load()
    frgC.store(res)

    for i in cutlass.range_constexpr(cute.size(frgC)):
        if frgPred[i]:  # DSL 4.5+: plain `if` is traced as a dynamic branch
            thrC[i] = frgC[i]


@cute.jit
def elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    assert mA.element_type == mB.element_type == mC.element_type
    dtype = mA.element_type

    # 128-bit vector load: how many elements is that?
    elts_per_vec: cutlass.Constexpr = 128 // dtype.width

    # 128 threads laid out (4 rows of 32), each reading a (4, elts_per_vec) sub-tile.
    # CTA tile size = thr * val per mode = (4*4, 32*elts_per_vec) = (16, ~256) for fp16.
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, elts_per_vec), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"[JIT] tiler_mn={tiler_mn}  tv_layout={tv_layout}")

    # zipped_divide handles non-divisible sizes: outer mode is rounded up via
    # zero-padding *of the layout* (still valid offsets, but the coord tensor
    # will go OOB at the edges — that's exactly what our predicate catches).
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    # Coordinate tensor — same shape as mC, value at (m, n) is the tuple (m, n).
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)

    elementwise_add_kernel(gA, gB, gC, cC, mC.shape, tv_layout).launch(
        grid=(cute.size(gC, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
    )


def run(M: int, N: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 04 needs a CUDA device.")
    cutlass.cuda.initialize_cuda_context()

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros_like(a)

    a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
    b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
    c_ = from_dlpack(c, assumed_align=16).mark_layout_dynamic()

    print(f"\n=== TV-layout elementwise add (M={M}, N={N}) ===")
    elementwise_add(a_, b_, c_)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)
    print("  OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    args = p.parse_args()
    # Deliberately try non-tile-multiple sizes — the predicate handles them.
    run(args.M, args.N)
    run(1023, 1025)   # odd primes-ish, hits the predicate path
    print("\nSuccess.")
