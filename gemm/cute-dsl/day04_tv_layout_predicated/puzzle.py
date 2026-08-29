#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 04 — CuTe DSL TV-layout elementwise add with OOB predication (puzzle).

The host code is filled in for you. Implement the kernel: TV-layout
partitioning + a predicate fragment built from a coordinate tensor.
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
    cC: cute.Tensor,
    shape: cute.Shape,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # TODO: implement the five steps below.
    #
    # 1. Slice each of gA, gB, gC, cC down to this CTA's tile:
    #        blk_coord = ((None, None), bidx)
    #        blkA = gA[blk_coord]            etc.
    #
    # 2. Compose each block with `tv_layout` so the result maps (tid, vid) -> addr:
    #        tidfrgA   = cute.composition(blkA,   tv_layout)
    #        ...
    #        tidfrgCrd = cute.composition(blkCrd, tv_layout)
    #
    # 3. Slice down to this thread's view: thr_coord = (tidx, None)
    #        thrA   = tidfrgA[thr_coord]
    #        ...
    #        thrCrd = tidfrgCrd[thr_coord]
    #
    # 4. Build the predicate fragment from thrCrd:
    #        frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
    #        for i in cutlass.range_constexpr(cute.size(frgPred)):
    #            frgPred[i] = cute.elem_less(thrCrd[i], shape)
    #
    # 5. Load A and B into register fragments under the predicate, compute,
    #    store C under the predicate. You can use:
    #        frgA = cute.make_fragment_like(thrA)
    #        if frgPred[i]: frgA[i] = thrA[i]    # DSL 4.5+ traces this dynamically
    #        result = frgA.load() + frgB.load()
    #        frgC.store(result)
    #        if frgPred[i]: thrC[i] = frgC[i]
    raise NotImplementedError("Day 04: fill in the TV-layout kernel")


@cute.jit
def elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    assert mA.element_type == mB.element_type == mC.element_type
    dtype = mA.element_type

    elts_per_vec: cutlass.Constexpr = 128 // dtype.width
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, elts_per_vec), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

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
    run(args.M, args.N)
    run(1023, 1025)
    print("\nSuccess.")
