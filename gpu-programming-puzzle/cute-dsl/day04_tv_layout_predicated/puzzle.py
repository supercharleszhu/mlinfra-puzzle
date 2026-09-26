#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 04: TV-layout elementwise add with OOB predication.

Official Notebook 08 develops elementwise add in four stages: scalar indexing,
vectorized slices, TV-layout ownership, and a reusable elementwise operator.
Day 03 covered the first two. This puzzle focuses on the TV-layout stage and
adds coordinate-based predication for partial edge tiles.
"""
import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

try:
    import cutlass.testing as testing
except ModuleNotFoundError as error:
    if error.name != "cutlass.testing":
        raise
    from cutlass.cute import testing


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

    # TODO(1): Select this CTA's data and coordinate tiles.
    #
    # 1. Slice each of gA, gB, gC, cC down to this CTA's tile:
    #        blk_coord = ((None, None), bidx)
    #        blkA = gA[blk_coord]            etc.
    # TODO(2): Compose each block with `tv_layout`.
    #
    # The mapping chain is:
    #   (tid, vid) --tv_layout--> (tile_m, tile_n)
    #              --blkA-------> global-memory address
    #
    # Compose each block so the result maps (tid, vid) -> address:
    #        tidfrgA   = cute.composition(blkA,   tv_layout)
    #        ...
    #        tidfrgCrd = cute.composition(blkCrd, tv_layout)
    #
    # TODO(3): Slice the composed tensors to this thread.
    #
    # Use thr_coord = (tidx, None):
    #        thrA   = tidfrgA[thr_coord]
    #        ...
    #        thrCrd = tidfrgCrd[thr_coord]
    #
    # TODO(4): Reprofile each thread tensor as `(1, num_values)`.
    #
    # `cute.copy` treats mode 0 as its V-mode and predicates the remaining
    # modes. A scalar V-mode therefore preserves one predicate per value:
    #        copy_layout = cute.make_layout(
    #            (1, cute.size(thrCrd)), stride=(0, 1)
    #        )
    #        copyA = cute.composition(thrA, copy_layout)
    #        ...
    #        copyCrd = cute.composition(thrCrd, copy_layout)
    #
    # TODO(5): Build a per-value predicate from the reprofiled coordinates:
    #        frgPred = cute.make_rmem_tensor(copyCrd.shape, cutlass.Boolean)
    #        for i in cutlass.range_constexpr(cute.size(frgPred)):
    #            frgPred[i] = cute.elem_less(copyCrd[i], shape)
    #
    # TODO(6): Create a universal copy atom, use predicated `cute.copy` to load
    # A and B into register fragments, apply the elementwise operation, then
    # use predicated `cute.copy` to store C.
    #
    # The arithmetic is intentionally the easy part. Let CuTe own the
    # per-element copy traversal:
    #        frgA = cute.make_fragment_like(copyA)
    #        copy_atom = cute.make_copy_atom(
    #            cute.nvgpu.CopyUniversalOp(),
    #            gA.element_type,
    #            num_bits_per_copy=gA.element_type.width,
    #        )
    #        cute.copy(copy_atom, copyA, frgA, pred=frgPred)
    #        cute.copy(copy_atom, copyB, frgB, pred=frgPred)
    #        result = frgA.load() + frgB.load()
    #        frgC.store(result)
    #        cute.copy(copy_atom, frgC, copyC, pred=frgPred)
    raise NotImplementedError("Day 04: fill in the TV-layout kernel")


@cute.jit
def elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    assert mA.element_type == mB.element_type == mC.element_type
    dtype = mA.element_type

    elts_per_vec: cutlass.Constexpr = 128 // dtype.width
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, elts_per_vec), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"[JIT] tiler_mn={tiler_mn} tv_layout={tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)

    elementwise_add_kernel(gA, gB, gC, cC, mC.shape, tv_layout).launch(
        grid=(cute.size(gC, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
    )


def benchmark(
    compiled,
    a_: cute.Tensor,
    b_: cute.Tensor,
    c_: cute.Tensor,
    total_bytes: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Return average kernel time in microseconds and effective GB/s."""
    avg_time_us = testing.benchmark(
        compiled,
        kernel_arguments=testing.JitArguments(a_, b_, c_),
        warmup_iterations=warmup,
        iterations=iterations,
    )
    bandwidth_gbps = total_bytes / (avg_time_us * 1_000)
    return avg_time_us, bandwidth_gbps


def run(M: int, N: int, run_benchmark: bool, warmup: int, iterations: int) -> None:
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
    compiled = cute.compile(elementwise_add, a_, b_, c_)
    compiled(a_, b_, c_)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)
    print("  OK")

    if run_benchmark:
        total_bytes = 3 * a.numel() * a.element_size()
        avg_time_us, bandwidth_gbps = benchmark(
            compiled, a_, b_, c_, total_bytes, warmup, iterations
        )
        print(f"  time={avg_time_us:.3f} us")
        print(f"  effective bandwidth={bandwidth_gbps:.2f} GB/s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()
    run(args.M, args.N, args.benchmark, args.warmup, args.iterations)
    run(1023, 1025, False, args.warmup, args.iterations)
    print("\nSuccess.")
