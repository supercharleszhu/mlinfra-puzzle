#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 05: build naive and partitioned shared-memory transpose variants."""

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
def naive_coalesced_read_kernel(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    tile_size: cutlass.Constexpr = 32
    threads_m: cutlass.Constexpr = 8
    values_per_thread: cutlass.Constexpr = tile_size // threads_m

    # TODO(A1): Interpret 256 linear threads as an (8, 32) row-major layout.
    #
    # lane_col = tidx % tile_size
    # lane_row = tidx // tile_size
    raise NotImplementedError("Day 05A TODO(A1): derive thread coordinates")

    # TODO(A2): Give each thread four values with coalesced input reads.
    #
    # For each compile-time `value`:
    #   local_row = lane_row + value * threads_m
    #   row = bidy * tile_size + local_row
    #   col = bidx * tile_size + lane_col
    #
    # Predicate against mA.shape and assign:
    #   mB[col, row] = mA[row, col]


@cute.kernel
def naive_coalesced_write_kernel(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    tile_size: cutlass.Constexpr = 32
    threads_m: cutlass.Constexpr = 8
    values_per_thread: cutlass.Constexpr = tile_size // threads_m

    lane_col = tidx % tile_size
    lane_row = tidx // tile_size

    # TODO(A3): Reverse ownership so adjacent lanes write one output row.
    #
    # For each compile-time `value`:
    #   local_output_row = lane_row + value * threads_m
    #   output_row = bidx * tile_size + local_output_row
    #   output_col = bidy * tile_size + lane_col
    #
    # Predicate against mB.shape and assign:
    #   mB[output_row, output_col] = mA[output_col, output_row]
    raise NotImplementedError("Day 05A TODO(A3): implement coalesced writes")


@cute.jit
def naive_transpose_coalesced_read(mA: cute.Tensor, mB: cute.Tensor):
    # TODO(A4): Launch the coalesced-read kernel over ceil(N/32) by
    # ceil(M/32) input tiles with 256 threads.
    raise NotImplementedError("Day 05A TODO(A4): launch coalesced-read kernel")


@cute.jit
def naive_transpose_coalesced_write(mA: cute.Tensor, mB: cute.Tensor):
    # TODO(A5): Launch the coalesced-write kernel over the same input-tile
    # grid with 256 threads.
    raise NotImplementedError("Day 05A TODO(A5): launch coalesced-write kernel")


@cute.kernel
def shared_transpose_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gACrd: cute.Tensor,
    gBCrd: cute.Tensor,
    shape_a: cute.Shape,
    shape_b: cute.Shape,
    thread_layout: cute.Layout,
    smem_layout,
    smem_layout_t,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    block_a = ((None, None), (bidy, bidx))
    block_b = ((None, None), (bidx, bidy))
    tile_a = gA[block_a]
    tile_b = gB[block_b]
    tile_a_crd = gACrd[block_a]
    tile_b_crd = gBCrd[block_b]

    allocator = cutlass.utils.SmemAllocator()
    sA = allocator.allocate_tensor(gA.element_type, smem_layout, 16)
    sAT = cute.make_tensor(sA.iterator, smem_layout_t)

    tAgA = cute.local_partition(tile_a, thread_layout, tidx)
    tAsA = cute.local_partition(sA, thread_layout, tidx)
    tAgACrd = cute.local_partition(tile_a_crd, thread_layout, tidx)
    pred_a = cute.make_rmem_tensor(tAgACrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tAgA)):
        pred_a[i] = cute.elem_less(tAgACrd[i], shape_a)

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gA.element_type,
        num_bits_per_copy=gA.element_type.width,
    )
    fragment_a = cute.make_fragment_like(tAgA)
    cute.copy(copy_atom, tAgA, fragment_a, pred=pred_a)
    cute.copy(copy_atom, fragment_a, tAsA, pred=pred_a)

    cute.arch.sync_threads()

    tBsAT = cute.local_partition(sAT, thread_layout, tidx)
    tBgB = cute.local_partition(tile_b, thread_layout, tidx)
    tBgBCrd = cute.local_partition(tile_b_crd, thread_layout, tidx)
    pred_b = cute.make_rmem_tensor(tBgBCrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tBgB)):
        pred_b[i] = cute.elem_less(tBgBCrd[i], shape_b)

    fragment_b = cute.make_fragment_like(tBgB)
    cute.copy(copy_atom, tBsAT, fragment_b, pred=pred_b)
    cute.copy(copy_atom, fragment_b, tBgB, pred=pred_b)


def launch_shared_transpose(
    mA: cute.Tensor,
    mB: cute.Tensor,
    smem_layout,
    smem_layout_t,
) -> None:
    tile = (32, 32)
    thread_layout = cute.make_ordered_layout((8, 32), order=(1, 0))
    gA = cute.zipped_divide(mA, tile)
    gB = cute.zipped_divide(mB, tile)
    gACrd = cute.zipped_divide(cute.make_identity_tensor(mA.shape), tile)
    gBCrd = cute.zipped_divide(cute.make_identity_tensor(mB.shape), tile)
    grid = (gA.shape[1][1], gA.shape[1][0], 1)

    shared_transpose_kernel(
        gA,
        gB,
        gACrd,
        gBCrd,
        mA.shape,
        mB.shape,
        thread_layout,
        smem_layout,
        smem_layout_t,
    ).launch(grid=grid, block=(256, 1, 1))


@cute.jit
def transpose_unpadded(mA: cute.Tensor, mB: cute.Tensor):
    smem_layout = cute.make_layout((32, 32), stride=(32, 1))
    smem_layout_t = cute.make_layout((32, 32), stride=(1, 32))
    launch_shared_transpose(mA, mB, smem_layout, smem_layout_t)


@cute.jit
def transpose_padded(mA: cute.Tensor, mB: cute.Tensor):
    smem_layout = cute.make_layout((32, 32), stride=(33, 1))
    smem_layout_t = cute.make_layout((32, 32), stride=(1, 33))
    launch_shared_transpose(mA, mB, smem_layout, smem_layout_t)


@cute.jit
def transpose_swizzled(mA: cute.Tensor, mB: cute.Tensor):
    swizzle = cute.make_swizzle(5, 0, 5)
    smem_layout = cute.make_layout((32, 32), stride=(32, 1))
    smem_layout_t = cute.make_layout((32, 32), stride=(1, 32))
    smem_layout = cute.make_composed_layout(
        swizzle,
        0,
        smem_layout,
    )
    smem_layout_t = cute.make_composed_layout(
        swizzle,
        0,
        smem_layout_t,
    )
    launch_shared_transpose(mA, mB, smem_layout, smem_layout_t)


def benchmark(
    compiled,
    a_: cute.Tensor,
    b_: cute.Tensor,
    total_bytes: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    average_us = testing.benchmark(
        compiled,
        kernel_arguments=testing.JitArguments(a_, b_),
        warmup_iterations=warmup,
        iterations=iterations,
    )
    return average_us, total_bytes / (average_us * 1_000)


def run(
    m: int,
    n: int,
    mode: str,
    run_benchmark: bool,
    warmup: int,
    iterations: int,
) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 05 needs a CUDA device.")
    cutlass.cuda.initialize_cuda_context()

    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    b = torch.empty(n, m, device="cuda", dtype=torch.float32)
    a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
    b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()

    variants = {
        "read": ("naive/coalesced-read", naive_transpose_coalesced_read),
        "write": ("naive/coalesced-write", naive_transpose_coalesced_write),
        "unpadded": ("shared/unpadded", transpose_unpadded),
        "padded": ("shared/padded", transpose_padded),
        "swizzled": ("shared/swizzled", transpose_swizzled),
    }
    selected = variants.items() if mode == "all" else [(mode, variants[mode])]
    for _, (name, transpose_fn) in selected:
        compiled = cute.compile(transpose_fn, a_, b_)
        compiled(a_, b_)
        torch.cuda.synchronize()
        torch.testing.assert_close(b, a.T)
        print(f"{name}: correctness OK for ({m}, {n}) -> ({n}, {m})")

        if run_benchmark:
            total_bytes = 2 * a.numel() * a.element_size()
            average_us, bandwidth_gbps = benchmark(
                compiled, a_, b_, total_bytes, warmup, iterations
            )
            print(f"{name}: average={average_us:.3f} us")
            print(f"{name}: effective bandwidth={bandwidth_gbps:.2f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1536)
    parser.add_argument(
        "--mode",
        choices=("read", "write", "unpadded", "padded", "swizzled", "all"),
        default="all",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    run(
        args.M,
        args.N,
        args.mode,
        args.benchmark,
        args.warmup,
        args.iterations,
    )
