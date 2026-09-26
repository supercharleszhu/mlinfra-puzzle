#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 05: naive, padded, and swizzled matrix transpose solutions."""

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
def naive_coalesced_read_kernel(
    gA,
    gB,
    gAcrd,
    shape,
    thread_layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    tgA = cute.local_partition(
        gA[(None, None), (bidy, bidx)], thread_layout, tidx
    )
    tgB = cute.local_partition(
        gB[(None, None), (bidy, bidx)], thread_layout, tidx
    )
    tgAcrd = cute.local_partition(
        gAcrd[(None, None), (bidy, bidx)], thread_layout, tidx
    )
    pred = cute.make_rmem_tensor(tgAcrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tgA)):
        pred[i] = cute.elem_less(tgAcrd[i], shape)

    fragment = cute.make_fragment_like(tgA)
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gA.element_type,
        num_bits_per_copy=gA.element_type.width,
    )
    cute.copy(copy_atom, tgA, fragment, pred=pred)
    cute.copy(copy_atom, fragment, tgB, pred=pred)


@cute.kernel
def naive_coalesced_write_kernel(
    gA,
    gB,
    gAcrd,
    shape,
    thread_layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    tgA = cute.local_partition(
        gA[(None, None), (bidy, bidx)], thread_layout, tidx
    )
    tgB = cute.local_partition(
        gB[(None, None), (bidy, bidx)], thread_layout, tidx
    )
    tgAcrd = cute.local_partition(
        gAcrd[(None, None), (bidy, bidx)], thread_layout, tidx
    )
    pred = cute.make_rmem_tensor(tgAcrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tgA)):
        pred[i] = cute.elem_less(tgAcrd[i], shape)

    fragment = cute.make_fragment_like(tgA)
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gA.element_type,
        num_bits_per_copy=gA.element_type.width,
    )
    cute.copy(copy_atom, tgA, fragment, pred=pred)
    cute.copy(copy_atom, fragment, tgB, pred=pred)


@cute.jit
def naive_transpose_coalesced_read(input: cute.Tensor, output: cute.Tensor):
    tileM = 32
    tileN = 32
    TM = 8
    TN = 32
    thread_layout = cute.make_ordered_layout((TM, TN), order=(1, 0))
    output_T = cute.make_tensor(
        output.iterator,
        layout=cute.make_layout(
            shape=(output.shape[1], output.shape[0]),
            stride=(1, output.shape[1]),
        ),
    )
    gA = cute.zipped_divide(input, (tileM, tileN))
    gB = cute.zipped_divide(output_T, (tileM, tileN))
    gAcrd = cute.zipped_divide(
        cute.make_identity_tensor(input.shape), (tileM, tileN)
    )
    grid = (gA.shape[1][1], gA.shape[1][0], 1)
    thread_per_block = TM * TN
    naive_coalesced_read_kernel(
        gA, gB, gAcrd, input.shape, thread_layout
    ).launch(grid=grid, block=(thread_per_block, 1, 1))


@cute.jit
def naive_transpose_coalesced_write(input: cute.Tensor, output: cute.Tensor):
    tileM = 32
    tileN = 32
    TM = 8
    TN = 32
    thread_layout = cute.make_ordered_layout((TM, TN), order=(1, 0))
    input_T = cute.make_tensor(
        input.iterator,
        layout=cute.make_layout(
            shape=(input.shape[1], input.shape[0]),
            stride=(1, input.shape[1]),
        ),
    )
    gA = cute.zipped_divide(input_T, (tileN, tileM))
    gB = cute.zipped_divide(output, (tileN, tileM))
    gAcrd = cute.zipped_divide(
        cute.make_identity_tensor(input_T.shape), (tileN, tileM)
    )
    grid = (gA.shape[1][1], gA.shape[1][0], 1)
    thread_per_block = TM * TN
    naive_coalesced_write_kernel(
        gA, gB, gAcrd, input_T.shape, thread_layout
    ).launch(grid=grid, block=(thread_per_block, 1, 1))


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
    # sAT[out_row, out_col] aliases sA[out_col, out_row].
    sAT = cute.make_tensor(sA.iterator, smem_layout_t)

    # Load phase: A and sA share the logical input-tile domain.
    tAgA = cute.local_partition(tile_a, thread_layout, tidx)
    tAsA = cute.local_partition(sA, thread_layout, tidx)
    tAgACrd = cute.local_partition(tile_a_crd, thread_layout, tidx)
    pred_a = cute.make_rmem_tensor(tAgACrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(pred_a)):
        pred_a[i] = cute.elem_less(tAgACrd[i], shape_a)

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gA.element_type,
        num_bits_per_copy=gA.element_type.width,
    )
    rA = cute.make_fragment_like(tAgA)
    cute.copy(copy_atom, tAgA, rA, pred=pred_a)
    cute.copy(copy_atom, rA, tAsA, pred=pred_a)

    cute.arch.sync_threads()

    # Store phase: sAT and B share the logical output-tile domain.
    tBsAT = cute.local_partition(sAT, thread_layout, tidx)
    tBgB = cute.local_partition(tile_b, thread_layout, tidx)
    tBgBCrd = cute.local_partition(tile_b_crd, thread_layout, tidx)
    pred_b = cute.make_rmem_tensor(tBgBCrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(pred_b)):
        pred_b[i] = cute.elem_less(tBgBCrd[i], shape_b)

    rB = cute.make_fragment_like(tBgB)
    cute.copy(copy_atom, tBsAT, rB, pred=pred_b)
    cute.copy(copy_atom, rB, tBgB, pred=pred_b)


def make_smem_layouts(kind: str):
    tile = (32, 32)
    if kind == "unpadded":
        return (
            cute.make_layout(tile, stride=(32, 1)),
            cute.make_layout(tile, stride=(1, 32)),
        )
    if kind == "padded":
        return (
            cute.make_layout(tile, stride=(33, 1)),
            cute.make_layout(tile, stride=(1, 33)),
        )
    if kind == "swizzled":
        # CuTe layouts use element offsets. XOR the five row bits beginning
        # at bit 5 into the five column bits used by a 32-element row.
        swizzle = cute.make_swizzle(5, 0, 5)
        return (
            cute.make_composed_layout(
                swizzle, 0, cute.make_layout(tile, stride=(32, 1))
            ),
            cute.make_composed_layout(
                swizzle, 0, cute.make_layout(tile, stride=(1, 32))
            ),
        )
    raise ValueError(f"unsupported shared layout kind: {kind}")


def banks_for_column(row_stride: int, column: int = 0) -> list[int]:
    return [(lane * row_stride + column) % 32 for lane in range(32)]


def padded_layout_model(
    tile_rows: int = 32,
    tile_cols: int = 32,
) -> tuple[int, int]:
    row_stride = tile_cols + 1
    allocated_words = (
        (tile_rows - 1) * row_stride + (tile_cols - 1) + 1
    )
    return row_stride, allocated_words


def swizzle_5_0_5(offset: int) -> int:
    return offset ^ ((offset >> 5) & 0x1F)


def launch_shared_transpose(
    mA: cute.Tensor,
    mB: cute.Tensor,
    kind: str,
) -> None:
    tile = (32, 32)
    thread_layout = cute.make_ordered_layout((8, 32), order=(1, 0))
    gA = cute.zipped_divide(mA, tile)
    gB = cute.zipped_divide(mB, tile)
    gACrd = cute.zipped_divide(cute.make_identity_tensor(mA.shape), tile)
    gBCrd = cute.zipped_divide(cute.make_identity_tensor(mB.shape), tile)
    smem_layout, smem_layout_t = make_smem_layouts(kind)

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
    ).launch(
        grid=(
            cute.ceil_div(mA.shape[1], 32),
            cute.ceil_div(mA.shape[0], 32),
            1,
        ),
        block=(256, 1, 1),
    )


@cute.jit
def transpose_unpadded(mA: cute.Tensor, mB: cute.Tensor):
    launch_shared_transpose(mA, mB, "unpadded")


@cute.jit
def transpose_padded(mA: cute.Tensor, mB: cute.Tensor):
    launch_shared_transpose(mA, mB, "padded")


@cute.jit
def transpose_swizzled(mA: cute.Tensor, mB: cute.Tensor):
    launch_shared_transpose(mA, mB, "swizzled")


# Backward-compatible name used by the notebook's original padded challenge.
transpose = transpose_padded


def benchmark(
    compiled,
    a_: cute.Tensor,
    b_: cute.Tensor,
    total_bytes: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Return average kernel time in microseconds and effective GB/s."""
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
    if m < 1 or n < 1:
        raise ValueError("M and N must be positive.")

    cutlass.cuda.initialize_cuda_context()
    variants = {
        "read": ("naive/coalesced-read", naive_transpose_coalesced_read),
        "write": ("naive/coalesced-write", naive_transpose_coalesced_write),
        "unpadded": ("shared/unpadded", transpose_unpadded),
        "padded": ("shared/padded", transpose_padded),
        "swizzled": ("shared/swizzled", transpose_swizzled),
    }
    selected = variants.items() if mode == "all" else [(mode, variants[mode])]

    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
    print(f"\n=== matrix transpose ({m}, {n}) -> ({n}, {m}) ===")
    for _, (name, transpose_fn) in selected:
        b = torch.empty(n, m, device="cuda", dtype=torch.float32)
        b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
        compiled = cute.compile(transpose_fn, a_, b_)
        compiled(a_, b_)
        torch.cuda.synchronize()
        torch.testing.assert_close(b, a.T)
        print(f"{name}: correctness OK")

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
    if (args.M, args.N) != (1003, 1507):
        run(1003, 1507, args.mode, False, args.warmup, args.iterations)
    print("\nSuccess.")
