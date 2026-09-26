#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 08 - CuTe DSL benchmark and autotune APIs (solution)."""
import argparse

import cutlass
import cutlass.cute as cute
import cutlass.testing as testing
from cutlass.cute.testing import autotune_jit


@cute.kernel
def add_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_c: cute.Tensor,
    coordinates: cute.Tensor,
    shape: cute.Shape,
    thread_layout: cute.Layout,
    value_layout: cute.Layout,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    block_coord = ((None, None), block_idx)

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), g_a.element_type
    )
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom, thread_layout, value_layout
    )
    thread_copy = tiled_copy.get_slice(thread_idx)

    thread_a = thread_copy.partition_S(g_a[block_coord])
    thread_b = thread_copy.partition_S(g_b[block_coord])
    thread_c = thread_copy.partition_D(g_c[block_coord])
    thread_coord = thread_copy.partition_S(coordinates[block_coord])

    fragment_a = cute.make_fragment_like(thread_a)
    fragment_b = cute.make_fragment_like(thread_b)
    fragment_c = cute.make_fragment_like(thread_c)
    predicates = cute.make_rmem_tensor(thread_coord.shape, cutlass.Boolean)

    for i in cutlass.range_constexpr(cute.size(predicates)):
        predicates[i] = cute.elem_less(thread_coord[i], shape)

    cute.copy(copy_atom, thread_a, fragment_a, pred=predicates)
    cute.copy(copy_atom, thread_b, fragment_b, pred=predicates)
    fragment_c.store(fragment_a.load() + fragment_b.load())
    cute.copy(copy_atom, fragment_c, thread_c, pred=predicates)


@autotune_jit(
    params_dict={"copy_bits": [64, 128]},
    update_on_change=["m", "n"],
    warmup_iterations=5,
    iterations=20,
)
@cute.jit
def add(
    tensor_a,
    tensor_b,
    tensor_c,
    m,
    n,
    copy_bits: cutlass.Constexpr = 128,
):
    vector_size = copy_bits // tensor_a.element_type.width
    thread_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    value_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tile_shape, tv_layout = cute.make_layout_tv(thread_layout, value_layout)

    g_a = cute.zipped_divide(tensor_a, tile_shape)
    g_b = cute.zipped_divide(tensor_b, tile_shape)
    g_c = cute.zipped_divide(tensor_c, tile_shape)
    coordinates = cute.zipped_divide(
        cute.make_identity_tensor(tensor_c.shape), tile_shape
    )

    add_kernel(
        g_a,
        g_b,
        g_c,
        coordinates,
        tensor_c.shape,
        thread_layout,
        value_layout,
    ).launch(
        grid=(cute.size(g_c, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
    )


def tune_add(tensor_a, tensor_b, tensor_c, m: int, n: int):
    def compile_config(a, b, c, rows, cols, copy_bits=128):
        compiled = cute.compile(add, a, b, c, rows, cols, copy_bits=copy_bits)
        return lambda: compiled(a, b, c, rows, cols)

    return testing.tune(
        compile_config,
        params_dict={"copy_bits": [64, 128]},
        kernel_arguments=testing.JitArguments(
            tensor_a, tensor_b, tensor_c, m, n
        ),
        warmup_iterations=5,
        iterations=20,
    )


def benchmark_add(m: int, n: int, dtype):
    import torch

    def make_workspace():
        a = torch.randn(m, n, device="cuda", dtype=dtype)
        b = torch.randn(m, n, device="cuda", dtype=dtype)
        c = torch.empty_like(a)
        return testing.JitArguments(a, b, c, m, n)

    return testing.benchmark(
        add,
        workspace_generator=make_workspace,
        workspace_count=3,
        warmup_iterations=5,
        iterations=20,
    )


def run(m: int, n: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 08 needs a CUDA device.")
    cutlass.cuda.initialize_cuda_context()

    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    b = torch.randn(m, n, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    best = tune_add(a, b, c, m, n)
    compiled = cute.compile(add, a, b, c, m, n, **best)
    compiled(a, b, c, m, n)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)

    average_us = benchmark_add(m, n, torch.float32)
    bytes_moved = 3 * a.numel() * a.element_size()
    bandwidth_gbps = bytes_moved / (average_us / 1e6) / 1e9
    print(f"best configuration={best}")
    print(f"average={average_us:.2f} us, bandwidth={bandwidth_gbps:.2f} GB/s")
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--N", type=int, default=512)
    args = parser.parse_args()
    run(args.M, args.N)
