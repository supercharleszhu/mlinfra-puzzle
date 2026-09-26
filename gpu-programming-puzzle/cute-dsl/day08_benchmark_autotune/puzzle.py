#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 08 - fill the CuTe DSL benchmark/autotune TODOs."""
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
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), g_a.element_type)
    tiled_copy = cute.make_tiled_copy_tv(atom, thread_layout, value_layout)
    thread_copy = tiled_copy.get_slice(thread_idx)
    a = thread_copy.partition_S(g_a[block_coord])
    b = thread_copy.partition_S(g_b[block_coord])
    c = thread_copy.partition_D(g_c[block_coord])
    coord = thread_copy.partition_S(coordinates[block_coord])
    fragment_a = cute.make_fragment_like(a)
    fragment_b = cute.make_fragment_like(b)
    fragment_c = cute.make_fragment_like(c)
    pred = cute.make_rmem_tensor(coord.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(pred)):
        pred[i] = cute.elem_less(coord[i], shape)
    cute.copy(atom, a, fragment_a, pred=pred)
    cute.copy(atom, b, fragment_b, pred=pred)
    fragment_c.store(fragment_a.load() + fragment_b.load())
    cute.copy(atom, fragment_c, c, pred=pred)


# TODO(1): add @autotune_jit with copy_bits=[64, 128],
# update_on_change=["m", "n"], and short warmup/measurement counts.
@cute.jit
def add(tensor_a, tensor_b, tensor_c, m, n, copy_bits: cutlass.Constexpr = 128):
    # TODO(2): convert copy_bits to an element count using dtype.width.
    raise NotImplementedError("Day 08 TODO(2): compute vector_size")

    thread_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    value_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tile_shape, tv_layout = cute.make_layout_tv(thread_layout, value_layout)
    g_a = cute.zipped_divide(tensor_a, tile_shape)
    g_b = cute.zipped_divide(tensor_b, tile_shape)
    g_c = cute.zipped_divide(tensor_c, tile_shape)
    coordinates = cute.zipped_divide(
        cute.make_identity_tensor(tensor_c.shape), tile_shape
    )

    # TODO(3): launch add_kernel with one CTA per outer g_c tile and one
    # thread per TV-layout thread coordinate.


def tune_add(tensor_a, tensor_b, tensor_c, m: int, n: int):
    # TODO(4): define a callback that compiles one copy_bits configuration and
    # returns a zero-argument lambda that invokes the compiled function.
    # TODO(5): return testing.tune(...), passing params_dict and JitArguments.
    raise NotImplementedError("Day 08 TODO(4-5): implement explicit tuning")


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

    # TODO(6): benchmark `add` with testing.benchmark and print microseconds.
    print(f"best configuration={best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--N", type=int, default=512)
    args = parser.parse_args()
    run(args.M, args.N)
