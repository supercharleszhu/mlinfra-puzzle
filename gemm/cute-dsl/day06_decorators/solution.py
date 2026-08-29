#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 06 -- CuTe DSL decorator boundaries (solution)."""

import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def ceil_div(value, divisor):
    """Plain Python helpers are evaluated while a decorated caller is traced."""
    return (value + divisor - 1) // divisor


@cute.jit
def affine_value(value, scale: cutlass.Constexpr, bias: cutlass.Constexpr):
    return scale * value + bias


@cute.kernel
def affine_kernel(
    g_src: cute.Tensor,
    g_dst: cute.Tensor,
    scale: cutlass.Constexpr,
    bias: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gid = bidx * bdim + tidx

    if gid < cute.size(g_dst):
        g_dst[gid] = affine_value(g_src[gid], scale, bias)


@cute.jit
def affine(
    m_src: cute.Tensor,
    m_dst: cute.Tensor,
    scale: cutlass.Constexpr,
    bias: cutlass.Constexpr,
):
    threads_per_block: cutlass.Constexpr = 128
    n = cute.size(m_dst)
    blocks = ceil_div(n, threads_per_block)
    affine_kernel(m_src, m_dst, scale, bias).launch(
        grid=[blocks, 1, 1],
        block=[threads_per_block, 1, 1],
    )


def run(n: int, scale: float, bias: float) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 06 needs an SM80+ CUDA device.")

    cutlass.cuda.initialize_cuda_context()
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    m_src = from_dlpack(src, assumed_align=16).mark_layout_dynamic()
    m_dst = from_dlpack(dst, assumed_align=16).mark_layout_dynamic()
    affine(m_src, m_dst, scale, bias)
    torch.cuda.synchronize()

    torch.testing.assert_close(dst, scale * src + bias)
    print(f"OK -- affine transform verified for {n} elements")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--bias", type=float, default=-0.75)
    args = parser.parse_args()
    run(args.n, args.scale, args.bias)
    print("Success.")
