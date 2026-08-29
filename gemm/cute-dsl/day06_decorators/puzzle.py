#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 06 -- CuTe DSL decorator boundaries (puzzle)."""

import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def ceil_div(value, divisor):
    """Plain Python helpers are evaluated while a decorated caller is traced."""
    return (value + divisor - 1) // divisor


@cute.jit
def affine_value(value, scale: cutlass.Constexpr, bias: cutlass.Constexpr):
    # TODO(1): Return scale * value + bias.
    #
    # This @jit function is called by a @kernel, so CuTe inlines it into the
    # generated device function at compile time.
    raise NotImplementedError("Day 06 TODO(1): implement the @jit helper")


@cute.kernel
def affine_kernel(
    g_src: cute.Tensor,
    g_dst: cute.Tensor,
    scale: cutlass.Constexpr,
    bias: cutlass.Constexpr,
):
    # TODO(2):
    #   1. Compute gid = blockIdx.x * blockDim.x + threadIdx.x.
    #   2. Guard gid with cute.size(g_dst).
    #   3. Store affine_value(g_src[gid], scale, bias) into g_dst[gid].
    raise NotImplementedError("Day 06 TODO(2): implement the GPU kernel")


@cute.jit
def affine(
    m_src: cute.Tensor,
    m_dst: cute.Tensor,
    scale: cutlass.Constexpr,
    bias: cutlass.Constexpr,
):
    threads_per_block: cutlass.Constexpr = 128
    n = cute.size(m_dst)

    # TODO(3): Call the plain Python ceil_div helper, then launch
    # affine_kernel with a 1-D grid and 1-D block:
    #
    #   blocks = ceil_div(n, threads_per_block)
    #   affine_kernel(...).launch(
    #       grid=[blocks, 1, 1],
    #       block=[threads_per_block, 1, 1],
    #   )
    #
    # Python may call @jit, and @jit may launch @kernel. Python must not call
    # affine_kernel directly.
    raise NotImplementedError("Day 06 TODO(3): launch the kernel from @jit")


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
