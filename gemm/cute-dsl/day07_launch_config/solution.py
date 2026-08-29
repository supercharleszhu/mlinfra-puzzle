#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 07 -- CuTe DSL kernel launch configuration (solution)."""

import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def scale_kernel(
    g_src: cute.Tensor,
    g_dst: cute.Tensor,
    factor: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gid = bidx * bdim + tidx

    if gid < cute.size(g_dst):
        g_dst[gid] = factor * g_src[gid]


@cute.jit
def scale(
    m_src: cute.Tensor,
    m_dst: cute.Tensor,
    factor: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
):
    n = cute.size(m_dst)
    blocks = (n + threads_per_block - 1) // threads_per_block
    scale_kernel(m_src, m_dst, factor).launch(
        grid=[blocks, 1, 1],
        block=[threads_per_block, 1, 1],
        max_number_threads=[256, 1, 1],
        min_blocks_per_mp=1,
    )


def run(n: int, factor: float, threads: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 07 needs an SM80+ CUDA device.")
    if threads not in (64, 128, 256):
        raise ValueError("--threads must be 64, 128, or 256")

    cutlass.cuda.initialize_cuda_context()
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    m_src = from_dlpack(src, assumed_align=16).mark_layout_dynamic()
    m_dst = from_dlpack(dst, assumed_align=16).mark_layout_dynamic()
    scale(m_src, m_dst, factor, threads)
    torch.cuda.synchronize()

    torch.testing.assert_close(dst, factor * src)
    blocks = (n + threads - 1) // threads
    print(f"OK -- grid={blocks}, block={threads}, elements={n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4099)
    parser.add_argument("--factor", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=128)
    args = parser.parse_args()
    run(args.n, args.factor, args.threads)
    print("Success.")
