#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 07 -- CuTe DSL kernel launch configuration (puzzle)."""

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
    # TODO(1): Compute a global 1-D thread id, bounds-check it, and store
    # factor * g_src[gid] into g_dst[gid].
    raise NotImplementedError("Day 07 TODO(1): implement scale_kernel")


@cute.jit
def scale(
    m_src: cute.Tensor,
    m_dst: cute.Tensor,
    factor: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
):
    n = cute.size(m_dst)
    blocks = (n + threads_per_block - 1) // threads_per_block

    # TODO(2): Launch scale_kernel with:
    #
    #   grid=[blocks, 1, 1]
    #   block=[threads_per_block, 1, 1]
    #   max_number_threads=[256, 1, 1]
    #   min_blocks_per_mp=1
    #
    # `block` selects this launch's actual dimensions. `max_number_threads`
    # emits a maxntid bound for the compiled kernel, while min_blocks_per_mp
    # emits an occupancy hint (minctasm).
    raise NotImplementedError("Day 07 TODO(2): provide the launch metadata")


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
