#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 01 — CuTe DSL Hello World (solution).

Two kernels:

  1. ``hello_world``     prints from thread 0 of block 0 to verify the toolchain.
  2. ``write_thread_id`` has every thread write its global linear thread index
     into a 1-D tensor; the host verifies the result against ``torch.arange``.

This is the smallest-possible CuTe-DSL program that actually touches GPU memory.
Concepts: ``@cute.kernel``, ``@cute.jit``, ``cute.arch.thread_idx`` /
``block_idx`` / ``block_dim``, ``cute.printf``, kernel ``.launch(grid=, block=)``,
``from_dlpack`` to wrap a PyTorch tensor as a ``cute.Tensor``.
"""
import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# -----------------------------------------------------------------------------
# Part 1: hello world
# -----------------------------------------------------------------------------
@cute.kernel
def hello_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    # Only thread 0 of block 0 prints — otherwise we'd get 32 copies of the line.
    # (DSL 4.5+: plain Python `if` is traced as a dynamic branch automatically.)
    if tidx == 0 and bidx == 0:
        cute.printf("hello from CuTe DSL: tidx=%d bidx=%d", tidx, bidx)


@cute.jit
def hello_world():
    # `print` here runs at JIT *compile* time (Python interpreter trace), while
    # `cute.printf` inside the kernel runs at device runtime. Both are useful.
    print("[JIT] launching hello_kernel with grid=(1,1,1) block=(32,1,1)")
    hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


# -----------------------------------------------------------------------------
# Part 2: every thread writes its global linear thread index
# -----------------------------------------------------------------------------
@cute.kernel
def write_tid_kernel(gA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    # Global linear thread index across the whole grid.
    gid = bidx * bdim + tidx

    # Bounds check — host launches just enough threads to cover gA.size, but
    # being defensive here lets readers reuse the kernel for non-divisible sizes.
    if gid < cute.size(gA):
        gA[gid] = gid.to(gA.element_type)


@cute.jit
def write_thread_id(mA: cute.Tensor):
    threads_per_block: cutlass.Constexpr = 128
    n = cute.size(mA)

    # `cute.ceil_div` would also work; using Python int math is fine because
    # we pass `mA` with `mark_layout_dynamic()` so only the shape is dynamic.
    num_blocks = (n + threads_per_block - 1) // threads_per_block

    print(f"[JIT] tensor size={mA.type}, launching {num_blocks} blocks "
          f"of {threads_per_block} threads")
    write_tid_kernel(mA).launch(
        grid=(num_blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def run(n: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 01 needs a CUDA device.")

    cutlass.cuda.initialize_cuda_context()

    print("\n=== Part 1: hello_world ===")
    hello_world()

    print("\n=== Part 2: write_thread_id ===")
    a = torch.full((n,), -1, device="cuda", dtype=torch.int32)
    write_thread_id(from_dlpack(a))
    torch.cuda.synchronize()

    expected = torch.arange(n, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(a, expected)
    print(f"OK — wrote {n} thread ids, first 8 = {a[:8].tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024, help="output tensor length")
    args = p.parse_args()
    run(args.n)
    print("\nSuccess.")
