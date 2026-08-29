#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 03 — CuTe DSL Elementwise Add (solution).

Two kernels, in increasing sophistication:

  1. ``naive_elementwise_add``      — 1 element per thread, scalar load/store.
  2. ``vectorized_elementwise_add`` — 8 elements per thread via ``zipped_divide``,
     resulting in 128-bit (16B) vector load/stores on Ampere+.

This is the canonical first kernel in the CuTe DSL tutorial. Mirrors the
official ``notebooks/elementwise_add.ipynb`` from CUTLASS.
"""
import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# -----------------------------------------------------------------------------
# 1. naive: each thread handles exactly one (m, n) element.
# -----------------------------------------------------------------------------
@cute.kernel
def naive_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    gid = bidx * bdim + tidx
    m, n = gA.shape
    mi = gid // n
    ni = gid % n
    # Bounds-check so we work on non-divisible sizes too.
    if mi < m:
        gC[mi, ni] = gA[mi, ni] + gB[mi, ni]


@cute.jit
def naive_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block: cutlass.Constexpr = 256
    n_elem = cute.size(mA)
    grid = (n_elem + threads_per_block - 1) // threads_per_block
    print(f"[JIT naive] launch grid={grid} block={threads_per_block}")
    naive_kernel(mA, mB, mC).launch(
        grid=(grid, 1, 1), block=(threads_per_block, 1, 1)
    )


# -----------------------------------------------------------------------------
# 2. vectorized: zipped_divide so each thread sees a (1, 8) sub-tensor.
#    `.load()` on a sub-tensor emits a single vector load.
# -----------------------------------------------------------------------------
@cute.kernel
def vectorized_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_id = bidx * bdim + tidx
    # After zipped_divide, gA.shape == ((1, 8), (M, N//8)).
    # We index the *outer* mode (M, N//8) by 2D thread coords.
    m, n = gA.shape[1]
    mi = thread_id // n
    ni = thread_id % n

    # `(None, (mi, ni))` keeps the inner (1, 8) tile, slices the outer.
    a_vec = gA[(None, (mi, ni))].load()
    b_vec = gB[(None, (mi, ni))].load()
    gC[(None, (mi, ni))] = a_vec + b_vec


@cute.jit
def vectorized_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block: cutlass.Constexpr = 256

    # Each thread handles a (1, 8) tile -> 8 elements -> 128 bits at fp16.
    gA = cute.zipped_divide(mA, (1, 8))
    gB = cute.zipped_divide(mB, (1, 8))
    gC = cute.zipped_divide(mC, (1, 8))
    print(f"[JIT vec] gA after zipped_divide = {gA.type}")

    num_tiles = cute.size(gC, mode=[1])
    grid = (num_tiles + threads_per_block - 1) // threads_per_block
    vectorized_kernel(gA, gB, gC).launch(
        grid=(grid, 1, 1), block=(threads_per_block, 1, 1)
    )


def run(M: int, N: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 03 needs a CUDA device.")
    cutlass.cuda.initialize_cuda_context()

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros_like(a)

    a_, b_, c_ = (from_dlpack(t, assumed_align=16) for t in (a, b, c))

    print(f"\n=== naive elementwise add, ({M}, {N}) ===")
    naive_elementwise_add(a_, b_, c_)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)
    print("  OK")

    c.zero_()
    c_ = from_dlpack(c, assumed_align=16)
    print(f"\n=== vectorized elementwise add, ({M}, {N}) ===")
    vectorized_elementwise_add(a_, b_, c_)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)
    print("  OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--N", type=int, default=1024)
    args = p.parse_args()
    assert args.N % 8 == 0, "vectorized kernel needs N divisible by 8"
    run(args.M, args.N)
    print("\nSuccess.")
