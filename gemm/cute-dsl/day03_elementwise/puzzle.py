#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 03 — CuTe DSL Elementwise Add (puzzle).

Two TODOs:

  1. Naive: each thread handles one element.
  2. Vectorized: each thread handles 8 contiguous elements via zipped_divide.
"""
import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# -----------------------------------------------------------------------------
# 1. naive — fill in the kernel body.
# -----------------------------------------------------------------------------
@cute.kernel
def naive_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    # TODO(1): each thread should:
    #   - compute its global linear id (bidx*bdim + tidx)
    #   - convert to (mi, ni) where m, n = gA.shape   and ni varies fastest
    #   - guard with `if mi < m:` (DSL 4.5+ traces plain Python ifs as dynamic)
    #   - do gC[mi, ni] = gA[mi, ni] + gB[mi, ni]
    raise NotImplementedError("Day 03 part 1: naive kernel")


@cute.jit
def naive_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block: cutlass.Constexpr = 256
    n_elem = cute.size(mA)
    grid = (n_elem + threads_per_block - 1) // threads_per_block
    naive_kernel(mA, mB, mC).launch(
        grid=(grid, 1, 1), block=(threads_per_block, 1, 1)
    )


# -----------------------------------------------------------------------------
# 2. vectorized — `zipped_divide` partitions the tensor so each thread sees a
#    (1, 8) sub-tile; `.load()` on a sub-tile emits a single vector load.
# -----------------------------------------------------------------------------
@cute.kernel
def vectorized_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    # After the host called `cute.zipped_divide(mA, (1, 8))`, this kernel sees
    # gA.shape == ((1, 8), (M, N // 8)).
    #
    # TODO(2):
    #   - compute thread_id = bidx*bdim + tidx
    #   - m, n = gA.shape[1]
    #   - mi = thread_id // n ;  ni = thread_id % n
    #   - load via:  a_vec = gA[(None, (mi, ni))].load()
    #   - store via: gC[(None, (mi, ni))] = a_vec + b_vec
    raise NotImplementedError("Day 03 part 2: vectorized kernel")


@cute.jit
def vectorized_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block: cutlass.Constexpr = 256
    gA = cute.zipped_divide(mA, (1, 8))
    gB = cute.zipped_divide(mB, (1, 8))
    gC = cute.zipped_divide(mC, (1, 8))

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
