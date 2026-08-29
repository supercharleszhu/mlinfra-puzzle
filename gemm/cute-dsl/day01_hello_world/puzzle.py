#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 01 — CuTe DSL Hello World (puzzle).

Fill in the two TODO blocks.  When done, run:

    python puzzle.py
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
    # TODO(1): only have thread 0 of block 0 call cute.printf, so we don't get
    #          one line per thread.  You need:
    #            - cute.arch.thread_idx()           returns (tidx, tidy, tidz)
    #            - cute.arch.block_idx()            returns (bidx, bidy, bidz)
    #            - plain `if tidx == 0 and bidx == 0:` — the DSL traces it as
    #              a dynamic branch automatically.
    #            - cute.printf("...", arg, ...)     printf with C-style format
    if cute.arch.thread_idx()[0] == 0:
        cute.printf("heheehe")


@cute.jit
def hello_world():
    print("[JIT] launching hello_kernel")
    hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


# -----------------------------------------------------------------------------
# Part 2: every thread writes its global linear thread index into gA[gid].
# -----------------------------------------------------------------------------
@cute.kernel
def write_tid_kernel(gA: cute.Tensor):
    # TODO(2): for each thread, compute its global linear id and store it
    #          into gA[gid] (cast to gA.element_type).  You need:
    #            - cute.arch.thread_idx() / block_idx() / block_dim()
    #            - cute.size(gA)            total number of elements
    #            - gA[i] = value            store
    #            - .to(gA.element_type)     cast from Int32 to gA's dtype
    thread_id,_,_ = cute.arch.thread_idx()
    block_dim,_,_ = cute.arch.block_dim()
    block_id,_,_ = cute.arch.block_idx()
    idx = thread_id + block_id * block_dim
    if idx <= cute.size(gA):
        gA[idx] = idx.to(gA.element_type)


@cute.jit
def write_thread_id(mA: cute.Tensor):
    threads_per_block: cutlass.Constexpr = 128
    n = cute.size(mA)
    num_blocks = (n + threads_per_block - 1) // threads_per_block
    print(f"[JIT] tensor={mA.type}, {num_blocks} blocks x {threads_per_block} threads")
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
    print(f"OK — first 8 = {a[:8].tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    args = p.parse_args()
    run(args.n)
    print("\nSuccess.")
