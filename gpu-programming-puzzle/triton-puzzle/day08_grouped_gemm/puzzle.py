#!/usr/bin/env python3
"""Day 08 puzzle: schedule differently sized GEMMs from device pointer arrays."""

import torch
import triton
import triton.language as tl


@triton.jit
def grouped_matmul_kernel(
    a_ptrs, b_ptrs, c_ptrs, sizes, strides, group_size,
    NUM_SM: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # EXERCISE(1): walk problem tile ranges, recover typed pointers, compute
    # owned tiles, and advance tile_idx by NUM_SM.
    pass


def grouped_matmul(group_a, group_b):
    raise NotImplementedError("EXERCISE(2): build device pointer/metadata arrays and launch")


def run() -> None:
    sizes = (128, 192, 256)
    a = [torch.randn((n, n), device="cuda", dtype=torch.float16) for n in sizes]
    b = [torch.randn_like(x) for x in a]
    grouped_matmul(a, b)
    print("Day 08 grouped GEMM: launch completed")


if __name__ == "__main__":
    run()
