#!/usr/bin/env python3
"""Day 10 puzzle: block-scaled matrix multiplication semantics."""

import torch
import triton
import triton.language as tl


@triton.jit
def block_scaled_matmul_kernel(
    a, b, scale_a, scale_b, c, M, N, K,
    BLOCK: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # EXERCISE(1): load one K block, broadcast its row/column scales,
    # scale operands before tl.dot, and accumulate.
    pass


def block_scaled_matmul(a, b, scale_a, scale_b):
    raise NotImplementedError("EXERCISE(2): validate scale shapes and launch")


def run() -> None:
    a = torch.randn((128, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((256, 192), device="cuda", dtype=torch.float16)
    sa = torch.rand((128, 8), device="cuda")
    sb = torch.rand((8, 192), device="cuda")
    block_scaled_matmul(a, b, sa, sb)
    print("Day 10 block-scaled matmul: launch completed")


if __name__ == "__main__":
    run()
