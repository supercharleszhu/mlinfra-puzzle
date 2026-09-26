#!/usr/bin/env python3
"""Day 03 puzzle: tiled matrix multiplication with grouped tile ordering."""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # EXERCISE(1): map pid with grouped M ordering, loop over K tiles, call
    # tl.dot, and mask the output store.
    pass


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("EXERCISE(2): validate shapes and launch matmul_kernel")


def run() -> None:
    a = torch.randn((257, 193), device="cuda", dtype=torch.float16)
    b = torch.randn((193, 311), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(matmul(a, b), a @ b, atol=2e-2, rtol=2e-2)
    print("Day 03 matmul: correctness OK")


if __name__ == "__main__":
    run()
