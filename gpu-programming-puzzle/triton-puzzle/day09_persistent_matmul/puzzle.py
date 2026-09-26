#!/usr/bin/env python3
"""Day 09 puzzle: persistent matrix multiplication."""

import torch
import triton
import triton.language as tl


@triton.jit
def persistent_matmul_kernel(
    a, b, c, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    NUM_SMS: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # EXERCISE(1): let each resident program process tile_id += NUM_SMS.
    pass


def persistent_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("EXERCISE(2): cap the launch grid at NUM_SMS")


def run() -> None:
    a = torch.randn((513, 321), device="cuda", dtype=torch.float16)
    b = torch.randn((321, 777), device="cuda", dtype=torch.float16)
    persistent_matmul(a, b)
    print("Day 09 persistent matmul: launch completed")


if __name__ == "__main__":
    run()
