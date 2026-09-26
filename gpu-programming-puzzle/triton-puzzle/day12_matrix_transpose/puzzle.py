#!/usr/bin/env python3
"""Day 12 puzzle: tiled matrix transpose."""

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    x_ptr, y_ptr, M, N, stride_xm, stride_xn, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # EXERCISE(1): load an MxN tile and store it at swapped coordinates.
    pass


def transpose(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("EXERCISE(2): allocate (N,M) and launch a 2-D grid")


def run() -> None:
    x = torch.randn((1003, 1507), device="cuda")
    torch.testing.assert_close(transpose(x), x.T.contiguous())
    print("Day 12 transpose: correctness OK")


if __name__ == "__main__":
    run()
