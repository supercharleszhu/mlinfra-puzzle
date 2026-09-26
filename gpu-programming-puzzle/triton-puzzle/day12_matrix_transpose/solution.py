#!/usr/bin/env python3
"""Day 12 solution: tiled matrix transpose."""

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    x_ptr, y_ptr, M, N, stride_xm, stride_xn, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tile = tl.load(
        x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn,
        mask=mask,
    )
    tl.store(
        y_ptr + cols[None, :] * stride_ym + rows[:, None] * stride_yn,
        tile,
        mask=mask,
    )


def transpose(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D tensor")
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    transpose_kernel[grid](
        x, y, M, N,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=32, BLOCK_N=32, num_warps=8,
    )
    return y


def run() -> None:
    for shape in ((32, 32), (64, 96), (1003, 1507)):
        x = torch.randn(shape, device="cuda")
        torch.testing.assert_close(transpose(x), x.T.contiguous())
    print("Day 12 transpose: correctness OK")


if __name__ == "__main__":
    run()
