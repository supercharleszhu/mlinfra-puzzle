#!/usr/bin/env python3
"""Day 02 solution: one-program-per-row fused softmax."""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    values = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(y_ptr + row * n_cols + cols, numerator / denominator, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D tensor")
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65_536:
        raise ValueError("row is too wide for this fused lesson")
    y = torch.empty_like(x)
    num_warps = min(max(block_size // 256, 1), 8)
    softmax_kernel[(n_rows,)](
        x, y, n_cols, BLOCK_SIZE=block_size, num_warps=num_warps
    )
    return y


def run() -> None:
    for shape in ((1, 17), (257, 1003), (1024, 4096)):
        x = torch.randn(shape, device="cuda")
        torch.testing.assert_close(
            softmax(x), torch.softmax(x, dim=1), atol=1e-6, rtol=1e-5
        )
    print("Day 02 softmax: correctness OK")


if __name__ == "__main__":
    run()
