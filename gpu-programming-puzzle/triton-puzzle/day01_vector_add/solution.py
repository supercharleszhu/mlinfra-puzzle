#!/usr/bin/env python3
"""Day 01 solution: vector addition with a masked Triton load/store."""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape or not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous tensors with equal shapes")
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 256),)
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=256)
    return out


def run() -> None:
    for size in (1, 257, 100_003):
        x = torch.randn(size, device="cuda")
        y = torch.randn_like(x)
        torch.testing.assert_close(vector_add(x, y), x + y)
    print("Day 01 vector add: correctness OK")


if __name__ == "__main__":
    run()
