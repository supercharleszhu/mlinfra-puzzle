#!/usr/bin/env python3
"""Day 04 solution: deterministic low-memory dropout."""

import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(x_ptr, out_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    keep = tl.rand(seed, offsets) > p
    out = tl.where(keep, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), 256),)
    dropout_kernel[grid](x, out, x.numel(), p, seed, BLOCK_SIZE=256)
    return out


def run() -> None:
    x = torch.ones(100_003, device="cuda")
    a = dropout(x, 0.25, 123)
    b = dropout(x, 0.25, 123)
    c = dropout(x, 0.25, 124)
    torch.testing.assert_close(a, b)
    if torch.equal(a, c):
        raise AssertionError("different seeds produced identical masks")
    if abs(a.float().mean().item() - 1.0) > 0.02:
        raise AssertionError("inverted-dropout mean is incorrect")
    print("Day 04 dropout: deterministic correctness OK")


if __name__ == "__main__":
    run()
