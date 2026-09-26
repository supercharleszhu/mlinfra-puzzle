#!/usr/bin/env python3
"""Day 07 solution: call the backend libdevice asin implementation."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def asin_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, libdevice.asin(x), mask=mask)


def asin(x: torch.Tensor) -> torch.Tensor:
    if x.dtype not in (torch.float32, torch.float64) or not x.is_contiguous():
        raise ValueError("x must be contiguous float32 or float64")
    out = torch.empty_like(x)
    asin_kernel[(triton.cdiv(x.numel(), 256),)](
        x, out, x.numel(), BLOCK_SIZE=256
    )
    return out


def run() -> None:
    for dtype in (torch.float32, torch.float64):
        x = torch.linspace(-1, 1, 100_003, device="cuda", dtype=dtype)
        torch.testing.assert_close(asin(x), torch.asin(x), atol=2e-6, rtol=2e-6)
    print("Day 07 libdevice asin: correctness OK")


if __name__ == "__main__":
    run()
