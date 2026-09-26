#!/usr/bin/env python3
"""Day 07 puzzle: call the backend libdevice asin implementation."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def asin_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # EXERCISE(1): masked-load x, call libdevice.asin, and masked-store output.
    pass


def asin(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("EXERCISE(2): launch asin_kernel")


def run() -> None:
    x = torch.linspace(-1, 1, 100_003, device="cuda")
    torch.testing.assert_close(asin(x), torch.asin(x), atol=2e-6, rtol=2e-6)
    print("Day 07 libdevice asin: correctness OK")


if __name__ == "__main__":
    run()
