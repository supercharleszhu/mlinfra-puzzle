#!/usr/bin/env python3
"""Day 01 puzzle: vector addition with a masked Triton load/store."""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # EXERCISE(1): derive this program's offsets, build the boundary mask,
    # load x/y, add them, and store the result.
    pass


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("EXERCISE(2): allocate output and launch vector_add_kernel")


def run() -> None:
    x = torch.randn(100_003, device="cuda")
    y = torch.randn_like(x)
    torch.testing.assert_close(vector_add(x, y), x + y)
    print("Day 01 vector add: correctness OK")


if __name__ == "__main__":
    run()
