#!/usr/bin/env python3
"""Day 02 puzzle: one-program-per-row fused softmax."""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    # EXERCISE(1): load one padded row, reduce max and sum, then store softmax.
    pass


def softmax(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("EXERCISE(2): choose a power-of-two block and launch one program per row")


def run() -> None:
    x = torch.randn((257, 1003), device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1), atol=1e-6, rtol=1e-5)
    print("Day 02 softmax: correctness OK")


if __name__ == "__main__":
    run()
