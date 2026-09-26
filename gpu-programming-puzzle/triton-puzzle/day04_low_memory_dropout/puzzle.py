#!/usr/bin/env python3
"""Day 04 puzzle: deterministic low-memory dropout."""

import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(x_ptr, out_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    # EXERCISE(1): generate counter-based random values from seed+offset,
    # create the keep mask, and apply inverted-dropout scaling.
    pass


def dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    raise NotImplementedError("EXERCISE(2): validate p and launch dropout_kernel")


def run() -> None:
    x = torch.ones(100_003, device="cuda")
    a = dropout(x, 0.25, 123)
    b = dropout(x, 0.25, 123)
    torch.testing.assert_close(a, b)
    print("Day 04 dropout: deterministic correctness OK")


if __name__ == "__main__":
    run()
