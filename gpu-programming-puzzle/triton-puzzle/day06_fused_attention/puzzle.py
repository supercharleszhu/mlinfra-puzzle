#!/usr/bin/env python3
"""Day 06 puzzle: compact fused attention with an explicit backward pass."""

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(q, k, v, out, scale, N: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr):
    # EXERCISE(1): load Q/K/V for one batch-head, form QK^T, apply stable
    # softmax and an optional causal mask, then multiply by V.
    pass


@triton.jit
def attention_backward_kernel(
    q, k, v, out, dout, dq, dk, dv, scale,
    N: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
):
    # EXERCISE(2): recompute probabilities and form dQ, dK, and dV.
    pass


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    raise NotImplementedError("EXERCISE(3): wrap forward/backward kernels in autograd")


def run() -> None:
    q = torch.randn((2, 3, 32, 32), device="cuda", dtype=torch.float16, requires_grad=True)
    y = attention(q, q, q, True)
    y.sum().backward()
    print("Day 06 attention: forward/backward completed")


if __name__ == "__main__":
    run()
