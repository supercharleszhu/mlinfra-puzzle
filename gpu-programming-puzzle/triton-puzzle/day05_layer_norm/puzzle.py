#!/usr/bin/env python3
"""Day 05 puzzle: LayerNorm forward and backward reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_fwd_kernel(x, w, b, y, mean, rstd, N, eps, BLOCK: tl.constexpr):
    # EXERCISE(1): compute row mean/variance, save statistics, normalize,
    # apply affine parameters, and store.
    pass


@triton.jit
def layer_norm_dx_kernel(dy, x, w, mean, rstd, dx, N, BLOCK: tl.constexpr):
    # EXERCISE(2): implement dx using the two row reductions c1 and c2.
    pass


@triton.jit
def layer_norm_param_grads_kernel(
    dy, x, mean, rstd, dw, db, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # EXERCISE(3): reduce partial dw/db over rows for one column block.
    pass


def layer_norm(x, normalized_shape, weight, bias, eps):
    raise NotImplementedError("EXERCISE(4): wrap kernels in torch.autograd.Function")


def run() -> None:
    x = torch.randn((257, 1003), device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(1003, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.randn(1003, device="cuda", dtype=torch.float16, requires_grad=True)
    layer_norm(x, (1003,), w, b, 1e-5).sum().backward()
    print("Day 05 LayerNorm: forward/backward completed")


if __name__ == "__main__":
    run()
