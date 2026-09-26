#!/usr/bin/env python3
"""Day 10 solution: portable block-scaled matrix multiplication semantics."""

import torch
import triton
import triton.language as tl


@triton.jit
def block_scaled_matmul_kernel(
    a, b, scale_a, scale_b, c, M, N, K,
    BLOCK: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK + tl.arange(0, BLOCK)
    cols = pid_n * BLOCK + tl.arange(0, BLOCK)
    k_lane = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    for block_k in range(0, K, BLOCK_K):
        ks = block_k + k_lane
        av = tl.load(
            a + rows[:, None] * K + ks[None, :],
            mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0,
        )
        bv = tl.load(
            b + ks[:, None] * N + cols[None, :],
            mask=(ks[:, None] < K) & (cols[None, :] < N), other=0.0,
        )
        scale_index = block_k // BLOCK_K
        sa = tl.load(scale_a + rows * tl.cdiv(K, BLOCK_K) + scale_index, mask=rows < M, other=0.0)
        sb = tl.load(scale_b + scale_index * N + cols, mask=cols < N, other=0.0)
        av = av * sa[:, None]
        bv = bv * sb[None, :]
        acc = tl.dot(av.to(tl.float16), bv.to(tl.float16), acc)
    tl.store(
        c + rows[:, None] * N + cols[None, :],
        acc.to(tl.float16),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def block_scaled_matmul(a, b, scale_a, scale_b):
    if a.dtype != torch.float16 or b.dtype != torch.float16 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible contiguous float16 matrices")
    M, K = a.shape
    _, N = b.shape
    blocks_k = triton.cdiv(K, 32)
    if scale_a.shape != (M, blocks_k) or scale_b.shape != (blocks_k, N):
        raise ValueError("scale_a/scale_b shapes must match 32-element K blocks")
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    block_scaled_matmul_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        a, b, scale_a, scale_b, c, M, N, K,
        BLOCK=64, BLOCK_K=32, num_warps=4, num_stages=3,
    )
    return c


def reference(a, b, scale_a, scale_b):
    pieces = []
    for block in range(scale_a.shape[1]):
        start = block * 32
        end = min(start + 32, a.shape[1])
        pieces.append(
            (a[:, start:end].float() * scale_a[:, block, None])
            @ (b[start:end, :].float() * scale_b[block, None, :])
        )
    return sum(pieces).to(a.dtype)


def run() -> None:
    M, K, N = 129, 224, 193
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    scale_a = torch.rand((M, triton.cdiv(K, 32)), device="cuda")
    scale_b = torch.rand((triton.cdiv(K, 32), N), device="cuda")
    torch.testing.assert_close(
        block_scaled_matmul(a, b, scale_a, scale_b),
        reference(a, b, scale_a, scale_b),
        atol=5e-2, rtol=5e-2,
    )
    print("Day 10 block-scaled matmul semantics: correctness OK")


if __name__ == "__main__":
    run()
