#!/usr/bin/env python3
"""Day 09 solution: persistent matrix multiplication."""

import torch
import triton
import triton.language as tl


@triton.jit
def persistent_matmul_kernel(
    a, b, c, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    NUM_SMS: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr,
):
    start = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK)
    grid_n = tl.cdiv(N, BLOCK)
    total_tiles = grid_m * grid_n
    k_lane = tl.arange(0, BLOCK_K)
    for tile in tl.range(start, total_tiles, NUM_SMS, flatten=True):
        pid_m = tile // grid_n
        pid_n = tile % grid_n
        rows = pid_m * BLOCK + tl.arange(0, BLOCK)
        cols = pid_n * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK, BLOCK), tl.float32)
        for start_k in range(0, K, BLOCK_K):
            ks = start_k + k_lane
            av = tl.load(
                a + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0,
            )
            bv = tl.load(
                b + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
                mask=(ks[:, None] < K) & (cols[None, :] < N), other=0.0,
            )
            acc = tl.dot(av, bv, acc)
        tl.store(
            c + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
            acc.to(tl.float16),
            mask=(rows[:, None] < M) & (cols[None, :] < N),
        )


def persistent_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype != torch.float16 or b.dtype != torch.float16 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible float16 matrices")
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    total_tiles = triton.cdiv(M, 64) * triton.cdiv(N, 64)
    persistent_matmul_kernel[(min(num_sms, total_tiles),)](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=num_sms, BLOCK=64, BLOCK_K=32, num_warps=4, num_stages=3,
    )
    return c


def run() -> None:
    for shape in ((64, 64, 64), (513, 321, 777)):
        M, K, N = shape
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        torch.testing.assert_close(persistent_matmul(a, b), a @ b, atol=3e-2, rtol=3e-2)
    print("Day 09 persistent matmul: correctness OK")


if __name__ == "__main__":
    run()
