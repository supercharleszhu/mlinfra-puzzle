#!/usr/bin/env python3
"""Day 08 solution: schedule differently sized GEMMs from device pointer arrays."""

import torch
import triton
import triton.language as tl


@triton.jit
def grouped_matmul_kernel(
    a_ptrs, b_ptrs, c_ptrs, sizes, strides, group_size,
    NUM_SM: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    problem_start = 0
    for group in range(group_size):
        M = tl.load(sizes + group * 3)
        N = tl.load(sizes + group * 3 + 1)
        K = tl.load(sizes + group * 3 + 2)
        grid_m = tl.cdiv(M, BLOCK)
        grid_n = tl.cdiv(N, BLOCK)
        problem_tiles = grid_m * grid_n
        while tile_idx >= problem_start and tile_idx < problem_start + problem_tiles:
            local = tile_idx - problem_start
            pid_m = local // grid_n
            pid_n = local % grid_n
            lda = tl.load(strides + group * 3)
            ldb = tl.load(strides + group * 3 + 1)
            ldc = tl.load(strides + group * 3 + 2)
            a = tl.load(a_ptrs + group).to(tl.pointer_type(tl.float16))
            b = tl.load(b_ptrs + group).to(tl.pointer_type(tl.float16))
            c = tl.load(c_ptrs + group).to(tl.pointer_type(tl.float16))
            offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
            offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK, BLOCK), tl.float32)
            for k_start in range(0, K, BLOCK_K):
                k = k_start + offs_k
                av = tl.load(
                    a + offs_m[:, None] * lda + k[None, :],
                    mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0,
                )
                bv = tl.load(
                    b + k[:, None] * ldb + offs_n[None, :],
                    mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
                )
                acc = tl.dot(av, bv, acc)
            tl.store(
                c + offs_m[:, None] * ldc + offs_n[None, :],
                acc.to(tl.float16),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
            tile_idx += NUM_SM
        problem_start += problem_tiles


def grouped_matmul(group_a, group_b):
    if len(group_a) != len(group_b) or not group_a:
        raise ValueError("groups must be non-empty and equally sized")
    outputs = []
    a_addrs, b_addrs, c_addrs, sizes, strides = [], [], [], [], []
    for a, b in zip(group_a, group_b):
        if a.dtype != torch.float16 or b.dtype != torch.float16 or a.shape[1] != b.shape[0]:
            raise ValueError("each problem must be a compatible float16 GEMM")
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        outputs.append(c)
        a_addrs.append(a.data_ptr())
        b_addrs.append(b.data_ptr())
        c_addrs.append(c.data_ptr())
        sizes.extend((M, N, K))
        strides.extend((a.stride(0), b.stride(0), c.stride(0)))
    device = group_a[0].device
    metadata = (
        torch.tensor(a_addrs, device=device, dtype=torch.int64),
        torch.tensor(b_addrs, device=device, dtype=torch.int64),
        torch.tensor(c_addrs, device=device, dtype=torch.int64),
        torch.tensor(sizes, device=device, dtype=torch.int32),
        torch.tensor(strides, device=device, dtype=torch.int32),
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    grouped_matmul_kernel[(num_sms,)](
        *metadata, len(group_a), NUM_SM=num_sms, BLOCK=64, BLOCK_K=32,
        num_warps=4, num_stages=3,
    )
    return outputs


def run() -> None:
    shapes = ((128, 96, 192), (257, 128, 193), (64, 224, 96))
    group_a = [torch.randn((M, K), device="cuda", dtype=torch.float16) for M, K, _ in shapes]
    group_b = [torch.randn((K, N), device="cuda", dtype=torch.float16) for _, K, N in shapes]
    outputs = grouped_matmul(group_a, group_b)
    for out, a, b in zip(outputs, group_a, group_b):
        torch.testing.assert_close(out, a @ b, atol=3e-2, rtol=3e-2)
    print("Day 08 grouped GEMM: correctness OK")


if __name__ == "__main__":
    run()
