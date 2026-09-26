#!/usr/bin/env python3
"""Day 06 solution: compact fused attention with an explicit backward pass."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(q, k, v, out, scale, N: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr):
    bh = tl.program_id(0)
    rows = tl.arange(0, N)
    dims = tl.arange(0, D)
    base = bh * N * D
    offsets = base + rows[:, None] * D + dims[None, :]
    q_block = tl.load(q + offsets)
    k_block = tl.load(k + offsets)
    v_block = tl.load(v + offsets)
    scores = tl.dot(q_block, tl.trans(k_block)) * scale
    if CAUSAL:
        scores = tl.where(rows[:, None] >= rows[None, :], scores, -float("inf"))
    scores -= tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs /= tl.sum(probs, axis=1)[:, None]
    result = tl.dot(probs.to(tl.float16), v_block)
    tl.store(out + offsets, result)


@triton.jit
def attention_backward_kernel(
    q, k, v, out, dout, dq, dk, dv, scale,
    N: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
):
    bh = tl.program_id(0)
    rows = tl.arange(0, N)
    dims = tl.arange(0, D)
    base = bh * N * D
    offsets = base + rows[:, None] * D + dims[None, :]
    q_block = tl.load(q + offsets)
    k_block = tl.load(k + offsets)
    v_block = tl.load(v + offsets)
    out_block = tl.load(out + offsets)
    do_block = tl.load(dout + offsets)
    scores = tl.dot(q_block, tl.trans(k_block)) * scale
    if CAUSAL:
        scores = tl.where(rows[:, None] >= rows[None, :], scores, -float("inf"))
    scores -= tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs /= tl.sum(probs, axis=1)[:, None]
    probs16 = probs.to(tl.float16)
    dv_block = tl.dot(tl.trans(probs16), do_block)
    dp = tl.dot(do_block, tl.trans(v_block))
    delta = tl.sum(out_block * do_block, axis=1)
    ds = probs * (dp - delta[:, None])
    ds16 = ds.to(tl.float16)
    dq_block = tl.dot(ds16, k_block) * scale
    dk_block = tl.dot(tl.trans(ds16), q_block) * scale
    tl.store(dq + offsets, dq_block)
    tl.store(dk + offsets, dk_block)
    tl.store(dv + offsets, dv_block)


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
            raise ValueError("q, k, and v must share shape (B,H,N,D)")
        B, H, N, D = q.shape
        if N > 64 or D > 64 or N % 16 or D % 16:
            raise ValueError("this compact lesson requires N,D <= 64 and multiples of 16")
        out = torch.empty_like(q)
        scale = 1.0 / math.sqrt(D)
        attention_kernel[(B * H,)](q, k, v, out, scale, N=N, D=D, CAUSAL=causal, num_warps=8)
        ctx.save_for_backward(q, k, v, out)
        ctx.causal = causal
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out = ctx.saved_tensors
        B, H, N, D = q.shape
        dq, dk, dv = (torch.empty_like(q) for _ in range(3))
        attention_backward_kernel[(B * H,)](
            q, k, v, out, dout.contiguous(), dq, dk, dv, ctx.scale,
            N=N, D=D, CAUSAL=ctx.causal, num_warps=8,
        )
        return dq, dk, dv, None


attention = Attention.apply


def run() -> None:
    torch.manual_seed(0)
    for causal in (False, True):
        inputs = [
            torch.randn((2, 3, 32, 32), device="cuda", dtype=torch.float16, requires_grad=True)
            for _ in range(3)
        ]
        q, k, v = inputs
        dout = torch.randn_like(q)
        actual_y = attention(q, k, v, causal)
        actual_y.backward(dout)
        actual_grads = [tensor.grad.detach().clone() for tensor in inputs]
        for tensor in inputs:
            tensor.grad = None
        ref_y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=1.0 / math.sqrt(q.shape[-1])
        )
        ref_y.backward(dout)
        torch.testing.assert_close(actual_y, ref_y, atol=3e-2, rtol=3e-2)
        for got, tensor in zip(actual_grads, inputs):
            torch.testing.assert_close(got, tensor.grad, atol=4e-2, rtol=4e-2)
    print("Day 06 attention: forward/backward correctness OK")


if __name__ == "__main__":
    run()
