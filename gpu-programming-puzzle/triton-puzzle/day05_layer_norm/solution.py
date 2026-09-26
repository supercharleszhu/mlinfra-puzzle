#!/usr/bin/env python3
"""Day 05 solution: LayerNorm forward and backward reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_fwd_kernel(x, w, b, y, mean, rstd, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    values = tl.load(x + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mu = tl.sum(values, axis=0) / N
    centered = tl.where(mask, values - mu, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N
    inv_std = tl.rsqrt(variance + eps)
    weights = tl.load(w + cols, mask=mask)
    biases = tl.load(b + cols, mask=mask)
    tl.store(mean + row, mu)
    tl.store(rstd + row, inv_std)
    tl.store(y + row * N + cols, centered * inv_std * weights + biases, mask=mask)


@triton.jit
def layer_norm_dx_kernel(dy, x, w, mean, rstd, dx, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x_row = tl.load(x + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    dy_row = tl.load(dy + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(w + cols, mask=mask, other=0.0).to(tl.float32)
    mu = tl.load(mean + row)
    inv_std = tl.load(rstd + row)
    x_hat = tl.where(mask, (x_row - mu) * inv_std, 0.0)
    wdy = tl.where(mask, weights * dy_row, 0.0)
    c1 = tl.sum(x_hat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    tl.store(dx + row * N + cols, (wdy - x_hat * c1 - c2) * inv_std, mask=mask)


@triton.jit
def layer_norm_param_grads_kernel(
    dy, x, mean, rstd, dw, db, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    acc_w = tl.zeros((BLOCK_N,), tl.float32)
    acc_b = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, M, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M) & col_mask[None, :]
        offsets = rows[:, None] * N + cols[None, :]
        values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(dy + offsets, mask=mask, other=0.0).to(tl.float32)
        mu = tl.load(mean + rows, mask=rows < M, other=0.0)
        inv_std = tl.load(rstd + rows, mask=rows < M, other=0.0)
        x_hat = (values - mu[:, None]) * inv_std[:, None]
        acc_w += tl.sum(grad * x_hat, axis=0)
        acc_b += tl.sum(grad, axis=0)
    tl.store(dw + cols, acc_w, mask=col_mask)
    tl.store(db + cols, acc_b, mask=col_mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        if x.shape[-1] != normalized_shape[0]:
            raise ValueError("normalized_shape must match the final dimension")
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x2d.shape
        block = triton.next_power_of_2(N)
        if block * x.element_size() > 65_536:
            raise ValueError("feature row exceeds the 64KB lesson limit")
        y = torch.empty_like(x2d)
        mean = torch.empty(M, device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        warps = min(max(block // 256, 1), 8)
        layer_norm_fwd_kernel[(M,)](
            x2d, weight, bias, y, mean, rstd, N, eps,
            BLOCK=block, num_warps=warps,
        )
        ctx.save_for_backward(x2d, weight, mean, rstd)
        ctx.original_shape = x.shape
        ctx.block = block
        ctx.warps = warps
        return y.reshape_as(x)

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        dy2d = dy.reshape_as(x).contiguous()
        M, N = x.shape
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        db = torch.empty_like(weight)
        layer_norm_dx_kernel[(M,)](
            dy2d, x, weight, mean, rstd, dx, N,
            BLOCK=ctx.block, num_warps=ctx.warps,
        )
        layer_norm_param_grads_kernel[(triton.cdiv(N, 64),)](
            dy2d, x, mean, rstd, dw, db, M, N,
            BLOCK_M=8, BLOCK_N=64, num_warps=4,
        )
        return dx.reshape(ctx.original_shape), None, dw, db, None


layer_norm = LayerNorm.apply


def run() -> None:
    torch.manual_seed(0)
    x = torch.randn((257, 1003), device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(1003, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.randn(1003, device="cuda", dtype=torch.float16, requires_grad=True)
    dy = torch.randn_like(x)
    y = layer_norm(x, (1003,), w, b, 1e-5)
    y.backward(dy)
    actual = (y.detach(), x.grad.detach(), w.grad.detach(), b.grad.detach())
    x.grad = w.grad = b.grad = None
    ref = torch.nn.functional.layer_norm(x, (1003,), w, b, 1e-5)
    ref.backward(dy)
    expected = (ref.detach(), x.grad, w.grad, b.grad)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=3e-2, rtol=3e-2)
    print("Day 05 LayerNorm: forward/backward correctness OK")


if __name__ == "__main__":
    run()
