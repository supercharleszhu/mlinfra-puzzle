#!/usr/bin/env python3
"""Day 13 solution: stable row softmax with tests and benchmarks."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    values = tl.load(
        x_ptr + row * n_cols + columns,
        mask=mask,
        other=-float("inf"),
    )
    values -= tl.max(values, axis=0)
    numerators = tl.exp(values)
    denominator = tl.sum(numerators, axis=0)
    tl.store(
        y_ptr + row * n_cols + columns,
        numerators / denominator,
        mask=mask,
    )


def _validate_input(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise TypeError("x must have dtype torch.float32")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D tensor")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x dimensions must be non-zero")
    if x.shape[1] > 65_536:
        raise ValueError("row is too wide for this one-program lesson")


def launch_softmax(x: torch.Tensor, y: torch.Tensor) -> None:
    _validate_input(x)
    if y.shape != x.shape or y.dtype != x.dtype or not y.is_contiguous():
        raise ValueError("y must be a contiguous tensor matching x")
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(max(block_size // 256, 1), 8)
    softmax_kernel[(n_rows,)](
        x,
        y,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def softmax(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    launch_softmax(x, y)
    return y


def test_softmax() -> None:
    torch.manual_seed(20260915)
    for shape in ((1, 1), (7, 31), (33, 128), (65, 512), (257, 1003)):
        x = torch.randn(shape, device="cuda", dtype=torch.float32) * 7.0
        actual = softmax(x)
        expected = torch.softmax(x, dim=1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            actual.sum(dim=1),
            torch.ones(shape[0], device="cuda"),
            atol=2e-6,
            rtol=2e-6,
        )

    extreme = torch.tensor(
        [[-1000.0, 0.0, 1000.0], [1000.0, 1000.0, 999.0]],
        device="cuda",
    )
    torch.testing.assert_close(
        softmax(extreme),
        torch.softmax(extreme, dim=1),
        atol=2e-6,
        rtol=2e-5,
    )
    print("Day 13 Triton softmax: correctness OK")


def benchmark_softmax(
    rows: int,
    columns: int,
    warmup: int = 25,
    repetitions: int = 100,
) -> tuple[float, float, float]:
    x = torch.randn(rows, columns, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    launch_softmax(x, y)
    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(
        lambda: launch_softmax(x, y),
        warmup=warmup,
        rep=repetitions,
    )
    torch_ms = triton.testing.do_bench(
        lambda: torch.softmax(x, dim=1),
        warmup=warmup,
        rep=repetitions,
    )
    bytes_moved = 2 * x.numel() * x.element_size()
    bandwidth_gbps = bytes_moved / (triton_ms * 1e6)
    return triton_ms * 1_000, bandwidth_gbps, torch_ms * 1_000


def benchmark_suite(rows: int = 4096) -> None:
    print("columns | Triton us | Torch us | effective GB/s")
    print("------- | --------- | -------- | --------------")
    for columns in (128, 512, 1003, 1024):
        triton_us, bandwidth, torch_us = benchmark_softmax(rows, columns)
        print(
            f"{columns:7d} | {triton_us:9.2f} | {torch_us:8.2f} | "
            f"{bandwidth:14.2f}"
        )


def run(rows: int, benchmark: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Day 13 needs a CUDA device")
    test_softmax()
    if benchmark:
        benchmark_suite(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    run(args.rows, args.benchmark)
