#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 11 - implement row-wise softmax with local and warp reductions."""

from __future__ import annotations

import argparse

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack


WARP_SIZE = 32


def warp_reduce_max(value: cutlass.Float32) -> cutlass.Float32:
    # TODO(1): combine all 32 lanes with shuffle_sync_bfly and cute.arch.fmax.
    raise NotImplementedError("Day 11 TODO(1): warp max reduction")


def warp_reduce_sum(value: cutlass.Float32) -> cutlass.Float32:
    # TODO(2): combine all 32 lanes with shuffle_sync_bfly and addition.
    raise NotImplementedError("Day 11 TODO(2): warp sum reduction")


class WarpSoftmax:
    """Compile-time-specialized one-warp-per-row softmax."""

    def __init__(self, n_cols: int) -> None:
        if n_cols <= 0:
            raise ValueError("n_cols must be positive")
        self.n_cols = n_cols
        self.values_per_lane = (n_cols + WARP_SIZE - 1) // WARP_SIZE

    @cute.jit
    def __call__(self, m_x: cute.Tensor, m_y: cute.Tensor) -> None:
        if cutlass.const_expr(
            m_x.element_type != cutlass.Float32
            or m_y.element_type != cutlass.Float32
        ):
            raise TypeError("Day 11 supports Float32 tensors")
        self.kernel(m_x, m_y).launch(
            grid=(m_x.shape[0], 1, 1),
            block=(WARP_SIZE, 1, 1),
        )

    @cute.kernel
    def kernel(self, g_x: cute.Tensor, g_y: cute.Tensor) -> None:
        lane, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        # TODO(3):
        # 1. Create a Float32 register tensor with self.values_per_lane values.
        # 2. Load columns lane + item * 32; pad OOB entries with -infinity.
        # 3. Use TensorSSA.reduce(MAX), then warp_reduce_max.
        # 4. Exponentiate values - row_max.
        # 5. Use TensorSSA.reduce(ADD), then warp_reduce_sum.
        # 6. Normalize and store only in-bounds columns.
        raise NotImplementedError("Day 11 TODO(3): stable row softmax")


def _validate_input(x) -> None:
    import torch

    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise TypeError("x must have dtype torch.float32")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D tensor")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x dimensions must be non-zero")
    if x.shape[1] > 4096:
        raise ValueError("this one-warp lesson supports at most 4096 columns")


def prepare_softmax(x):
    import torch

    _validate_input(x)
    y = torch.empty_like(x)
    x_cute = from_dlpack(x, assumed_align=4)
    y_cute = from_dlpack(y, assumed_align=4)
    compiled = cute.compile(WarpSoftmax(x.shape[1]), x_cute, y_cute)
    return y, compiled, x_cute, y_cute


def softmax(x):
    y, compiled, x_cute, y_cute = prepare_softmax(x)
    compiled(x_cute, y_cute)
    return y


def test_softmax() -> None:
    import torch

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
    print("Day 11 CuTe softmax: correctness OK")


def benchmark_softmax(
    rows: int,
    columns: int,
    warmup: int = 25,
    iterations: int = 100,
) -> tuple[float, float]:
    import torch

    x = torch.randn(rows, columns, device="cuda", dtype=torch.float32)
    y, compiled, x_cute, y_cute = prepare_softmax(x)
    compiled(x_cute, y_cute)
    torch.cuda.synchronize()
    average_us = testing.benchmark(
        compiled,
        kernel_arguments=testing.JitArguments(x_cute, y_cute),
        warmup_iterations=warmup,
        iterations=iterations,
    )
    bytes_moved = 2 * x.numel() * x.element_size()
    return average_us, bytes_moved / (average_us * 1_000)


def benchmark_suite(rows: int = 4096) -> None:
    print("columns | CuTe us | effective GB/s")
    print("------- | ------- | --------------")
    for columns in (128, 512, 1003, 1024):
        average_us, bandwidth = benchmark_softmax(rows, columns)
        print(f"{columns:7d} | {average_us:7.2f} | {bandwidth:14.2f}")


def run(rows: int, benchmark: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 11 needs a CUDA device")
    cutlass.cuda.initialize_cuda_context()
    test_softmax()
    if benchmark:
        benchmark_suite(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    run(args.rows, args.benchmark)
