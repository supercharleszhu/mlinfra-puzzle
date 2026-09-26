#!/usr/bin/env python3
"""Generate the explanation-rich Day 11 softmax notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


DSL_ROOT = Path(__file__).resolve().parents[1]
DAY_ROOT = DSL_ROOT / "day11_softmax_reduction"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def puzzle_source() -> str:
    source = (DAY_ROOT / "puzzle.py").read_text(encoding="utf-8")
    return source.split('\nif __name__ == "__main__":', 1)[0].strip()


def main() -> None:
    guide = (DAY_ROOT / "README.md").read_text(encoding="utf-8")
    cells = [
        markdown(guide),
        markdown(
            """
            ## Reduction checkpoint: local values are not warp values

            A lane first owns a static register tensor. Calling
            `TensorSSA.reduce` combines only that tensor:

            ```text
            lane 0: [x0, x32, x64, ...] -> partial_0
            lane 1: [x1, x33, x65, ...] -> partial_1
            ...
            lane31: [x31, x63, x95, ...] -> partial_31
            ```

            The five `shuffle_sync_bfly` steps then exchange partials at XOR
            distances 16, 8, 4, 2, and 1. After the fifth step, every lane has
            the same row-wide result. The kernel performs this hierarchy twice:
            first for the maximum and then for the exponential sum.
            """
        ),
        code(
            """
            def butterfly_partners(lane):
                return [lane ^ offset for offset in (16, 8, 4, 2, 1)]


            for lane in (0, 1, 17, 31):
                print(f"lane {lane:2d} exchanges with {butterfly_partners(lane)}")
            """
        ),
        markdown(
            """
            ## Challenge implementation

            Complete TODO(1)-TODO(3). The host wrapper specializes
            `values_per_lane = ceil(n_cols / 32)`, so the register tensor has a
            static shape. Padding uses the correct reduction identities:
            negative infinity for max and, after exponentiation, zero for sum.
            """
        ),
        code(puzzle_source()),
        markdown(
            """
            ## Correctness tests

            These tests include non-power-of-two widths and logits near
            floating-point overflow. Passing ordinary random values is not
            enough: the extreme test catches a missing max subtraction, and
            the row-sum assertion catches incorrect reduction scope.
            """
        ),
        code("test_softmax()"),
        markdown(
            """
            ## CuTe-only benchmark

            This compiles once per shape and measures only repeated compiled
            launches. Effective bandwidth counts one FP32 read and one FP32
            write. Run it after correctness passes.
            """
        ),
        code("benchmark_suite(rows=4096)"),
        markdown(
            """
            ## Matched Triton comparison

            Triton expresses the same stable algorithm with `tl.max` and
            `tl.sum`. The compiler owns the lane-level reduction strategy. The
            cell below uses the same tensors, output buffers, warmup,
            repetitions, and `triton.testing.do_bench` timing path for both
            kernels. This avoids comparing numbers gathered under different
            benchmark harnesses.
            """
        ),
        code(
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def triton_softmax_kernel(
                x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr
            ):
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


            def launch_triton_softmax(x, y):
                rows, columns = x.shape
                block_size = triton.next_power_of_2(columns)
                num_warps = min(max(block_size // 256, 1), 8)
                triton_softmax_kernel[(rows,)](
                    x,
                    y,
                    columns,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
            """
        ),
        code(
            """
            def compare_cute_and_triton(rows=4096, warmup=25, repetitions=100):
                print("columns | CuTe us | Triton us | CuTe GB/s | Triton GB/s")
                print("------- | ------- | --------- | --------- | -----------")
                for columns in (128, 512, 1003, 1024):
                    x = torch.randn(
                        rows, columns, device="cuda", dtype=torch.float32
                    )
                    cute_y, compiled, x_cute, y_cute = prepare_softmax(x)
                    triton_y = torch.empty_like(x)
                    compiled(x_cute, y_cute)
                    launch_triton_softmax(x, triton_y)
                    torch.cuda.synchronize()
                    reference = torch.softmax(x, dim=1)
                    torch.testing.assert_close(
                        cute_y, reference, atol=2e-6, rtol=2e-5
                    )
                    torch.testing.assert_close(
                        triton_y, reference, atol=2e-6, rtol=2e-5
                    )

                    cute_ms = triton.testing.do_bench(
                        lambda: compiled(x_cute, y_cute),
                        warmup=warmup,
                        rep=repetitions,
                    )
                    triton_ms = triton.testing.do_bench(
                        lambda: launch_triton_softmax(x, triton_y),
                        warmup=warmup,
                        rep=repetitions,
                    )
                    bytes_moved = 2 * x.numel() * x.element_size()
                    cute_gbps = bytes_moved / (cute_ms * 1e6)
                    triton_gbps = bytes_moved / (triton_ms * 1e6)
                    print(
                        f"{columns:7d} | {cute_ms * 1000:7.2f} | "
                        f"{triton_ms * 1000:9.2f} | {cute_gbps:9.2f} | "
                        f"{triton_gbps:11.2f}"
                    )


            compare_cute_and_triton()
            """
        ),
        markdown(
            """
            ## Interpret the result

            Do not reduce the comparison to language syntax. Check how the
            gap changes with width:

            - Small rows emphasize launch and scheduling overhead.
            - Wider rows increase CuTe's per-lane register fragment.
            - Width 1003 measures the cost of padded work and predicates.
            - Triton's `num_warps` changes at larger power-of-two blocks, while
              this CuTe kernel deliberately remains one warp per row.

            A production follow-up would tune warps per row, vectorize global
            accesses, and implement a shared-memory cross-warp reduction.
            """
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "CuTe DSL (H100)",
                "language": "python",
                "name": "cutlass-cute-dsl",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = DAY_ROOT / "day11.ipynb"
    destination.write_text(
        json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
