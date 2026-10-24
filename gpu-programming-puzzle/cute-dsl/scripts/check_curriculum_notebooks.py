#!/usr/bin/env python3
"""Check generated CuTe DSL curriculum notebooks for structure and syntax."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


DSL_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "day07_async_pipeline/day07.ipynb": {
        "minimum_cells": 15,
        "minimum_markdown": 5_500,
        "markers": {"PipelineAsync", "producer.tail()", "Day 07 TODO(1)"},
    },
    "day08_benchmark_autotune/day08.ipynb": {
        "minimum_cells": 14,
        "minimum_markdown": 5_000,
        "markers": {"autotune_jit", "workspace_generator", "Day 08 TODO(2)"},
    },
    "day09_cuda_graphs/day09.ipynb": {
        "minimum_cells": 13,
        "minimum_markdown": 4_500,
        "markers": {"cute.compile", "CUstream", "Day 09 TODO(1)"},
    },
    "day10_gemm_tour/day10.ipynb": {
        "minimum_cells": 20,
        "minimum_markdown": 8_500,
        "markers": {
            "warp_tiled_matmul_kernel",
            "cute.local_partition",
            "compute_thread_layout",
            "cute.make_rmem_tensor",
            "tiled_mma_sgemm_kernel",
            "cute.make_tiled_copy_tv",
            "thr_mma.partition_A",
        },
    },
    "day11_softmax_reduction/day11.ipynb": {
        "minimum_cells": 12,
        "minimum_markdown": 5_000,
        "markers": {
            "TensorSSA.reduce",
            "shuffle_sync_bfly",
            "triton.testing.do_bench",
            "test_softmax()",
            "compare_cute_and_triton()",
        },
    },
    "day12_softmax_attention/day12.ipynb": {
        "minimum_cells": 13,
        "minimum_markdown": 9_000,
        "markers": {
            "simplified_flash_attention2_kernel",
            "def simplified_flash_attention2(",
            "MmaTF32Op",
            "MmaTF32CompatOp",
            "MmaAtomSM80Type",
            "atom_layout_mnk=(WARPS_PER_CTA, 1, 1)",
            "WARPS_PER_CTA = 4",
            "QUERY_TILE = QUERY_ROWS_PER_WARP * WARPS_PER_CTA",
            "quad_reduce_sum",
            "r_running_max",
            "(4096, 64)",
            "r_output.fill(0.0)",
            "Float32 Q/K/V/output storage",
            "CUTE_TEST_SHAPES",
            "Q/O API shape: [M, D]",
            "triton_flash_attention2_kernel",
            "tl.dot(q, tl.trans(keys))",
            "output.to(tl.float16)",
            "running_max",
            "output_accumulator",
            "test_triton_attention()",
            "triton_benchmark_suite()",
            "test_cute_attention()",
            "benchmark_naive_attention",
            "cute_benchmark_suite()",
            "Nsight Compute: why fused CuTe is slower",
            "profile_summary.csv",
            "542,941",
            "255 registers/thread",
        },
    },
    "day05_matrix_transpose/day05.ipynb": {
        "minimum_cells": 26,
        "minimum_markdown": 9_000,
        "markers": {
            "cute.make_swizzle(5, 0, 5)",
            "sAT = cute.make_tensor",
            "Challenge 5.1: naive shared-memory transpose",
            "return 33, (tile_rows - 1) * 33 + tile_cols",
            "right = (offset >> 5) & 0x1F",
            "input_T = cute.make_tensor",
            "sAT = cute.make_tensor",
            "stride=(1, 32)",
            "Challenge 5.3 shared/swizzled",
            "benchmark_transpose_bandwidth",
        },
    },
    "day06_ncu_profiling/day06.ipynb": {
        "minimum_cells": 18,
        "minimum_markdown": 8_000,
        "markers": {
            "gpu__time_duration.sum",
            "S<5,0,5>",
            "naive_write.csv",
            "Day 06 TODO(1)",
        },
    },
}


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def check(relative_path: str, requirements: dict[str, object]) -> None:
    path = DSL_ROOT / relative_path
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("nbformat") != 4:
        raise ValueError("expected nbformat 4")

    cells = data.get("cells")
    if not isinstance(cells, list):
        raise ValueError("cells must be a list")
    minimum_cells = int(requirements["minimum_cells"])
    if len(cells) < minimum_cells:
        raise ValueError(f"expected at least {minimum_cells} cells, got {len(cells)}")

    markdown_text = "\n".join(
        source(cell)
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
    )
    minimum_markdown = int(requirements["minimum_markdown"])
    if len(markdown_text) < minimum_markdown:
        raise ValueError(
            f"expected at least {minimum_markdown} markdown characters, "
            f"got {len(markdown_text)}"
        )

    all_text = "\n".join(
        source(cell) for cell in cells if isinstance(cell, dict)
    )
    for marker in requirements["markers"]:
        if marker not in all_text:
            raise ValueError(f"missing required marker {marker!r}")

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError(f"code cell {index} contains execution output")
        cell_source = source(cell)
        if any(
            line.lstrip().startswith(("%", "!"))
            for line in cell_source.splitlines()
        ):
            continue
        try:
            ast.parse(cell_source)
        except SyntaxError as error:
            raise SyntaxError(f"invalid code cell {index}: {error}") from error


def main() -> int:
    failures: list[str] = []
    for relative_path, requirements in EXPECTED.items():
        try:
            check(relative_path, requirements)
        except Exception as error:
            failures.append(relative_path)
            print(f"FAIL {relative_path}: {error}", file=sys.stderr)
        else:
            print(f"PASS {relative_path}")

    if failures:
        return 1
    print(f"All {len(EXPECTED)} advanced curriculum notebooks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
