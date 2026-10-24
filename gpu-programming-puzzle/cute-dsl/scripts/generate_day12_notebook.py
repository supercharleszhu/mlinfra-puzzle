#!/usr/bin/env python3
"""Generate the explanation-rich Day 12 attention notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from textwrap import dedent


DSL_ROOT = Path(__file__).resolve().parents[1]
DAY_ROOT = DSL_ROOT / "day12_softmax_attention"


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


def solution_sections() -> dict[str, str]:
    source = (DAY_ROOT / "solution.py").read_text(encoding="utf-8")
    source = source.split('\nif __name__ == "__main__":', 1)[0].strip()
    lines = source.splitlines()
    tree = ast.parse(source)

    def node_source(name: str) -> str:
        for node in tree.body:
            node_name = getattr(node, "name", None)
            assignment_names = {
                target.id
                for target in getattr(node, "targets", [])
                if isinstance(target, ast.Name)
            }
            if node_name == name or name in assignment_names:
                start = node.lineno
                decorators = getattr(node, "decorator_list", [])
                if decorators:
                    start = min(start, *(item.lineno for item in decorators))
                return "\n".join(lines[start - 1 : node.end_lineno])
        raise ValueError(f"missing top-level definition {name!r}")

    cute_constants = "\n".join(
        node_source(name)
        for name in (
            "WARP_SIZE",
            "WARPS_PER_CTA",
            "CTA_THREADS",
            "QUERY_ROWS_PER_WARP",
            "QUERY_TILE",
            "KEY_TILE",
        )
    )
    common = "\n\n".join(
        [
            """
import math

import torch

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.atom import Trait, make_atom
from cutlass.cute.core import _pack_shape
from cutlass.cute.runtime import from_dlpack

cutlass.cuda.initialize_cuda_context()
            """.strip(),
            cute_constants,
            node_source("_validate_inputs"),
            node_source("attention_reference"),
            node_source("TEST_SHAPES"),
            node_source("BENCHMARK_SHAPES"),
            node_source("CUTE_TEST_SHAPES"),
            node_source("CUTE_BENCHMARK_SHAPES"),
            node_source("_make_inputs"),
            node_source("attention_flops"),
        ]
    )
    triton = "\n\n".join(
        [
            node_source("build_triton_attention"),
            "_TRITON_MODULE = None\n_TRITON_KERNEL = None",
            node_source("launch_triton_attention"),
        ]
    )
    cute = "\n\n".join(
        [
            """
import math

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.atom import Trait, make_atom
from cutlass.cute.core import _pack_shape
from cutlass.cute.runtime import from_dlpack
            """.strip(),
            cute_constants,
            node_source("_validate_inputs"),
            node_source("MmaTF32CompatTrait"),
            node_source("MmaTF32CompatOp"),
            node_source("MmaTF32Op"),
            node_source("quad_reduce_sum"),
            node_source("quad_reduce_max"),
            node_source("simplified_flash_attention2_kernel"),
            node_source("simplified_flash_attention2"),
            node_source("prepare_cute_attention"),
        ]
    )
    return {
        "common": common,
        "triton": triton,
        "triton_test": node_source("test_triton_attention"),
        "triton_benchmark": "\n\n".join(
            [
                node_source("benchmark_triton_attention"),
                node_source("triton_benchmark_suite"),
            ]
        ),
        "cute": cute,
        "cute_test": node_source("test_cute_attention"),
        "cute_benchmark": "\n\n".join(
            [
                node_source("benchmark_cute_attention"),
                node_source("benchmark_naive_attention"),
                node_source("cute_benchmark_suite"),
            ]
        ),
    }


def main() -> None:
    guide = (DAY_ROOT / "README.md").read_text(encoding="utf-8")
    before_implementations, implementation_guide = guide.split(
        "## 6. Implementations", 1
    )
    implementation_guide, after_implementations = implementation_guide.split(
        "## 7. Compare the implementations", 1
    )
    triton_guide, cute_guide = implementation_guide.split(
        "### 6.2 CuTe implementation", 1
    )
    sections = solution_sections()
    cells = [
        markdown(before_implementations),
        markdown(
            """
            ## Equation checkpoint: test the invariant on ordinary tensors

            Before writing a kernel, verify the update independently of GPU
            ownership. The code below streams key/value tiles and compares
            `(m, l, U)` with direct softmax. Notice that `U` is normalized only
            after the loop.
            """
        ),
        code(
            """
            import math
            import torch


            def online_attention_reference(q, k, v, key_tile=16):
                scale = 1.0 / math.sqrt(q.shape[-1])
                running_max = torch.tensor(
                    -torch.inf, device=q.device, dtype=torch.float32
                )
                running_sum = torch.tensor(
                    0.0, device=q.device, dtype=torch.float32
                )
                output_accumulator = torch.zeros_like(q)

                for start in range(0, k.shape[0], key_tile):
                    k_tile = k[start : start + key_tile]
                    v_tile = v[start : start + key_tile]
                    scores = (k_tile @ q) * scale
                    new_max = torch.maximum(
                        running_max, scores.max()
                    )
                    previous_scale = torch.exp(
                        running_max - new_max
                    )
                    probabilities = torch.exp(scores - new_max)
                    running_sum = (
                        running_sum * previous_scale
                        + probabilities.sum()
                    )
                    output_accumulator = (
                        output_accumulator * previous_scale
                        + probabilities @ v_tile
                    )
                    running_max = new_max

                return output_accumulator / running_sum


            torch.manual_seed(0)
            q_demo = torch.randn(32, device="cuda")
            k_demo = torch.randn(37, 32, device="cuda")
            v_demo = torch.randn(37, 32, device="cuda")
            direct_demo = torch.softmax(
                k_demo @ q_demo / math.sqrt(32), dim=0
            ) @ v_demo
            online_demo = online_attention_reference(
                q_demo, k_demo, v_demo
            )
            torch.testing.assert_close(
                online_demo, direct_demo, atol=2e-6, rtol=2e-5
            )
            print("online recurrence matches direct attention")
            """
        ),
        markdown(
            """
            ## Notebook setup

            The implementations share only input validation, the direct
            PyTorch oracle, deterministic input generation, and the FLOP
            convention. Their correctness and timing functions remain
            separate so each implementation is evaluated immediately after it
            is introduced.
            """
        ),
        code(sections["common"]),
        markdown(
            "## 6. Implementations\n\n" + triton_guide
        ),
        code(sections["triton"]),
        markdown(
            """
            #### Triton correctness

            Run the irregular and regular sequence-length cases immediately.
            Lengths 17 and 33 exercise both partial K/V tiles and partially
            populated 64-row Q blocks. The reference computes in FP32, then
            converts to FP16 for the output contract.
            """
        ),
        code(sections["triton_test"]),
        code("test_triton_attention()"),
        markdown(
            """
            #### Triton benchmark

            Each FP16 shape is checked against PyTorch again before timing.
            The reported FLOPs count the two matrix multiplications and omit
            softmax operations, matching the usual attention convention.
            """
        ),
        code(sections["triton_benchmark"]),
        code("triton_benchmark_suite()"),
        markdown("### 6.2 CuTe implementation\n\n" + cute_guide),
        code(sections["cute"]),
        markdown(
            """
            #### CuTe correctness

            The CuTe API intentionally removes the batch-head axis: Q/O have
            shape `[M, head_dim]`, while K/V have shape `[N, head_dim]`.
            Different M and N values exercise partial 64-row Q tiles and
            partial 64-key K/V tiles. The `M=512, N=256` case verifies that
            the launch grid covers every query row when M is larger than N;
            D=4 exercises the TF32 `m16n8k4` atom.
            Four warps share each K/V tile and independently own 16 Q rows.
            Storage remains FP32, while `MmaTF32Op` converts operand
            fragments to TF32 for both tensor-core matrix products.
            """
        ),
        code(sections["cute_test"]),
        code("test_cute_attention()"),
        markdown(
            """
            #### CuTe benchmark

            Correctness is rechecked before timing each single-head `[N, D]`
            FP32-storage, TF32-compute shape.
            """
        ),
        code(sections["cute_benchmark"]),
        code("cute_benchmark_suite()"),
        markdown(
            "## 7. Compare the implementations\n\n"
            + after_implementations
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
    destination = DAY_ROOT / "day12.ipynb"
    destination.write_text(
        json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
