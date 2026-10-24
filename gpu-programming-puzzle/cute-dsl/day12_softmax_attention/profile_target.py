#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic Day 12 target for Nsight Compute."""

from __future__ import annotations

import argparse

import cutlass

from solution import (
    _make_inputs,
    attention_reference,
    prepare_cute_attention,
)


def run(variant: str, sequence_length: int, head_dim: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 12 profiling needs a CUDA device")

    cutlass.cuda.initialize_cuda_context()
    torch.set_float32_matmul_precision("high")
    q, k, v = _make_inputs(
        (sequence_length, head_dim), seed=20261019
    )

    if variant == "cute":
        output, compiled, arguments = prepare_cute_attention(q, k, v)
        launch = lambda: compiled(*arguments)
    else:
        output = attention_reference(q, k, v)
        launch = lambda: attention_reference(q, k, v)

    launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output if variant == "cute" else launch(),
        attention_reference(q, k, v),
        atol=3e-2,
        rtol=3e-2,
    )
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    profiled_output = launch()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    result = output if variant == "cute" else profiled_output
    torch.testing.assert_close(
        result,
        attention_reference(q, k, v),
        atol=3e-2,
        rtol=3e-2,
    )
    print(
        f"PROFILE_TARGET variant={variant} "
        f"shape=({sequence_length},{head_dim}) correctness=OK"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("naive", "cute"), required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()
    run(args.variant, args.sequence_length, args.head_dim)
