#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic Day 05 transpose target for Nsight Compute."""

import argparse
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


DSL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DSL_ROOT))

from day05_matrix_transpose.solution import (  # noqa: E402
    naive_transpose_coalesced_read,
    naive_transpose_coalesced_write,
    transpose_padded,
    transpose_swizzled,
    transpose_unpadded,
)


VARIANTS = {
    "naive-read": naive_transpose_coalesced_read,
    "naive-write": naive_transpose_coalesced_write,
    "shared-unpadded": transpose_unpadded,
    "shared-padded": transpose_padded,
    "shared-swizzled": transpose_swizzled,
}


def run(m: int, n: int, variant: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 06 needs a CUDA device.")

    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(0)
    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    b = torch.empty(n, m, device="cuda", dtype=torch.float32)
    a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
    b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()

    transpose_fn = VARIANTS[variant]
    compiled = cute.compile(transpose_fn, a_, b_)
    compiled(a_, b_)
    torch.cuda.synchronize()
    torch.testing.assert_close(b, a.T)
    print(f"PROFILE_TARGET variant={variant} shape=({m},{n}) correctness=OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    args = parser.parse_args()
    run(args.M, args.N, args.variant)
