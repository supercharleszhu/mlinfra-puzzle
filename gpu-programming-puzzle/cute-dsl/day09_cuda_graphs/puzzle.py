#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 09 - fill the CuTe DSL CUDA Graph capture TODOs."""
import argparse

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings.driver import CUstream
from torch.cuda import current_stream


@cute.kernel
def increment_kernel(output: cute.Tensor):
    output[0] = output[0] + 1.0


@cute.jit
def increment(output: cute.Tensor, stream: CUstream):
    # TODO(1): launch increment_kernel into the supplied stream.
    raise NotImplementedError("Day 09 TODO(1): launch with stream=stream")


def run(operations: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 09 needs a CUDA device.")
    output = torch.zeros(1, device="cuda", dtype=torch.float32)
    output_dsl = from_dlpack(output)

    # TODO(2): wrap current_stream().cuda_stream in CUstream.
    # TODO(3): precompile increment with `output_dsl` and that stream.
    raise NotImplementedError("Day 09 TODO(2-3): stream and precompile")

    output.zero_()
    graph = torch.cuda.CUDAGraph()

    # TODO(4): use `with torch.cuda.graph(graph):` and enqueue `operations`
    # calls to the compiled function using the stream active during capture.

    output.zero_()
    # TODO(5): replay and assert output == operations, then replay again and
    # assert output == operations * 2. Synchronize before each assertion.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations", type=int, default=8)
    args = parser.parse_args()
    run(args.operations)
