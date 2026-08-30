#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 07 - capture and replay a CuTe DSL kernel with CUDA Graphs."""
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
    increment_kernel(output).launch(
        grid=(1, 1, 1), block=(1, 1, 1), stream=stream
    )


def run(operations: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 07 needs a CUDA device.")
    if operations < 1:
        raise ValueError("operations must be positive.")

    output = torch.zeros(1, device="cuda", dtype=torch.float32)
    output_dsl = from_dlpack(output)
    stream = CUstream(current_stream().cuda_stream)

    # Compilation allocates memory and loads a module, so it must happen before
    # capture. The resulting callable accepts runtime arguments only.
    compiled = cute.compile(increment, output_dsl, stream)
    compiled(output_dsl, stream)
    torch.cuda.synchronize()

    output.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_stream = CUstream(current_stream().cuda_stream)
        for _ in range(operations):
            compiled(output_dsl, graph_stream)

    # Discard the work performed while recording; now each replay should add
    # exactly `operations`.
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, torch.tensor([float(operations)], device="cuda")
    )

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, torch.tensor([float(operations * 2)], device="cuda")
    )
    print(f"after two replays={output.item():.0f}")
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations", type=int, default=8)
    args = parser.parse_args()
    run(args.operations)
