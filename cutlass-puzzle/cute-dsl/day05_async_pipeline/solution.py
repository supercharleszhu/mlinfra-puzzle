#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 05 - CuTe DSL asynchronous producer/consumer pipeline (solution)."""
import argparse

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def pipeline_kernel(
    shared_storage: cutlass.Constexpr,
    result: cute.Tensor,
    observed_stages: cute.Tensor,
):
    stages = cute.size(observed_stages)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage, 64)
    stage_buffer = storage.stage_buffer.get_tensor(observed_stages.layout)
    stage_buffer.fill(0)
    cute.arch.sync_threads()

    producer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread, 32
    )
    consumer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread, 32
    )
    pipeline = cutlass.pipeline.PipelineAsync.create(
        num_stages=stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        barrier_storage=storage.barriers.data_ptr(),
    )
    producer, consumer = pipeline.make_participants()

    if warp_idx == 0:
        for i in cutlass.range(cute.size(result)):
            handle = producer.acquire_and_advance()
            stage_buffer[handle.index] = i * 1.0
            handle.commit()
        producer.tail()

    if warp_idx == 1:
        for i in cutlass.range(cute.size(result)):
            handle = consumer.wait_and_advance()
            result[i] = stage_buffer[handle.index]
            handle.release()

    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        observed_stages.store(stage_buffer.load())


@cute.jit
def run_pipeline(result: cute.Tensor, observed_stages: cute.Tensor):
    stages = cute.size(observed_stages)

    @cute.struct
    class SharedStorage:
        barriers: cute.struct.MemRange[cutlass.Int64, stages * 2]
        stage_buffer: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, stages], 1024
        ]

    pipeline_kernel(SharedStorage, result, observed_stages).launch(
        grid=(1, 1, 1), block=(64, 1, 1)
    )


def run(length: int, stages: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 05 needs a CUDA device.")
    if length < 1 or stages < 1:
        raise ValueError("length and stages must be positive.")

    cutlass.cuda.initialize_cuda_context()
    result = torch.full((length,), -1, device="cuda", dtype=torch.float32)
    observed_stages = torch.zeros((stages,), device="cuda", dtype=torch.float32)

    run_pipeline(from_dlpack(result), from_dlpack(observed_stages))
    torch.cuda.synchronize()

    expected = torch.arange(length, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result, expected)
    print(f"result={result.cpu().tolist()}")
    print(f"final stage buffer={observed_stages.cpu().tolist()}")
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--stages", type=int, default=3)
    args = parser.parse_args()
    run(args.length, args.stages)
