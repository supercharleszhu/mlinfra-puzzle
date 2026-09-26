#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 07 - fill the CuTe DSL asynchronous-pipeline TODOs."""
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

    # TODO(1): describe one producer warp and one consumer warp.
    # producer_group = cutlass.pipeline.CooperativeGroup(
    #     cutlass.pipeline.Agent.Thread, 32
    # )
    # consumer_group = ...
    raise NotImplementedError("Day 07 TODO(1): create cooperative groups")

    # TODO(2): create PipelineAsync with `stages` slots and the shared barriers.
    # pipeline = cutlass.pipeline.PipelineAsync.create(
    #     num_stages=stages,
    #     producer_group=producer_group,
    #     consumer_group=consumer_group,
    #     barrier_storage=storage.barriers.data_ptr(),
    # )
    # producer, consumer = pipeline.make_participants()

    if warp_idx == 0:
        for i in cutlass.range(cute.size(result)):
            # TODO(3): acquire a free slot, write `i`, then commit the slot.
            pass
        # TODO(4): call producer.tail() before the producer warp retires.

    if warp_idx == 1:
        for i in cutlass.range(cute.size(result)):
            # TODO(5): wait for a full slot, consume it, then release it.
            pass

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
        raise RuntimeError("Day 07 needs a CUDA device.")
    cutlass.cuda.initialize_cuda_context()
    result = torch.full((length,), -1, device="cuda", dtype=torch.float32)
    observed_stages = torch.zeros((stages,), device="cuda", dtype=torch.float32)
    run_pipeline(from_dlpack(result), from_dlpack(observed_stages))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        result, torch.arange(length, device="cuda", dtype=torch.float32)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--stages", type=int, default=3)
    args = parser.parse_args()
    run(args.length, args.stages)
