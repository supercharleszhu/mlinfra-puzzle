#!/usr/bin/env python3
"""Generate explanation-rich challenge notebooks for CuTe DSL Days 07-10."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


DSL_ROOT = Path(__file__).resolve().parents[1]


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


def puzzle_body(day: str, start_marker: str) -> str:
    source = (DSL_ROOT / day / "puzzle.py").read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index("\n\nif __name__")
    return source[start:end].strip()


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
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


def day07_cells() -> list[dict[str, object]]:
    scaffold = puzzle_body("day07_async_pipeline", "@cute.kernel")
    return [
        markdown(
            """
            # Day 07 - Warp-specialized asynchronous pipelines

            This lesson develops the producer/consumer progression from official
            [Notebook 09](../notebooks/09_async_pipeline.ipynb), then turns it
            into a five-step challenge. Two warps exchange values through a
            shared-memory ring buffer:

            ```text
            producer warp -> shared-memory stages -> consumer warp
            ```

            The goal is not merely to call `PipelineAsync`. You should be able
            to explain who owns each stage, which event changes its state, why a
            stage cannot be overwritten early, and why the producer must drain
            the protocol before retiring.
            """
        ),
        code(
            """
            import torch

            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack

            cutlass.cuda.initialize_cuda_context()
            """
        ),
        markdown(
            """
            ## 1. Why two CTA barriers are correct but restrictive

            A synchronous one-slot exchange can use this sequence:

            ```text
            producer writes slot
                 |
                 +-- CTA barrier: value is now visible
                 |
            consumer reads slot
                 |
                 +-- CTA barrier: slot may be overwritten
            ```

            Both barriers involve the entire CTA. Even when warp 0 only
            produces and warp 1 only consumes, both warps repeatedly enter
            lockstep. A pipeline replaces global rendezvous with stage-local
            ownership transitions. That permits the producer to prepare a
            future stage while the consumer works on an older one.

            `PipelineAsync` coordinates readiness; the payload still lives in
            shared memory. The pipeline does not move data for you.
            """
        ),
        markdown(
            """
            ## 2. The four transitions of one stage

            Every reusable stage follows the same state machine:

            | Actor | Operation | State transition | Meaning |
            | --- | --- | --- | --- |
            | Producer | `acquire_and_advance()` | empty -> producer-owned | Wait until this slot is reusable |
            | Producer | `handle.commit()` | producer-owned -> full | Publish the completed write |
            | Consumer | `wait_and_advance()` | full -> consumer-owned | Wait until the value is ready |
            | Consumer | `handle.release()` | consumer-owned -> empty | Permit the next overwrite |

            Acquire and wait are ordering points. Commit and release are
            notifications. Omitting any one transition can produce stale data,
            an overwrite race, or a deadlock.
            """
        ),
        code(
            """
            # Trace the indices used by a three-stage circular buffer.
            stages = 3
            items = 8
            producer_slots = [item % stages for item in range(items)]
            consumer_slots = [item % stages for item in range(items)]

            assert producer_slots == consumer_slots
            for item, slot in enumerate(producer_slots):
                print(f"item {item}: acquire/commit/wait/release stage {slot}")
            """
        ),
        markdown(
            """
            ## 3. Why multiple stages enable overlap

            With one stage, the producer cannot begin item `i+1` until the
            consumer releases item `i`. With three stages, it can publish items
            0, 1, and 2 before wrapping back to stage 0. On wraparound,
            `acquire_and_advance()` supplies backpressure: it waits until the
            consumer has released that stage.

            More stages are not automatically faster. Each stage consumes
            shared memory and barrier state. Choose enough stages to cover the
            latency gap between producer and consumer, but not so many that
            shared-memory use reduces occupancy.

            In this small exercise one Float32 is stored per stage. In GEMM,
            each stage usually contains complete A and B tiles, so stage count
            has a much larger resource cost.
            """
        ),
        code(
            """
            def stage_storage_bytes(stages, elements_per_stage, element_bytes=2):
                payload = stages * elements_per_stage * element_bytes
                barrier_words = stages * 2
                barrier_bytes = barrier_words * 8
                return payload, barrier_bytes


            for stage_count in (1, 2, 3, 4):
                payload, barriers = stage_storage_bytes(
                    stage_count, elements_per_stage=4096
                )
                print(
                    f"{stage_count} stages: payload={payload / 1024:.1f} KiB, "
                    f"barriers={barriers} B"
                )
            """
        ),
        markdown(
            """
            ## 4. CuTe objects that implement the protocol

            Two cooperative groups describe the participant counts:

            ```python
            producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, 32
            )
            consumer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, 32
            )
            ```

            They describe one 32-thread producer warp and one 32-thread
            consumer warp. `PipelineAsync.create` combines those groups with:

            - `num_stages`, the circular-buffer depth;
            - shared barrier storage, with two 64-bit words per stage;
            - participant state returned by `make_participants()`.

            The stage index comes from the acquired or waited handle. Always
            use `handle.index` to address the matching payload slot; do not
            maintain an unrelated manual modulo counter.
            """
        ),
        markdown(
            """
            ## 5. Why `producer.tail()` matters

            Kernel completion does not replace protocol completion. The
            producer can finish issuing commits while the consumer still owns
            stages. `producer.tail()` waits for the expected final consumer
            arrivals before the producer warp retires and the CTA's shared
            storage disappears.

            A useful rule:

            ```text
            commit publishes one item;
            tail drains all outstanding producer obligations.
            ```

            Tail belongs after the producer loop, not after each item. Calling
            it per item removes the overlap the pipeline was intended to
            provide.
            """
        ),
        markdown(
            """
            ## 6. Challenge: build the staged exchange

            Fill the TODOs in order:

            1. Describe one producer and one consumer cooperative group.
            2. Create a `PipelineAsync` over the shared barriers.
            3. Acquire, write, and commit each producer item.
            4. Drain the producer with `tail()`.
            5. Wait, read, and release each consumer item.

            The result must be `0..length-1` even when `length` exceeds the
            stage count many times. That wraparound is the important case.
            """
        ),
        code(scaffold),
        markdown(
            """
            ## 7. Correctness experiment

            Run after replacing all five TODOs. Try a stage count that does not
            divide the item count, ensuring the ring wraps at different points.
            """
        ),
        code(
            """
            run(length=17, stages=3)
            print("Pipeline preserved all 17 values across stage wraparound.")
            """
        ),
        markdown(
            """
            ## 8. Debugging a pipeline

            | Symptom | Likely protocol error |
            | --- | --- |
            | Kernel hangs | Missing `commit`, `release`, or mismatched participant counts |
            | Repeated/stale values | Consumer read before the matching wait |
            | Values disappear after wraparound | Producer did not acquire before overwrite |
            | Only stage 0 changes | Payload indexed manually instead of with `handle.index` |
            | Final iterations hang | Producer omitted `tail()` or loop trip counts differ |

            Pipeline bugs often appear as hangs rather than Python exceptions.
            First compare producer and consumer trip counts. Then verify each
            acquired handle is committed exactly once and each waited handle is
            released exactly once.
            """
        ),
        markdown(
            """
            ## 9. Connection to a pipelined GEMM

            Replace the scalar payload with A/B tiles and the roles become:

            ```text
            Producer: acquire -> global-to-shared copy -> commit
            Consumer: wait -> shared-to-register copy + MMA -> release
            ```

            Day 10 initially uses one synchronous shared-memory stage so the
            dataflow remains visible. The optimization path is to allocate
            multiple A/B stages, prefetch future K tiles, and use this same
            state machine to overlap movement with MMA.

            **Checkpoint:** explain why `release()` must occur after the final
            read or MMA that consumes the stage, not immediately after `wait()`.
            """
        ),
    ]


def day08_cells() -> list[dict[str, object]]:
    scaffold = puzzle_body("day08_benchmark_autotune", "@cute.kernel")
    return [
        markdown(
            """
            # Day 08 - Benchmarking and autotuning CuTe kernels

            This lesson follows official
            [Notebook 10](../notebooks/10_benchmark_autotune.ipynb) and reuses
            the predicated elementwise mapping from Day 04. The new problem is
            experimental method:

            ```text
            prove correctness -> define candidates -> tune fairly
                              -> compile the winner -> benchmark steadily
            ```

            You will compare 64-bit and 128-bit copy widths, use both automatic
            and explicit tuning APIs, and convert elapsed microseconds into
            effective memory bandwidth.
            """
        ),
        code(
            """
            import torch

            import cutlass
            import cutlass.cute as cute
            import cutlass.testing as testing
            from cutlass.cute.testing import autotune_jit

            cutlass.cuda.initialize_cuda_context()
            """
        ),
        markdown(
            """
            ## 1. Keep four activities separate

            | Activity | Question | Should be inside timing? |
            | --- | --- | --- |
            | Correctness | Does every candidate compute `C = A + B`? | No |
            | JIT compilation | Can CuTe specialize this Python program? | No |
            | Tuning | Which valid configuration is fastest here? | Measures candidates |
            | Final benchmark | How fast is the selected operation? | Kernel invocation only |

            Timing a first call commonly measures Python tracing, MLIR
            lowering, module loading, allocation, and GPU work together. That
            number does not describe kernel performance. Compile first, warm
            up, synchronize through the timing utility, and then measure.
            """
        ),
        markdown(
            """
            ## 2. A compile-time parameter changes the generated kernel

            `copy_bits` is a `cutlass.Constexpr`, so each candidate becomes a
            separate specialization:

            ```python
            vector_size = copy_bits // tensor_a.element_type.width
            ```

            For Float32, 64-bit and 128-bit copies correspond to 2 and 4
            elements. That value changes the value layout and therefore the CTA
            tile width. Tuning is choosing between generated programs, not
            changing a runtime branch.
            """
        ),
        code(
            """
            dtype_width = 32
            thread_shape = (4, 32)

            for copy_bits in (64, 128):
                vector_size = copy_bits // dtype_width
                value_shape = (4, vector_size)
                tile_shape = (
                    thread_shape[0] * value_shape[0],
                    thread_shape[1] * value_shape[1],
                )
                print(
                    f"{copy_bits:3} bits -> {vector_size} Float32 values/copy "
                    f"-> CTA tile {tile_shape}"
                )
            """
        ),
        markdown(
            """
            ## 3. Decorator-based autotuning

            `@autotune_jit` wraps a `@cute.jit` function and searches a
            parameter dictionary:

            ```python
            @autotune_jit(
                params_dict={"copy_bits": [64, 128]},
                update_on_change=["m", "n"],
                warmup_iterations=5,
                iterations=20,
            )
            @cute.jit
            def add(..., copy_bits: cutlass.Constexpr = 128):
                ...
            ```

            `update_on_change` defines the cache key dimensions that require a
            new search. A best choice for one `(m, n)` may not remain best for
            another shape. Include every runtime property that can change the
            ranking; avoid including unrelated values that cause needless
            retuning.

            Set `CUTE_DSL_LOG_AUTOTUNE=1` before execution when you need to see
            candidate times and cache hits.
            """
        ),
        markdown(
            """
            ## 4. Explicit `testing.tune`

            The explicit API makes the winning dictionary visible. Its callback
            has an important two-level contract:

            ```python
            def compile_config(a, b, c, rows, cols, copy_bits):
                compiled = cute.compile(
                    add, a, b, c, rows, cols, copy_bits=copy_bits
                )
                return lambda: compiled(a, b, c, rows, cols)
            ```

            The outer callback receives one candidate and compiles it. The
            returned zero-argument callable performs only the runtime launch.
            Returning the result of the launch, instead of a callable, would
            execute too early and leave the tuner nothing repeatable to time.
            """
        ),
        markdown(
            """
            ## 5. Reliable benchmark inputs

            `testing.benchmark` returns average time in **microseconds**. Its
            controls address different sources of noise:

            - `warmup_iterations`: populate code/data paths and stabilize clocks;
            - `iterations`: amortize timer noise;
            - `stream`: use the same stream as the measured launch;
            - `workspace_generator`: construct independent argument sets;
            - `workspace_count`: rotate those sets to reduce persistent L2 hits;
            - `use_cuda_graphs`: reduce host launch overhead for very short work.

            Workspace rotation changes what is being measured. One workspace
            can describe warm-cache reuse; several larger workspaces better
            approximate streaming traffic. State which experiment you chose.
            """
        ),
        code(
            """
            def effective_bandwidth_gbps(m, n, element_bytes, average_us):
                # Two input reads plus one output write.
                bytes_moved = 3 * m * n * element_bytes
                return bytes_moved / (average_us * 1_000)


            example = effective_bandwidth_gbps(
                m=4096, n=4096, element_bytes=4, average_us=200
            )
            print(f"Example effective bandwidth: {example:.2f} GB/s")
            """
        ),
        markdown(
            """
            ## 6. A fair tuning checklist

            Before comparing candidates, verify:

            1. Every candidate passes correctness, including edge shapes.
            2. Candidate compilation is outside the measured callback.
            3. All candidates use identical inputs, stream, warmup, and trials.
            4. The search space contains only legal alignments and vector widths.
            5. The winner is recompiled or retrieved and checked again.
            6. The final benchmark reports units and the byte/FLOP convention.

            A tuner can faithfully select the fastest incorrect kernel. Tuning
            never substitutes for validation.
            """
        ),
        markdown(
            """
            ## 7. Challenge: complete both tuning paths

            Fill six TODOs:

            1. Add the `@autotune_jit` search over 64 and 128 bits.
            2. Convert copy width into elements per vector.
            3. Launch one CTA for each outer tile.
            4. Compile one explicit-tuning candidate.
            5. Return the winner from `testing.tune`.
            6. Benchmark and report microseconds.

            The data mapping is already complete so you can focus on experiment
            construction.
            """
        ),
        code(scaffold),
        markdown(
            """
            ## 8. Run correctness, tuning, and measurement

            Run only after all TODOs are filled. Repeat with a larger,
            non-square shape and observe whether the winning copy width changes.
            """
        ),
        code(
            """
            run(512, 512)
            run(1024, 1536)
            """
        ),
        markdown(
            """
            ## 9. Interpret the result rather than only printing it

            Record:

            | Shape | Winning copy width | Average us | Effective GB/s |
            | --- | ---: | ---: | ---: |
            | `512 x 512` | fill in | fill in | fill in |
            | `1024 x 1536` | fill in | fill in | fill in |

            Then answer:

            - Did the wider copy always win?
            - Is the small shape dominated by launch overhead?
            - Does rotating workspaces lower apparent bandwidth?
            - Would `use_cuda_graphs=True` change the kernel or only submission?

            Day 09 isolates the last question by capturing repeated launches.
            """
        ),
    ]


def day09_cells() -> list[dict[str, object]]:
    scaffold = puzzle_body("day09_cuda_graphs", "@cute.kernel")
    return [
        markdown(
            """
            # Day 09 - CUDA Graph capture and replay

            This lesson expands official
            [Notebook 11](../notebooks/11_cuda_graphs.ipynb) into a stateful
            correctness exercise. A tiny CuTe kernel increments one scalar.
            Capturing `operations` launches creates one reusable graph; each
            replay must add exactly `operations` to the scalar.

            CUDA Graphs optimize repeated submission. They do not make the
            increment instruction itself faster.
            """
        ),
        code(
            """
            import torch

            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            from cuda.bindings.driver import CUstream
            from torch.cuda import current_stream
            """
        ),
        markdown(
            """
            ## 1. Direct launch versus graph replay

            Without a graph, Python submits every launch:

            ```text
            Python -> launch 0 -> launch 1 -> ... -> launch N-1
            ```

            With a graph, Python records the sequence once and later submits the
            complete dependency graph:

            ```text
            capture: Python -> [launch 0, launch 1, ..., launch N-1]
            replay:  Python -> graph launch
            ```

            This matters most for short kernels, where CPU dispatch latency is a
            meaningful fraction of end-to-end time. Long compute-bound kernels
            may show little difference.
            """
        ),
        markdown(
            """
            ## 2. Compile before stream capture

            `cute.compile` may trace Python, generate MLIR/PTX, load a module,
            and allocate runtime state. Those activities are not valid graph
            nodes and may not be capture-safe. The required order is:

            ```text
            construct representative arguments
                -> cute.compile(...)
                -> optional warmup
                -> begin graph capture
                -> call only the compiled runtime callable
                -> end capture
                -> replay
            ```

            Calling the original `@cute.jit` function for the first time inside
            capture can accidentally trigger compilation. Keep a separate
            variable such as `compiled` and invoke that variable in the capture
            region.
            """
        ),
        markdown(
            """
            ## 3. Bridge PyTorch streams to the CUDA driver

            PyTorch owns the capture stream, while CuTe's launch accepts a
            CUDA-driver stream handle:

            ```python
            stream = CUstream(current_stream().cuda_stream)
            ```

            Create the stream argument used for compilation before capture.
            Inside `with torch.cuda.graph(graph):`, query `current_stream()`
            again and wrap that active capture stream. Passing a different
            stream means the launch may occur outside the captured graph or the
            runtime may reject the mismatch.
            """
        ),
        code(
            """
            operations = 8
            replays = 2
            expected_after_each_replay = [
                operations * replay for replay in range(1, replays + 1)
            ]
            assert expected_after_each_replay == [8, 16]
            print(expected_after_each_replay)
            """
        ),
        markdown(
            """
            ## 4. Replay uses stable topology and addresses

            A graph records concrete operations, dependencies, and argument
            addresses. Replay does not rerun the Python loop and does not
            automatically substitute newly allocated tensors.

            Safe baseline:

            - keep captured tensors alive;
            - update values in those tensors in place;
            - keep shapes and launch topology fixed;
            - synchronize before reading host-visible results.

            Advanced graph-update APIs can change some node parameters, but
            this lesson intentionally uses fixed storage.
            """
        ),
        markdown(
            """
            ## 5. Challenge: capture repeated CuTe launches

            Fill five TODOs:

            1. Launch the kernel on the supplied `CUstream`.
            2. Wrap PyTorch's current stream.
            3. Compile before capture.
            4. Capture `operations` compiled calls on the active graph stream.
            5. Replay twice and validate cumulative state.

            The first replay should produce `operations`; the second should
            produce `2 * operations`.
            """
        ),
        code(scaffold),
        markdown(
            """
            ## 6. Correctness experiment

            Run after filling all TODOs. Try one and many operations to verify
            that capture topology determines the increment count.
            """
        ),
        code(
            """
            run(operations=1)
            run(operations=8)
            """
        ),
        markdown(
            """
            ## 7. Measure submission overhead

            After correctness passes, compare a direct Python loop with one
            graph replay. CUDA events measure GPU-stream elapsed time; wall
            time includes more Python overhead. For very small kernels, report
            which timer you used.

            ```python
            # Conceptual experiment:
            start.record()
            for _ in range(operations):
                compiled(output_dsl, stream)
            end.record()

            start.record()
            graph.replay()
            end.record()
            ```

            Warm both paths before measuring. A graph can reduce gaps between
            kernels even though each kernel's device instructions are unchanged.
            """
        ),
        markdown(
            """
            ## 8. Common capture failures

            | Symptom | Likely cause |
            | --- | --- |
            | Capture error during first call | JIT compilation happened inside capture |
            | Graph replays but output stays zero | Launch used a non-captured stream |
            | First replay is too large | Recording work was not cleared before replay |
            | Later tensor is unchanged | A new allocation replaced the captured address |
            | Host reads stale output | Missing synchronization before assertion |

            The capture itself executes work while recording. This lesson calls
            `output.zero_()` after capture so replay accounting starts from zero.
            """
        ),
        markdown(
            """
            ## 9. Where graphs fit in real workloads

            Graphs are useful when a stable sequence repeats: inference steps,
            optimizer updates, communication/computation schedules, or a fixed
            stack of small kernels. They compose with Day 08 benchmarking and
            Day 10 GEMM, but solve a different bottleneck:

            - kernel optimization reduces device execution time;
            - CUDA Graphs reduce repeated host submission overhead.

            **Checkpoint:** explain why changing tensor contents can be safe
            while replacing the tensor allocation is not.
            """
        ),
    ]


def day10_cells() -> list[dict[str, object]]:
    scaffold = puzzle_body("day10_gemm_tour", "class SimpleSGemm")
    return [
        markdown(
            """
            # Day 10 - Tour from layouts to a complete GEMM

            Official [Notebook 12](../notebooks/12_tour_to_sol_gemm.ipynb)
            builds a Blackwell speed-of-light GEMM with TMA, UMMA, tensor
            memory, software pipelining, and a vectorized epilogue. This
            notebook teaches the same algorithmic roles with an
            H100-compatible, single-stage FP32 SIMT GEMM.

            The objective is to connect the earlier lessons:

            ```text
            layouts -> CTA tiles -> cooperative copy -> shared memory
                    -> per-thread MMA fragments -> K reduction -> epilogue
            ```

            The local kernel favors visibility over peak throughput. The final
            sections map each stage to the more specialized Blackwell version.
            """
        ),
        code(
            """
            from typing import Tuple

            import torch

            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack

            cutlass.cuda.initialize_cuda_context()
            """
        ),
        markdown(
            """
            ## 1. GEMM equation and the tensor convention

            This lesson computes:

            \[
            C_{m,n} = \sum_{k=0}^{K-1} A_{m,k} B_{n,k}.
            \]

            Therefore A has logical shape `(M, K)`, B has logical shape
            `(N, K)`, and C has shape `(M, N)`. B is represented as `(N, K)`
            so the expression is `A @ B.T`.

            Each output element performs K multiplies and K accumulations,
            conventionally counted as `2*K` floating-point operations. Total
            work is:

            ```text
            FLOPs = 2 * M * N * K
            ```
            """
        ),
        code(
            """
            def gemm_work(m, n, k, element_bytes=4):
                flops = 2 * m * n * k
                minimum_bytes = (m * k + n * k + m * n) * element_bytes
                return flops, minimum_bytes, flops / minimum_bytes


            for dims in ((256, 256, 64), (1024, 1024, 256), (8192, 8192, 8192)):
                flops, bytes_moved, intensity = gemm_work(*dims)
                print(
                    f"{dims}: {flops / 1e9:.2f} GFLOP, "
                    f"minimum traffic={bytes_moved / 1e6:.2f} MB, "
                    f"intensity={intensity:.1f} FLOP/byte"
                )
            """
        ),
        markdown(
            """
            ## 2. The blocked-GEMM hierarchy

            ![Blocked GEMM](../notebooks/images/blocked_gemm.svg)

            One CTA owns a `(BLK_M, BLK_N)` output tile and traverses K in
            `BLK_K` chunks:

            ```text
            grid tile (block_m, block_n)
                -> A tile: BLK_M x BLK_K
                -> B tile: BLK_N x BLK_K
                -> C tile: BLK_M x BLK_N
                -> repeat across K tiles
            ```

            This lesson uses `(64, 64, 8)`. M and N must be multiples of 64 and
            K must be a multiple of 8 because the challenge intentionally omits
            edge predication. Day 04 showed how an identity coordinate tensor
            would be used to remove that restriction.
            """
        ),
        markdown(
            """
            ## 3. Why the host tensors look transposed

            PyTorch allocates row-major tensors, but this kernel wants stride 1
            in the M mode for A/C and the N mode for B. It creates row-major
            `(K, M)` storage and then permutes the view:

            ```python
            a = torch.empty(K, M).permute(1, 0)  # logical (M,K), stride (1,M)
            b = torch.empty(K, N).permute(1, 0)  # logical (N,K), stride (1,N)
            c = torch.empty(N, M).permute(1, 0)  # logical (M,N), stride (1,M)
            ```

            `permute` changes the view, not the stored values. Contiguous major
            modes make each thread's four-Float32, 128-bit copy contiguous.
            """
        ),
        code(
            """
            M, N, K = 256, 256, 64
            a_host = torch.empty(K, M).permute(1, 0)
            b_host = torch.empty(K, N).permute(1, 0)
            c_host = torch.empty(N, M).permute(1, 0)

            print("A shape/stride:", a_host.shape, a_host.stride())
            print("B shape/stride:", b_host.shape, b_host.stride())
            print("C shape/stride:", c_host.shape, c_host.stride())
            assert a_host.stride()[0] == 1
            assert b_host.stride()[0] == 1
            assert c_host.stride()[0] == 1
            """
        ),
        markdown(
            """
            ## 4. `local_tile` selects one CTA's views

            A conceptual `(M,N,K)` CTA tiler must be projected onto tensors that
            do not contain all three modes:

            | Tensor | Logical modes | Projection | Result |
            | --- | --- | --- | --- |
            | A | `(M,K)` | `(1,None,1)` | `(BLK_M,BLK_K,k_tiles)` |
            | B | `(N,K)` | `(None,1,1)` | `(BLK_N,BLK_K,k_tiles)` |
            | C | `(M,N)` | `(1,1,None)` | `(BLK_M,BLK_N)` |

            With `coord=(bidx,bidy,None)`, the CTA fixes its M/N grid location
            while preserving the complete K-tile mode. The trailing K-tile
            mode is the mainloop iteration space.
            """
        ),
        code(
            """
            def tile_counts(m, n, k, tile=(64, 64, 8)):
                return tuple(dim // block for dim, block in zip((m, n, k), tile))


            counts = tile_counts(256, 256, 64)
            assert counts == (4, 4, 8)
            print(
                f"grid={counts[0]} x {counts[1]} CTAs; "
                f"each CTA performs {counts[2]} K-tile iterations"
            )
            """
        ),
        markdown(
            """
            ## 5. Global-to-shared movement

            A and B tiles are cooperatively copied into shared memory. A
            `CopyAtom` describes one hardware copy operation; `TiledCopy`
            distributes those operations across `(thread,value)` coordinates.

            This kernel uses:

            ```python
            cute.nvgpu.cpasync.CopyG2SOp()
            num_bits_per_copy = 4 * 32  # four Float32 values = 128 bits
            ```

            `thr_copy.partition_S(gA)` and `partition_D(sA)` create aligned
            per-thread source and destination views. The views describe
            ownership; `cute.copy` performs the transfer.

            After issuing copies:

            1. `cp_async_commit_group()` closes the copy group.
            2. `cp_async_wait_group(0)` waits until no group remains pending.
            3. `barrier()` makes the shared tile safe for all consumer threads.
            """
        ),
        markdown(
            """
            ## 6. Shared-to-register movement and MMA ownership

            `TiledMma` describes how MMA work is distributed. Each thread first
            obtains a slice:

            ```python
            thr_mma = tiled_mma.get_slice(tidx)
            ```

            Then:

            - `partition_A(sA)` maps the shared A tile into that thread's MMA A view;
            - `partition_B(sB)` does the same for B;
            - `partition_C(gC)` maps accumulator ownership to output addresses;
            - `make_fragment_A/B/C` allocates matching register fragments.

            Partitioned tensors are mappings, not copies. Data enters the A/B
            register fragments only when `autovec_copy` executes. The C
            fragment must be initialized to zero before the K reduction.
            """
        ),
        markdown(
            """
            ## 7. The single-stage K mainloop

            For every K tile:

            ```text
            global A/B --cp.async--> shared A/B
                    wait + CTA barrier
            shared A/B --autovec_copy--> register A/B
            register A/B --cute.gemm--> register accumulator
                    CTA barrier before shared-memory overwrite
            ```

            This ordering is intentionally conservative. The wait prevents
            compute from reading incomplete data; the final barrier prevents
            the next iteration from overwriting shared memory while another
            thread still consumes it.

            The inner `k_block` loop is compile-time unrolled because fragment
            shapes are static. The outer `k_tile` loop depends on the problem's
            dynamic K extent.
            """
        ),
        code(
            """
            BLK_M, BLK_N, BLK_K = 64, 64, 8
            outputs_per_cta = BLK_M * BLK_N
            fmas_per_k_tile = BLK_M * BLK_N * BLK_K
            print(f"outputs per CTA: {outputs_per_cta}")
            print(f"FMAs per K tile: {fmas_per_k_tile}")
            print(f"FLOPs per K tile: {2 * fmas_per_k_tile}")
            """
        ),
        markdown(
            """
            ## 8. Epilogue: accumulator to global C

            After all K tiles, `tCrC` contains the complete per-thread
            accumulator. The simple epilogue copies it directly to the matching
            `tCgC` global view.

            Production epilogues often add:

            - conversion from FP32 accumulation to FP16/BF16 output;
            - scaling, bias, activation, or residual fusion;
            - vectorized stores with explicit alignment and divisibility;
            - accumulator subtiling to reduce register pressure.

            Official Notebook 12 demonstrates why vector-store eligibility can
            materially affect throughput. CuTe needs correct layout,
            alignment, and divisibility information to select wide stores.
            """
        ),
        markdown(
            """
            ## 9. H100 lesson versus Blackwell SOL GEMM

            | Algorithmic role | This notebook | Official Notebook 12 |
            | --- | --- | --- |
            | Global-to-shared load | Cooperative `cp.async` | TMA |
            | Matrix operation | FP32 `MmaUniversalOp` | tcgen05/UMMA |
            | Accumulator storage | Registers | Tensor memory |
            | A/B synchronization | CTA barriers | TMA-UMMA pipelines |
            | Staging | One stage | Multiple prefetched stages |
            | Epilogue | Direct register-to-global copy | Subtiled/vectorized |

            The Blackwell notebook is SM100-only. Do not try to execute its
            TMA/UMMA/tensor-memory kernel on H100. The local challenge preserves
            the decomposition while using mechanisms supported by the current
            pod.
            """
        ),
        markdown(
            """
            ## 10. Challenge: connect the complete dataflow

            Fill three connected regions:

            1. **CTA tiles:** use `local_tile` with the correct projections.
            2. **Ownership:** allocate shared tensors, partition copies/MMA, and
               allocate register fragments.
            3. **Mainloop and epilogue:** move each K tile through memory spaces,
               accumulate it, and store C.

            Track every variable by writing a sentence of the form:

            ```text
            tAgA maps this thread's copy coordinates
            to addresses in the CTA's global A tile.
            ```

            If you cannot state both the domain and destination, inspect that
            tensor before continuing.
            """
        ),
        code(scaffold),
        markdown(
            """
            ## 11. Correctness before performance

            Complete all TODO regions, then run a small problem. The PyTorch
            reference uses the same `(M,K) x (N,K)` convention:

            ```python
            torch.einsum("mk,nk->mn", a, b)
            ```
            """
        ),
        code(
            """
            run(M=256, N=256, K=64)
            """
        ),
        markdown(
            """
            ## 12. Benchmark and compute throughput

            Use CUDA events after warmup. If average duration is in
            milliseconds:

            ```text
            GFLOP/s = (2*M*N*K) / (average_ms / 1000) / 1e9
            TFLOP/s = GFLOP/s / 1000
            ```

            This SIMT teaching kernel is not expected to approach H100 tensor
            core peak. Its value is a transparent baseline whose movement,
            synchronization, and ownership can be inspected.
            """
        ),
        code(
            """
            # The reference script includes warmup, CUDA-event timing, and TFLOP/s.
            # Run it after your notebook implementation is correct.
            %run solution.py --M 1024 --N 1024 --K 256 --warmup 5 --iterations 50
            """
        ),
        markdown(
            """
            ## 13. Optimization path

            The official tour applies two major transformations after its first
            correct GEMM:

            1. **Software pipeline A/B:** prefetch future K tiles so memory
               latency overlaps current MMA. The stage count and protocol come
               from Day 07.
            2. **Vectorize the epilogue:** expose alignment and divisibility so
               CuTe can issue wide global stores.

            ![Software pipeline](../notebooks/images/software_pipelining_ab_stages_minus_2.svg)

            A practical progression for this H100 baseline is:

            ```text
            add edge predication
              -> add two or three A/B stages
              -> overlap copy and compute
              -> vectorize output
              -> replace SIMT MMA with an H100 tensor-core atom
            ```

            Make one transformation at a time, preserve the correctness check,
            and use Day 08's benchmark discipline to evaluate it.
            """
        ),
        markdown(
            """
            ## What you should know afterward

            - GEMM performance comes from tiling and reuse, not only the FMA.
            - `local_tile` selects a CTA problem; partitions assign its work.
            - Copy and MMA layouts define ownership before instructions execute.
            - Shared memory enables reuse but requires correct producer/consumer ordering.
            - Register fragments hold per-thread operands and accumulators.
            - A correct single-stage kernel exposes where pipelining can overlap work.
            - Hardware-specific SOL kernels preserve these roles while replacing
              mechanisms with TMA, tensor cores, specialized pipelines, and
              vectorized epilogues.
            """
        ),
    ]


def main() -> None:
    notebooks = {
        "day07_async_pipeline/day07.ipynb": day07_cells(),
        "day08_benchmark_autotune/day08.ipynb": day08_cells(),
        "day09_cuda_graphs/day09.ipynb": day09_cells(),
        "day10_gemm_tour/day10.ipynb": day10_cells(),
    }
    for relative_path, cells in notebooks.items():
        destination = DSL_ROOT / relative_path
        destination.write_text(
            json.dumps(notebook(cells), ensure_ascii=True, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
