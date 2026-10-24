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

            The first call for a new key benchmarks every candidate. Later
            calls with the same key invoke the cached compiled winner:

            ```python
            add(a, b, c, 512, 512)   # tunes 64 versus 128 bits
            add(a, b, c, 512, 512)   # cache hit; no benchmark
            add(a, b, c, 1024, 512)  # new (m, n), so tune again
            ```

            This cache is process-local. Restarting the Python or Jupyter
            kernel clears it. Calling `add_kernel(...).launch(...)` directly
            also bypasses tuning because the decorator wraps `add`, not the
            device kernel.

            To print candidate timings, the winner, and cache hits, set the
            environment variable before executing the decorated definition
            (restart the Jupyter kernel first if the function already exists):

            ```python
            import os
            os.environ["CUTE_DSL_LOG_AUTOTUNE"] = "1"
            ```

            After at least one call, the wrapper's internal cache shows the
            selected configuration for each key:

            ```python
            print(add._best_config)
            # {(512, 512): {"copy_bits": 128}}

            print(add._best_config[(512, 512)])
            # {"copy_bits": 128}
            ```

            `_best_config` is useful for debugging but is an internal API.
            Use explicit `testing.tune` when application code needs the winning
            dictionary as a supported return value.
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


def day10_optimization_cells() -> list[dict[str, object]]:
    return [
        markdown(
            """
            ## Baseline: one thread computes one output

            Start from the simplest correct mapping for
            `C[M,K] = A[M,N] @ B[N,K]`. A 32x32 CTA contains 1024 threads,
            and each thread computes one scalar output by reading the full
            reduction dimension from global memory.

            The baseline is intentionally inefficient. It gives the next two
            optimizations a stable correctness and timing reference.
            """
        ),
        code(
            """
            @cute.kernel
            def baseline_matmul_kernel(A, B, C, M, N, K):
                BLOCK_M: cutlass.Constexpr = 32
                BLOCK_K: cutlass.Constexpr = 32

                thread_idx, _, _ = cute.arch.thread_idx()
                block_m, block_k, _ = cute.arch.block_idx()
                local_m = thread_idx // BLOCK_K
                local_k = thread_idx % BLOCK_K
                row = block_m * BLOCK_M + local_m
                col = block_k * BLOCK_K + local_k

                if row < M and col < K:
                    accum = cutlass.Float32(0.0)
                    for n in cutlass.range(0, N):
                        accum += (
                            cutlass.Float32(A[row, n])
                            * cutlass.Float32(B[n, col])
                        )
                    C[row, col] = accum


            @cute.jit
            def solve_baseline(
                A: cute.Tensor,
                B: cute.Tensor,
                C: cute.Tensor,
                M: cute.Int32,
                N: cute.Int32,
                K: cute.Int32,
            ):
                baseline_matmul_kernel(A, B, C, M, N, K).launch(
                    grid=[cute.ceil_div(M, 32), cute.ceil_div(K, 32), 1],
                    block=[1024, 1, 1],
                )


            def measure_tflops(launch, m, n, k, *, warmup=3, iterations=20):
                for _ in range(warmup):
                    launch()
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    launch()
                end.record()
                torch.cuda.synchronize()

                average_ms = start.elapsed_time(end) / iterations
                tflops = (2.0 * m * n * k) / (average_ms / 1e3) / 1e12
                return average_ms, tflops


            M = N = K = 8192
            torch.manual_seed(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            A = torch.randn((M, N), device="cuda", dtype=torch.float16)
            B = torch.randn((N, K), device="cuda", dtype=torch.float16)
            C_baseline = torch.empty(
                (M, K), device="cuda", dtype=torch.float32
            )

            A_cute = from_dlpack(A)
            B_cute = from_dlpack(B)
            C_baseline_cute = from_dlpack(C_baseline)
            reference = A.float() @ B.float()

            compiled_baseline = cute.compile(
                solve_baseline,
                A_cute,
                B_cute,
                C_baseline_cute,
                M,
                N,
                K,
            )


            def launch_baseline():
                compiled_baseline(
                    A_cute, B_cute, C_baseline_cute, M, N, K
                )


            launch_baseline()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                C_baseline, reference, rtol=1e-2, atol=1e-2
            )
            baseline_ms, baseline_tflops = measure_tflops(
                launch_baseline, M, N, K
            )
            print(
                f"baseline: {baseline_ms:.4f} ms, "
                f"{baseline_tflops:.2f} TFLOP/s"
            )
            """
        ),
        markdown(
            """
            ## Optimization 1: reuse CTA tiles through shared memory

            The baseline reloads the same A and B values from global memory for
            many output threads. This version stages one 32x32 A tile and one
            32x32 B tile in shared memory for every reduction step.

            Every CTA thread must execute both barriers:

            1. after the cooperative stores, before any thread reads the tile;
            2. after all reads, before any thread overwrites the tile.

            Shared memory does not guarantee a speedup. On modern GPUs, the
            hardware caches already capture some reuse, while this kernel adds
            two CTA barriers after only 32 FMAs per thread.
            """
        ),
        code(
            """
            import triton
            from triton.runtime import driver

            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            triton_device = (
                triton.runtime.driver.active.get_active_torch_device()
            )
            triton_props = driver.active.utils.get_device_properties(
                triton_device.index
            )

            print("SM count:", props.multi_processor_count)
            print("registers/SM:", triton_props["max_num_regs"])
            print("threads/SM:", props.max_threads_per_multi_processor)
            print("threads/block:", props.max_threads_per_block)
            print("warp size:", props.warp_size)
            print(
                "maximum resident warps/SM:",
                props.max_threads_per_multi_processor // props.warp_size,
            )
            print(
                "opt-in shared memory/block:",
                props.shared_memory_per_block_optin,
                "bytes",
            )
            """
        ),
        code(
            """
            @cute.kernel
            def shared_matmul_kernel(
                A, B, C, sA_layout, sB_layout, M, N, K
            ):
                BLOCK_M: cutlass.Constexpr = 32
                BLOCK_N: cutlass.Constexpr = 32
                BLOCK_K: cutlass.Constexpr = 32

                thread_idx, _, _ = cute.arch.thread_idx()
                block_m, block_k, _ = cute.arch.block_idx()
                local_m = thread_idx // BLOCK_K
                local_k = thread_idx % BLOCK_K
                row = block_m * BLOCK_M + local_m
                col = block_k * BLOCK_K + local_k

                smem = cutlass.utils.SmemAllocator()
                sA = smem.allocate_tensor(A.element_type, sA_layout, 16)
                sB = smem.allocate_tensor(B.element_type, sB_layout, 16)
                accum = cutlass.Float32(0.0)

                for n0 in cutlass.range(0, N, BLOCK_N):
                    a_col = n0 + local_k
                    b_row = n0 + local_m
                    if row < M and a_col < N:
                        sA[local_m, local_k] = A[row, a_col]
                    else:
                        sA[local_m, local_k] = cutlass.Float16(0.0)
                    if b_row < N and col < K:
                        sB[local_m, local_k] = B[b_row, col]
                    else:
                        sB[local_m, local_k] = cutlass.Float16(0.0)

                    cute.arch.sync_threads()
                    if row < M and col < K:
                        for inner_n in cutlass.range_constexpr(32):
                            accum += (
                                cutlass.Float32(sA[local_m, inner_n])
                                * cutlass.Float32(sB[inner_n, local_k])
                            )
                    cute.arch.sync_threads()

                if row < M and col < K:
                    C[row, col] = accum


            @cute.jit
            def solve_shared(
                A: cute.Tensor,
                B: cute.Tensor,
                C: cute.Tensor,
                M: cute.Int32,
                N: cute.Int32,
                K: cute.Int32,
            ):
                sA_layout = cute.make_layout(
                    (32, 32), stride=(32, 1)
                )
                sB_layout = cute.make_layout(
                    (32, 32), stride=(32, 1)
                )
                shared_matmul_kernel(
                    A, B, C, sA_layout, sB_layout, M, N, K
                ).launch(
                    grid=[cute.ceil_div(M, 32), cute.ceil_div(K, 32), 1],
                    block=[1024, 1, 1],
                )


            C_shared = torch.empty_like(C_baseline)
            C_shared_cute = from_dlpack(C_shared)
            compiled_shared = cute.compile(
                solve_shared,
                A_cute,
                B_cute,
                C_shared_cute,
                M,
                N,
                K,
            )


            def launch_shared():
                compiled_shared(A_cute, B_cute, C_shared_cute, M, N, K)


            launch_shared()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                C_shared, reference, rtol=1e-2, atol=1e-2
            )
            shared_ms, shared_tflops = measure_tflops(
                launch_shared, M, N, K
            )
            print(
                f"shared: {shared_ms:.4f} ms, "
                f"{shared_tflops:.2f} TFLOP/s"
            )
            """
        ),
        markdown(
            """
            ### Why shared memory can still be slower

            Each CTA allocates two 32x32 FP16 tiles, or 4096 bytes total.
            Shared-memory capacity therefore is not the occupancy limiter.
            The 1024-thread CTA permits at most two resident CTAs on a GPU with
            2048 threads per SM, and every reduction tile executes two
            full-CTA barriers.

            The important lesson is that shared memory is a mechanism for
            controlled reuse, not an automatic optimization. Reuse must be
            large enough to amortize synchronization and address-generation
            overhead.
            """
        ),
        markdown(
            """
            ## Optimization 2: CuTe thread-layout and register tiling

            Follow the ownership model in
            `cuda-cpp/solutions/20_cute_sgemm_1.cu`. A 256-thread CTA computes
            a 128x128 output tile while traversing the reduction dimension in
            depth-8 tiles.

            CuTe layouts express two different ownership maps:

            - `(32,8)` copy layouts partition the 128x8 A and B shared tiles;
            - a `(16,16)` compute layout partitions the 128x128 C tile.

            Projecting the compute layout onto A's row mode and B's output
            mode gives every thread an 8x8 register accumulator. Each loaded A
            value is reused across eight output columns, and each loaded B
            value is reused across eight output rows.

            ```text
            CTA output tile:       128 x 128
            reduction tile:          8
            copy thread layout:     32 x 8
            compute thread layout:  16 x 16
            per-thread register:     8 x 8
            threads per CTA:       256
            ```
            """
        ),
        markdown(
            """
            ### Partition copy and compute with CuTe layouts

            This mirrors the C++ tutorial's `tA`, `tB`, and `tC` layouts:

            ```python
            copy_thread_layout = cute.make_layout(
                (32, 8), stride=(8, 1)
            )
            compute_thread_layout = cute.make_layout(
                (16, 16), stride=(1, 16)
            )
            ```

            `cute.local_partition` applies these layouts directly:

            - `tAgA` and `tAsA` partition matching global/shared A tiles;
            - `tBgB` and `tBsB` partition matching transposed-B/shared tiles;
            - projecting `compute_thread_layout` with `(1,None)` partitions A;
            - projecting it with `(None,1)` partitions B;
            - applying both modes partitions C and defines the accumulator
              shape.

            Like the referenced C++ tutorial, this first layout-tiled version
            assumes M and K are multiples of 128 and N is a multiple of 8.
            Predicated edge tiles are a separate transformation.
            """
        ),
        code(
            """
            @cute.kernel
            def warp_tiled_matmul_kernel(
                A,
                B,
                C,
                sA_layout,
                sB_layout,
                transposed_gB_layout,
                copy_thread_layout,
                compute_thread_layout,
                M,
                N,
                K,
            ):
                BLOCK_M: cutlass.Constexpr = 128
                BLOCK_N: cutlass.Constexpr = 8
                BLOCK_K: cutlass.Constexpr = 128

                thread_idx, _, _ = cute.arch.thread_idx()
                block_m, block_k, _ = cute.arch.block_idx()

                gC = cute.local_tile(
                    C,
                    (BLOCK_M, BLOCK_K),
                    (block_m, block_k),
                )
                tCgC = cute.local_partition(
                    gC, compute_thread_layout, thread_idx
                )

                smem = cutlass.utils.SmemAllocator()
                sA = smem.allocate_tensor(
                    A.element_type, sA_layout, 16
                )
                sB = smem.allocate_tensor(
                    B.element_type, sB_layout, 16
                )

                tCsA = cute.local_partition(
                    sA,
                    compute_thread_layout,
                    thread_idx,
                    proj=(1, None),
                )
                tCsB = cute.local_partition(
                    sB,
                    compute_thread_layout,
                    thread_idx,
                    proj=(None, 1),
                )
                accum = cute.make_rmem_tensor(
                    tCgC.shape, cutlass.Float32
                )
                accum.fill(0.0)

                for n0 in cutlass.range(0, N, BLOCK_N):
                    n_tile = n0 // BLOCK_N
                    gA = cute.local_tile(
                        A,
                        (BLOCK_M, BLOCK_N),
                        (block_m, n_tile),
                    )
                    gB = cute.local_tile(
                        B,
                        (BLOCK_N, BLOCK_K),
                        (n_tile, block_k),
                    )
                    gB_transposed = cute.make_tensor(
                        gB.iterator, transposed_gB_layout
                    )

                    tAgA = cute.local_partition(
                        gA, copy_thread_layout, thread_idx
                    )
                    tAsA = cute.local_partition(
                        sA, copy_thread_layout, thread_idx
                    )
                    tBgB = cute.local_partition(
                        gB_transposed,
                        copy_thread_layout,
                        thread_idx,
                    )
                    tBsB = cute.local_partition(
                        sB, copy_thread_layout, thread_idx
                    )

                    for value in cutlass.range_constexpr(
                        cute.size(tAgA)
                    ):
                        tAsA[value] = tAgA[value]
                    for value in cutlass.range_constexpr(
                        cute.size(tBgB)
                    ):
                        tBsB[value] = tBgB[value]

                    cute.arch.sync_threads()

                    for inner_n in cutlass.range_constexpr(8):
                        for i in cutlass.range_constexpr(8):
                            a_value = cutlass.Float32(
                                tCsA[i, inner_n]
                            )
                            for j in cutlass.range_constexpr(8):
                                accum[i, j] += (
                                    a_value
                                    * cutlass.Float32(
                                        tCsB[j, inner_n]
                                    )
                                )

                    cute.arch.sync_threads()

                for value in cutlass.range_constexpr(cute.size(tCgC)):
                    tCgC[value] = accum[value]


            @cute.jit
            def solve_warp_tiled(
                A: cute.Tensor,
                B: cute.Tensor,
                C: cute.Tensor,
                M: cute.Int32,
                N: cute.Int32,
                K: cute.Int32,
            ):
                sA_layout = cute.make_layout(
                    (128, 8), stride=(8, 1)
                )
                sB_layout = cute.make_layout(
                    (128, 8), stride=(8, 1)
                )
                transposed_gB_layout = cute.make_layout(
                    (128, 8), stride=(1, B.stride[0])
                )
                copy_thread_layout = cute.make_layout(
                    (32, 8), stride=(8, 1)
                )
                compute_thread_layout = cute.make_layout(
                    (16, 16), stride=(1, 16)
                )

                warp_tiled_matmul_kernel(
                    A,
                    B,
                    C,
                    sA_layout,
                    sB_layout,
                    transposed_gB_layout,
                    copy_thread_layout,
                    compute_thread_layout,
                    M,
                    N,
                    K,
                ).launch(
                    grid=[
                        cute.ceil_div(M, 128),
                        cute.ceil_div(K, 128),
                        1,
                    ],
                    block=[256, 1, 1],
                )


            C_warp_tiled = torch.empty_like(C_baseline)
            C_warp_tiled_cute = from_dlpack(C_warp_tiled)
            compiled_warp_tiled = cute.compile(
                solve_warp_tiled,
                A_cute,
                B_cute,
                C_warp_tiled_cute,
                M,
                N,
                K,
            )


            def launch_warp_tiled():
                compiled_warp_tiled(
                    A_cute, B_cute, C_warp_tiled_cute, M, N, K
                )


            launch_warp_tiled()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                C_warp_tiled, reference, rtol=1e-2, atol=1e-2
            )
            warp_tiled_ms, warp_tiled_tflops = measure_tflops(
                launch_warp_tiled, M, N, K
            )
            print(
                f"warp tiled: {warp_tiled_ms:.4f} ms, "
                f"{warp_tiled_tflops:.2f} TFLOP/s"
            )
            """
        ),
    ]


def day10_tiled_mma_cells() -> list[dict[str, object]]:
    return [
        markdown(
            """
            ## 10. Optimization 3: package ownership with TiledCopy and TiledMma

            The previous kernel exposed ownership with raw thread layouts.
            This implementation follows
            `cuda-cpp/challenge/21_cute_sgemm_2.cu` and packages the same
            structure into two higher-level CuTe objects:

            - `TiledCopy` combines a copy atom, thread layout, and value layout;
            - `TiledMma` combines a universal FMA atom with its replicated
              thread layout.

            The CTA remains 128x128x8 with 256 threads. Each copy atom moves
            four FP32 values, while the 16x16 MMA-atom layout partitions the
            CTA output across all threads. The mainloop also demonstrates the
            Day 21 register-prefetch sequence:

            ```text
            global -> prefetched register fragment
                   -> shared-memory tile
                   -> MMA register fragments
                   -> FP32 accumulator
            ```

            `MmaUniversalOp(Float32)` uses FP32 A/B fragments, so this cell
            builds column-major FP32 views matching the C++ SGEMM exercise.
            The next section then defines the FP16 tensor-core contract.
            """
        ),
        code(
            """
            @cute.kernel
            def tiled_mma_sgemm_kernel(
                mA,
                mB,
                mC,
                sA_layout,
                sB_layout,
                tiled_copy_A,
                tiled_copy_B,
                tiled_mma,
            ):
                cta_tiler = (128, 128, 8)
                tidx, _, _ = cute.arch.thread_idx()
                bidx, bidy, _ = cute.arch.block_idx()

                coord = (bidx, bidy, None)
                gA = cute.local_tile(
                    mA,
                    tiler=cta_tiler,
                    coord=coord,
                    proj=(1, None, 1),
                )
                gB = cute.local_tile(
                    mB,
                    tiler=cta_tiler,
                    coord=coord,
                    proj=(None, 1, 1),
                )
                gC = cute.local_tile(
                    mC,
                    tiler=cta_tiler,
                    coord=coord,
                    proj=(1, 1, None),
                )

                smem = cutlass.utils.SmemAllocator()
                sA = smem.allocate_tensor(
                    mA.element_type, sA_layout, 16
                )
                sB = smem.allocate_tensor(
                    mB.element_type, sB_layout, 16
                )

                thr_copy_A = tiled_copy_A.get_slice(tidx)
                thr_copy_B = tiled_copy_B.get_slice(tidx)
                tAgA = thr_copy_A.partition_S(gA)
                tAsA = thr_copy_A.partition_D(sA)
                tBgB = thr_copy_B.partition_S(gB)
                tBsB = thr_copy_B.partition_D(sB)
                tArA = cute.make_fragment_like(tAsA)
                tBrB = cute.make_fragment_like(tBsB)

                thr_mma = tiled_mma.get_slice(tidx)
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCgC = thr_mma.partition_C(gC)
                tCrA = tiled_mma.make_fragment_A(tCsA)
                tCrB = tiled_mma.make_fragment_B(tCsB)
                tCrC = tiled_mma.make_fragment_C(tCgC)
                tCrC.fill(0.0)

                k_tile_count = cute.size(tAgA, mode=[3])
                k_block_max = cute.size(tCrA, mode=[2])

                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, 0],
                    tArA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, 0],
                    tBrB,
                )

                for k_tile in range(k_tile_count):
                    cute.arch.sync_threads()
                    cute.copy(tiled_copy_A, tArA, tAsA)
                    cute.copy(tiled_copy_B, tBrB, tBsB)
                    cute.arch.sync_threads()

                    k_tile_next = (
                        k_tile + 1
                        if k_tile + 1 < k_tile_count
                        else k_tile
                    )
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, k_tile_next],
                        tArA,
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, k_tile_next],
                        tBrB,
                    )

                    for k_block in cutlass.range_constexpr(
                        k_block_max
                    ):
                        cute.autovec_copy(
                            tCsA[None, None, k_block],
                            tCrA[None, None, k_block],
                        )
                        cute.autovec_copy(
                            tCsB[None, None, k_block],
                            tCrB[None, None, k_block],
                        )
                        cute.gemm(
                            tiled_mma,
                            tCrC,
                            tCrA[None, None, k_block],
                            tCrB[None, None, k_block],
                            tCrC,
                        )

                store_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), mC.element_type
                )
                cute.copy(store_atom, tCrC, tCgC)


            @cute.jit
            def solve_tiled_mma(
                mA: cute.Tensor,
                mB: cute.Tensor,
                mC: cute.Tensor,
            ):
                sA_layout = cute.make_layout(
                    (128, 8), stride=(1, 128)
                )
                sB_layout = cute.make_layout(
                    (128, 8), stride=(1, 128)
                )

                copy_atom_A = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    mA.element_type,
                    num_bits_per_copy=128,
                )
                copy_atom_B = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    mB.element_type,
                    num_bits_per_copy=128,
                )
                copy_thread_layout = cute.make_layout(
                    (32, 8), stride=(1, 32)
                )
                copy_value_layout = cute.make_layout((4, 1))
                tiled_copy_A = cute.make_tiled_copy_tv(
                    copy_atom_A,
                    copy_thread_layout,
                    copy_value_layout,
                )
                tiled_copy_B = cute.make_tiled_copy_tv(
                    copy_atom_B,
                    copy_thread_layout,
                    copy_value_layout,
                )

                mma_op = cute.nvgpu.MmaUniversalOp(
                    cutlass.Float32
                )
                mma_atom_layout = cute.make_layout(
                    (16, 16, 1), stride=(1, 16, 0)
                )
                tiled_mma = cute.make_tiled_mma(
                    mma_op, mma_atom_layout
                )

                tiled_mma_sgemm_kernel(
                    mA,
                    mB,
                    mC,
                    sA_layout,
                    sB_layout,
                    tiled_copy_A,
                    tiled_copy_B,
                    tiled_mma,
                ).launch(
                    grid=[
                        cute.ceil_div(mC.shape[0], 128),
                        cute.ceil_div(mC.shape[1], 128),
                        1,
                    ],
                    block=[cute.size(mma_atom_layout), 1, 1],
                )


            A_tiled_mma = A.float().T.contiguous().T
            B_tiled_mma = B.float().T
            C_tiled_mma = (
                torch.empty(
                    (K, M), device="cuda", dtype=torch.float32
                )
                .zero_()
                .permute(1, 0)
            )

            A_tiled_mma_cute = from_dlpack(
                A_tiled_mma, assumed_align=16
            )
            B_tiled_mma_cute = from_dlpack(
                B_tiled_mma, assumed_align=16
            )
            C_tiled_mma_cute = from_dlpack(
                C_tiled_mma, assumed_align=16
            )
            compiled_tiled_mma = cute.compile(
                solve_tiled_mma,
                A_tiled_mma_cute,
                B_tiled_mma_cute,
                C_tiled_mma_cute,
            )


            def launch_tiled_mma():
                compiled_tiled_mma(
                    A_tiled_mma_cute,
                    B_tiled_mma_cute,
                    C_tiled_mma_cute,
                )


            launch_tiled_mma()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                C_tiled_mma, reference, rtol=1e-4, atol=1e-3
            )
            tiled_mma_ms, tiled_mma_tflops = measure_tflops(
                launch_tiled_mma, M, N, K
            )
            print(
                f"TiledMma: {tiled_mma_ms:.4f} ms, "
                f"{tiled_mma_tflops:.2f} TFLOP/s"
            )
            """
        ),
    ]


def day10_cells() -> list[dict[str, object]]:
    return [
        markdown(
            """
            # Day 10 - Tour from layouts to a complete GEMM

            Official [Notebook 12](../notebooks/12_tour_to_sol_gemm.ipynb)
            builds a Blackwell speed-of-light GEMM with TMA, UMMA, tensor
            memory, software pipelining, and a vectorized epilogue. This
            notebook teaches the same algorithmic roles with an
            H100-compatible, single-stage SIMT progression using FP16 inputs
            and FP32 accumulation, then develops a tensor-core roadmap for:

            - H100/SM90: TMA plus warpgroup MMA (WGMMA);
            - B300/SM103: TMA plus `tcgen05.mma`, tensor memory, and a
              subtiled FP16 epilogue.

            The objective is to connect the earlier lessons:

            ```text
            layouts -> CTA tiles -> cooperative copy -> shared memory
                    -> per-thread MMA fragments -> K reduction -> epilogue
            ```

            The local challenge favors visibility over peak throughput. The
            final sections preserve that dataflow while replacing one mechanism
            at a time with the architecture-specific fast path.
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
            C_{m,k} = \sum_{n=0}^{N-1} A_{m,n} B_{n,k}.
            \]

            Therefore A has logical shape `(M, N)`, B has logical shape
            `(N, K)`, and C has shape `(M, K)`, matching the usual
            `torch.mm(A, B)` convention.

            Each output element performs N multiplies and N accumulations,
            conventionally counted as `2*N` floating-point operations. Total
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
                minimum_bytes = (m * n + n * k + m * k) * element_bytes
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
        *day10_optimization_cells(),
        markdown(
            """
            ## 6. From thread-layout tiling to MMA ownership

            Optimization 2 uses the same partitioning idea as the CuTe C++
            tutorial: copy layouts own global/shared movement, while a compute
            layout owns shared/register movement and C. The 8x8 accumulator
            shape is derived from `tCgC.shape`, not duplicated in the
            allocation.

            CuTe's next abstraction, `TiledMma`, packages that ownership around
            a selected matrix instruction. Each thread first obtains a slice:

            ```python
            thr_mma = tiled_mma.get_slice(tidx)
            ```

            Then:

            - `partition_A(sA)` maps the shared A tile into that thread's MMA A view;
            - `partition_B(sB)` does the same for B;
            - `partition_C(gC)` maps accumulator ownership to output addresses;
            - `make_fragment_A/B/C` allocates matching register fragments.

            Partitioned tensors are mappings, not copies. Data enters A/B
            register fragments only when a copy executes. The accumulator
            fragment must be initialized to zero before the reduction.
            """
        ),
        markdown(
            """
            ## 7. The single-stage reduction mainloop

            The standalone kernel reduces over N tiles:

            ```text
            global A/B --cooperative loads--> shared A/B
                    CTA barrier
            shared A/B --scalar reads--> register values
            register values --SIMT FMA--> 8x8 register accumulator
                    CTA barrier before shared-memory overwrite
            ```

            This ordering is intentionally conservative. The first barrier
            prevents compute from reading incomplete data; the second prevents
            the next iteration from overwriting shared memory while another
            thread still consumes it. TMA and asynchronous pipelines later
            overlap these phases without changing the dependency.
            """
        ),
        code(
            """
            BLK_M, REDUCTION_TILE, BLK_K = 128, 8, 128
            outputs_per_cta = BLK_M * BLK_K
            fmas_per_reduction_tile = (
                BLK_M * BLK_K * REDUCTION_TILE
            )
            print(f"outputs per CTA: {outputs_per_cta}")
            print(f"FMAs per N tile: {fmas_per_reduction_tile}")
            print(f"FLOPs per N tile: {2 * fmas_per_reduction_tile}")
            """
        ),
        markdown(
            """
            ## 8. Epilogue: accumulator to global C

            After all N tiles, each thread's 8x8 register tensor contains its
            complete outputs. The simple epilogue copies the flattened
            accumulator into the matching `tCgC` partition and stores FP32
            values directly to C.

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
            ## 9. Implementation ladder for the same GEMM

            | Role | Layout-tiled SIMT | TiledCopy/TiledMma bridge | H100 / SM90 | B300 / SM103 |
            | --- | --- | --- | --- | --- |
            | A/B datatype | FP16 | FP32 | FP16 | FP16 |
            | Accumulator | FP32 registers | FP32 registers | FP32 registers | FP32 tensor memory |
            | Global-to-shared | copy thread layout | `TiledCopy` | TMA | TMA |
            | Matrix operation | scalar/SIMT FMA | `MmaUniversalOp` | warpgroup MMA | `tcgen05.mma` |
            | Synchronization | CTA barriers | CTA barriers | `PipelineTmaAsync` | `PipelineTmaUmma` |
            | Scheduling unit | copy/compute layouts | `TiledCopy`/`TiledMma` slices | 128-thread warpgroup | one- or two-CTA MMA group |
            | Epilogue | direct FP32 register store | universal FP32 store | vectorized register/shared store | TMEM -> registers -> FP16 vector/TMA store |

            H100 has compute capability 9.0 and can execute the baseline plus
            the SM90 path. B300 is compute capability 10.3 and needs an SM103
            toolchain. The pinned CUTLASS release documents CUDA 13.0 as the
            minimum for B300 and CuTe DSL GB300 support with CUDA Toolkit 13.1.
            The current H100 pod cannot execute or validate SM103 code.

            WGMMA is not the compatibility path for B300. The checked-in CuTe
            headers guard it behind `CUTE_ARCH_MMA_SM90A_ENABLED`, while
            SM100/SM103 enable the `tcgen05` and TMEM feature families. Keep a
            shared host interface if useful, but compile separate SM90 WGMMA
            and SM103 `tcgen05.mma` kernels.

            Official Notebook 12 is an SM100 implementation. Its `tcgen05`,
            TMEM, and pipeline concepts transfer to SM103, but instruction
            selection, target flags such as `sm_103a`, tile choices, and
            performance must be revalidated on B300.
            """
        ),
        *day10_tiled_mma_cells(),
        markdown(
            """
            ## 11. Connect the complete standalone dataflow

            Optimization 2 exposes every ownership mapping directly. Read it
            from the output coordinate backward:

            1. `compute_thread_layout` partitions the 128x128 C tile.
            2. Projecting it onto one mode partitions shared A and B.
            3. The resulting C partition defines an 8x8 register accumulator.
            4. `copy_thread_layout` partitions matching global/shared tiles.
            5. Two CTA barriers protect each shared-memory stage.

            This keeps the implementation in the same standalone
            `@cute.kernel` plus `@cute.jit` style as the baseline and
            Optimization 1, instead of switching to the later class-based
            `SimpleSGemm` scaffold. Optimization 3 preserves that dataflow but
            packages the copy and compute ownership in `TiledCopy` and
            `TiledMma`.
            """
        ),
        markdown(
            """
            ## 12. Define the FP16 contract before tensor-core optimization

            Use FP16 for A, B, and the final output, but accumulate in FP32:

            ```text
            A: FP16
            B: FP16
            accumulator: FP32
            C/D: FP16 after epilogue conversion
            ```

            The first three teaching kernels use FP16 A/B with FP32
            accumulation and output. The `TiledMma` bridge deliberately uses
            FP32 operands because `MmaUniversalOp(Float32)` creates FP32
            fragments. The tensor-core contract returns to FP16 A/B, changes
            the epilogue output to FP16, and retains FP32 accumulation.
            Converting to FP16 should happen once, after the complete
            reduction. A correctness reference should accumulate in FP32:

            ```python
            reference = (A.float() @ B.float()).half()
            torch.testing.assert_close(
                output, reference, atol=2e-2, rtol=2e-2
            )
            ```

            The tolerance must be chosen for the tested K range and data
            distribution rather than copied blindly. Also test NaN/Inf behavior,
            odd edge tiles, and inputs with very different magnitudes.
            """
        ),
        code(
            """
            def staged_ab_smem_bytes(tile_mnk, stages, element_bytes=2):
                block_m, block_n, block_k = tile_mnk
                one_stage = (block_m * block_k + block_n * block_k) * element_bytes
                return one_stage, one_stage * stages


            candidates = [
                ("H100", (128, 128, 64), 3),
                ("B300", (128, 256, 64), 3),
            ]
            for name, tile, stages in candidates:
                one, total = staged_ab_smem_bytes(tile, stages)
                print(
                    f"{name}: tile={tile}, one A/B stage={one / 1024:.1f} KiB, "
                    f"{stages} stages={total / 1024:.1f} KiB"
                )
            """
        ),
        markdown(
            """
            ## 13. H100 path: FP16 TMA plus WGMMA

            The first H100 transformation is not merely replacing the copy
            instruction. TMA changes ownership and synchronization:

            1. A host-side descriptor encodes the global tensor and shared tile.
            2. One elected thread issues a bulk tensor load.
            3. An mbarrier records the expected transaction byte count.
            4. A 128-thread warpgroup waits for a full stage and issues WGMMA.
            5. The consumer releases that stage so TMA can refill it.

            The installed CuTe DSL Hopper helper constructs an FP16 WGMMA with
            FP32 accumulation:

            ```python
            import cutlass.pipeline as pipeline
            import cutlass.utils.hopper_helpers as sm90_utils

            tiled_mma = sm90_utils.make_trivial_tiled_mma(
                cutlass.Float16,                 # A
                cutlass.Float16,                 # B
                a_major_mode,
                b_major_mode,
                cutlass.Float32,                 # accumulator
                atom_layout_mnk=(1, 1, 1),
                tiler_mn=(64, 128),
            )
            ```

            `a_major_mode` and `b_major_mode` must match the shared-memory
            layouts consumed by WGMMA. A convenient starting CTA tile is
            `(128, 128, 64)` with three A/B stages, but it is a benchmark
            candidate rather than a universal optimum.
            """
        ),
        markdown(
            """
            ### 13.1 Build TMA descriptors and a staged pipeline

            A simplified host-side descriptor pattern is:

            ```python
            op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            a_tma_atom, a_tma_tensor = (
                cute.nvgpu.cpasync.make_tiled_tma_atom(
                    op, a, a_smem_layout_one_stage, (block_m, block_k)
                )
            )
            b_tma_atom, b_tma_tensor = (
                cute.nvgpu.cpasync.make_tiled_tma_atom(
                    op, b, b_smem_layout_one_stage, (block_n, block_k)
                )
            )
            ```

            In the kernel, `PipelineTmaAsync` replaces the unconditional
            `cp_async_wait_group(0)`:

            ```python
            mainloop_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
                num_stages=ab_stages,
                producer_group=producer_group,
                consumer_group=consumer_group,
                tx_count=tma_copy_bytes,
                cta_layout_vmnk=cta_layout_vmnk,
            )
            ```

            The producer acquires an empty stage, issues A/B TMA loads, and
            commits it. The WGMMA consumer waits for that stage, accumulates,
            and releases it. Prologue and tail handling are required so the
            ring never reads an unfilled slot or overwrites an in-use slot.

            ![Software pipeline](../notebooks/images/software_pipelining_ab_stages_minus_2.svg)
            """
        ),
        code(
            """
            def pipeline_schedule(k_tiles, stages=3):
                if stages < 2:
                    raise ValueError("overlap needs at least two stages")
                prefetch_distance = stages - 1
                events = []
                for step in range(k_tiles + prefetch_distance):
                    load = step if step < k_tiles else None
                    compute_idx = step - prefetch_distance
                    compute = compute_idx if 0 <= compute_idx < k_tiles else None
                    events.append((step, load, compute))
                return events


            for step, load, compute in pipeline_schedule(k_tiles=6, stages=3):
                print(
                    f"step {step}: "
                    f"TMA load={load if load is not None else '-'}; "
                    f"WGMMA compute={compute if compute is not None else '-'}"
                )
            """
        ),
        markdown(
            """
            ### 13.2 H100 optimization order

            Apply and measure these independently:

            1. FP16 inputs with FP32 accumulation.
            2. WGMMA while retaining a simple copy path.
            3. TMA descriptors and swizzled staged shared layouts.
            4. Two- then three-stage producer/consumer overlap.
            5. Warp specialization and persistent tile scheduling.
            6. FP32-to-FP16 conversion plus vectorized or TMA output stores.
            7. Cluster multicast only when neighboring CTAs reuse A or B.

            The complete tracked C++ reference is
            [`24_cute_hopper_wgmma_tma.cu`](../../cuda-cpp/solutions/24_cute_hopper_wgmma_tma.cu).
            It uses a `(128,128,64)` tile, three stages, FP16 TMA loads, and
            WGMMA. The production-style CuTe DSL implementation in the pinned
            CUTLASS source is `examples/python/CuTeDSL/cute/hopper/kernel/`
            `dense_gemm/dense_gemm.py`.
            """
        ),
        markdown(
            """
            ## 14. B300 path: SM103 `tcgen05.mma` and tensor memory

            Blackwell changes the accumulator dataflow. `tcgen05.mma` consumes
            A/B from shared memory but places the FP32 accumulator in tensor
            memory (TMEM), not ordinary per-thread registers:

            ```text
            FP16 A/B in GMEM
                 -> TMA -> staged, swizzled SMEM
                 -> tcgen05.mma
                 -> FP32 accumulator in TMEM
                 -> tcgen05.ld -> per-thread registers
                 -> FP16 conversion/fusion
                 -> vectorized or TMA store
            ```

            The official CuTe DSL construction used by Notebook 12 is:

            ```python
            from cutlass.cute.nvgpu import tcgen05

            op = tcgen05.MmaF16BF16Op(
                cutlass.Float16,
                cutlass.Float32,
                (128, 256, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
            )
            tiled_mma = cute.make_tiled_mma(op)
            ```

            FP16 `tcgen05.mma` has native K depth 16 in this configuration.
            A practical K tile groups several instructions, for example
            `4 * 16 = 64`.
            """
        ),
        markdown(
            """
            ### 14.1 TMA/UMMA pipeline and TMEM lifetime

            Blackwell's mainloop uses two protocols:

            - `PipelineTmaUmma`: TMA produces full A/B stages and UMMA releases
              consumed stages;
            - `PipelineUmmaAsync`: the MMA producer signals when the TMEM
              accumulator is ready for the epilogue.

            The kernel must explicitly manage TMEM:

            ```python
            tmem = cutlass.utils.TmemAllocator(
                storage.tmem_holding_buf.ptr,
                barrier_for_retrieve=tmem_alloc_barrier,
            )
            tmem.allocate(num_tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            ...
            tmem.relinquish_alloc_permit()
            ...
            tmem.free(tmem_ptr)
            ```

            Only an elected warp needs to issue TMA and `tcgen05.mma`, but all
            participating threads must obey the allocation, pipeline, and
            epilogue barriers. A missing release or TMEM lifetime transition can
            deadlock even when the tensor layouts are correct.
            """
        ),
        markdown(
            """
            ### 14.2 Subtile the FP16 epilogue

            A `(128,256)` FP32 accumulator is too large to copy from TMEM into
            registers in one piece without severe register pressure. Partition
            it into epilogue subtiles:

            ```python
            tmem_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
                cutlass.Float32,
            )
            tmem_copy = tcgen05.make_tmem_copy(
                tmem_atom, accumulator_subtile
            )

            for subtile in range(subtile_count):
                cute.copy(tmem_copy, tmem_source[subtile], rmem_acc)
                rmem_fp16.store(rmem_acc.load().to(cutlass.Float16))
                cute.autovec_copy(rmem_fp16, global_output[subtile])
            ```

            Subtiling reduces live registers and permits overlap between
            `tcgen05.ld`, conversion/fusion, and stores. The next steps are a
            TMA epilogue, one-to-two CTA-group experiments, and multicast when
            the cluster shape creates real operand reuse.
            """
        ),
        markdown(
            """
            ### 14.3 B300 execution gate

            Do not run the SM100 notebook binary on the H100 pod and do not
            assume SM100 launch parameters are optimal for B300. For B300:

            - compile for the SM103 architecture family, commonly `sm_103a`
              when architecture-specific features are used;
            - use CUDA 13.0 or newer, with CUDA Toolkit 13.1 for the pinned
              release's CuTe DSL GB300 support;
            - rebuild descriptors and kernels on the B300 host;
            - rerun correctness, stage-count, tile, CTA-group, and cluster
              searches on that hardware.

            Follow the tracked references in order:

            1. [`25_cute_blackwell_mma.cu`](../../cuda-cpp/solutions/25_cute_blackwell_mma.cu):
               `tcgen05.mma` plus TMEM.
            2. [`26_cute_blackwell_mma_tma.cu`](../../cuda-cpp/solutions/26_cute_blackwell_mma_tma.cu):
               add TMA.
            3. [`27_cute_blackwell_mma_tma_2sm.cu`](../../cuda-cpp/solutions/27_cute_blackwell_mma_tma_2sm.cu):
               two-SM MMA and multicast.
            4. [`28_cute_blackwell_mma_tma_epilogue.cu`](../../cuda-cpp/solutions/28_cute_blackwell_mma_tma_epilogue.cu):
               optimized TMA epilogue.
            """
        ),
        markdown(
            """
            ## 15. Measurement ladder

            Keep one correctness oracle and record a row after every isolated
            transformation:

            | Version | Required observation |
            | --- | --- |
            | FP16 SIMT + FP32 output | Correct mapping and synchronization |
            | FP16 epilogue + FP32 accumulate | Error remains within the declared contract |
            | Tensor-core MMA | Tensor-core instructions appear in profiling |
            | TMA | Lower load-instruction pressure; descriptor path is active |
            | Multistage pipeline | Fewer long scoreboard / memory dependency stalls |
            | Persistent schedule | Better steady-state utilization on large grids |
            | Optimized epilogue | Lower store time and controlled register pressure |

            Report TFLOP/s, achieved bandwidth, occupancy, registers/thread,
            dynamic shared memory, and the dominant stall reason. A faster
            kernel with incorrect edge behavior or unexplained numerical drift
            does not advance to the next stage.
            """
        ),
        markdown(
            """
            ## 16. Register and compare the tensor-core result

            Package the architecture-specific implementation in
            `optimized_case`. The H100 version should use WGMMA; the B300
            version should use `tcgen05.mma`. Both should use FP16 A/B/output
            with FP32 accumulation and the same benchmark dimensions as the
            baseline.

            ```python
            optimized_case = {
                "name": "B300 FP16 TMA + tcgen05",
                "launch": launch_b300_tcgen05,
                "output": b300_output,
                "reference": b300_reference,
                "atol": 2e-2,
                "rtol": 2e-2,
            }
            ```

            Once registered, the existing `measure_tflops` helper compares it
            against the CuTe layout-tiled teaching kernel. No second benchmark
            framework is needed.
            """
        ),
        code(
            """
            device_capability = torch.cuda.get_device_capability()
            if device_capability == (9, 0):
                expected_path = "H100/SM90: register FP16 TMA + WGMMA"
            elif device_capability >= (10, 3):
                expected_path = "B300/SM103: register FP16 TMA + tcgen05"
            else:
                expected_path = (
                    f"unsupported optimization target {device_capability}; "
                    "the SIMT teaching kernels may still run"
                )
            print(expected_path)

            if "optimized_case" not in globals():
                print(
                    "Define optimized_case after implementing the architecture-"
                    "specific kernel, then rerun this cell."
                )
            else:
                optimized_case["output"].zero_()
                optimized_case["launch"]()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    optimized_case["output"],
                    optimized_case["reference"],
                    atol=optimized_case["atol"],
                    rtol=optimized_case["rtol"],
                )
                optimized_ms, optimized_tflops = measure_tflops(
                    optimized_case["launch"],
                    M,
                    N,
                    K,
                    warmup=10,
                    iterations=50,
                )
                print(
                    f"CuTe layout-tiled SIMT: {warp_tiled_ms:.4f} ms, "
                    f"{warp_tiled_tflops:.2f} TFLOP/s"
                )
                print(
                    f"{optimized_case['name']}: {optimized_ms:.4f} ms, "
                    f"{optimized_tflops:.2f} TFLOP/s, "
                    f"{optimized_tflops / warp_tiled_tflops:.2f}x"
                )
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
            - H100 replaces cooperative copies and SIMT FMA with TMA, WGMMA,
              staged mbarrier pipelines, and an optimized FP16 epilogue.
            - B300 moves FP32 accumulators into TMEM and coordinates TMA with
              `tcgen05.mma` through architecture-specific pipelines.
            - SM100 teaching code is a conceptual bridge to SM103, not a
              substitute for compiling and tuning on B300.
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
