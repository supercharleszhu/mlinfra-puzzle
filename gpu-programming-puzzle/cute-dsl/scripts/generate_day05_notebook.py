#!/usr/bin/env python3
"""Generate the explanation-rich Day 05 matrix-transpose notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


DSL_ROOT = Path(__file__).resolve().parents[1]
DAY_ROOT = DSL_ROOT / "day05_matrix_transpose"


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


def shared_naive_puzzle_scaffold() -> str:
    source = (DAY_ROOT / "puzzle.py").read_text(encoding="utf-8")
    start = source.index("@cute.kernel\ndef shared_transpose_kernel")
    end = source.index("\n\n@cute.jit\ndef transpose_padded")
    return source[start:end].strip()


def padded_puzzle_scaffold() -> str:
    source = (DAY_ROOT / "puzzle.py").read_text(encoding="utf-8")
    start = source.index("@cute.jit\ndef transpose_padded")
    end = source.index("\n\n@cute.jit\ndef transpose_swizzled")
    return source[start:end].strip()


def swizzled_puzzle_scaffold() -> str:
    source = (DAY_ROOT / "puzzle.py").read_text(encoding="utf-8")
    start = source.index("@cute.jit\ndef transpose_swizzled")
    end = source.index("\n\ndef benchmark(")
    return source[start:end].strip()


def challenge_a_user_source() -> str:
    source = (DAY_ROOT / "solution.py").read_text(encoding="utf-8")
    start = source.index("@cute.kernel")
    end = source.index("\n\n@cute.kernel\ndef shared_transpose_kernel")
    return source[start:end].strip()


def main() -> None:
    cells = [
        markdown(
            """
            # Day 05 - Matrix transpose: naive to tiled

            This challenge is an original Python CuTe DSL lesson based on the
            algorithmic progression in Lei Mao's
            [CuTe Matrix Transpose](https://leimao.github.io/article/CuTe-Matrix-Transpose/):

            1. express transpose as two views of the same logical coordinates;
            2. observe that a direct kernel must make either reads or writes strided;
            3. stage a tile through shared memory so both global directions are coalesced;
            4. remove the resulting shared-memory bank conflict with padding or swizzling.

            The article uses C++ CuTe. This notebook independently implements
            two naive direct mappings plus unpadded, padded, and swizzled
            shared-memory variants with CuTe DSL on H100.
            """
        ),
        code(
            """
            import torch

            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack

            try:
                import cutlass.testing as testing
            except ModuleNotFoundError as error:
                if error.name != "cutlass.testing":
                    raise
                from cutlass.cute import testing

            cutlass.cuda.initialize_cuda_context()

            H100_SXM_PEAK_GBPS = 3_350.0


            def check_transpose_correctness(
                name,
                transpose_fn,
                shapes=((64, 96), (1003, 1507)),
            ):
                for m, n in shapes:
                    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
                    b = torch.empty(n, m, device="cuda", dtype=torch.float32)
                    a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
                    b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
                    compiled = cute.compile(transpose_fn, a_, b_)
                    compiled(a_, b_)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(b, a.T)
                print(f"{name}: correctness OK")


            def benchmark_transpose_bandwidth(
                name,
                transpose_fn,
                m=4096,
                n=4096,
                warmup=10,
                iterations=100,
            ):
                a = torch.randn(m, n, device="cuda", dtype=torch.float32)
                b = torch.empty(n, m, device="cuda", dtype=torch.float32)
                a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
                b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
                compiled = cute.compile(transpose_fn, a_, b_)
                compiled(a_, b_)
                torch.cuda.synchronize()
                torch.testing.assert_close(b, a.T)

                average_us = testing.benchmark(
                    compiled,
                    kernel_arguments=testing.JitArguments(a_, b_),
                    warmup_iterations=warmup,
                    iterations=iterations,
                )
                traffic_bytes = 2 * a.numel() * a.element_size()
                bandwidth_gbps = traffic_bytes / (average_us * 1_000)
                utilization = 100 * bandwidth_gbps / H100_SXM_PEAK_GBPS
                print(
                    f"{name}: {average_us:.3f} us, "
                    f"{bandwidth_gbps:.2f} GB/s, "
                    f"{utilization:.2f}% of H100 SXM peak"
                )
                return average_us, bandwidth_gbps, utilization
            """
        ),
        markdown(
            """
            ## 1. Transpose is a coordinate permutation

            For an input `A` of shape `(M, N)`, the row-major output `B` has
            shape `(N, M)` and:

            \[
            B[n,m] = A[m,n].
            \]

            The values do not change; only their coordinate-to-address mapping
            changes. For row-major layouts:

            ```text
            A offset(m,n) = m*N + n
            B offset(n,m) = n*M + m
            ```

            CuTe encourages this view: a tensor is an engine plus a layout.
            The transpose can be understood as pairing the same logical
            `(m,n)` domain with row-major source addressing and column-major
            destination addressing. In this challenge the output remains a
            normal row-major `(N,M)` PyTorch tensor, so the kernel explicitly
            swaps the output coordinates.
            """
        ),
        code(
            """
            M, N = 3, 5
            a = torch.arange(M * N).reshape(M, N)
            b = a.T.contiguous()
            print("A:\\n", a)
            print("B=A.T:\\n", b)

            for m in range(M):
                for n in range(N):
                    assert a[m, n] == b[n, m]
            """
        ),
        markdown(
            """
            ## 2. Why a direct transpose has one bad global direction

            A warp obtains coalesced global access when adjacent lanes touch
            adjacent words. Suppose lanes 0-31 read one input row:

            ```text
            A[m, n + lane] -> consecutive addresses
            ```

            Writing those values directly to their transposed positions gives:

            ```text
            B[n + lane, m] -> addresses separated by output row stride M
            ```

            Reads are coalesced, writes are strided. Reversing thread ownership
            makes writes coalesced but reads strided. Layout algebra makes the
            tradeoff concise, but it cannot remove the physical fact that one
            global direction is a column walk.

            Shared memory breaks this dependency: load with one thread mapping,
            synchronize, then read the same tile with a transposed thread
            mapping before a coalesced output store.
            """
        ),
        code(
            """
            # Address deltas for one warp in a 1024-column row-major matrix.
            width = 1024
            input_offsets = [lane for lane in range(32)]
            direct_output_offsets = [lane * width for lane in range(32)]
            print("coalesced read deltas:", input_offsets[:8])
            print("strided write deltas:", direct_output_offsets[:8])
            """
        ),
        markdown(
            """
            ## 3. Challenge A: implement both naive mappings

            The first challenge makes the tradeoff concrete. Both kernels use
            `32 x 32` input tiles, 256 threads interpreted as `(8,32)`, and
            four values per thread.

            **Coalesced-read ownership**

            ```text
            lane varies global_col
            load  A[global_row, global_col]   contiguous
            store B[global_col, global_row]   strided
            ```

            **Coalesced-write ownership**

            ```text
            lane varies output_col
            load  A[output_col, output_row]   strided
            store B[output_row, output_col]   contiguous
            ```

            Fill the five TODOs below. Notice that no thread mapping can make
            both sides contiguous: the direct assignment couples the input and
            output ownership.

            The reference uses `cute.copy` by first making both operands share
            one logical coordinate domain:

            ```text
            coalesced read:  mBT[m,n] aliases row-major mB[n,m]
                             mBT shape/stride = (M,N):(1,M)

            coalesced write: mAT[n,m] aliases row-major mA[m,n]
                             mAT shape/stride = (N,M):(1,N)
            ```

            The read variant partitions `mA` and `mBT`; the write variant
            partitions `mAT` and `mB`. Matching logical profiles let both use
            the same predicated `cute.copy` pattern while the physical strides
            determine which global direction is coalesced.
            """
        ),
        code(challenge_a_user_source()),
        markdown(
            """
            Test both mappings on a divisible shape and an irregular shape.
            The irregular case proves that the predicates follow the selected
            mapping rather than assuming complete tiles.
            """
        ),
        code(
            """
            def check_naive(transpose_fn, m, n):
                a = torch.randn(m, n, device="cuda", dtype=torch.float32)
                b = torch.empty(n, m, device="cuda", dtype=torch.float32)
                a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
                b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
                compiled = cute.compile(transpose_fn, a_, b_)
                compiled(a_, b_)
                torch.cuda.synchronize()
                torch.testing.assert_close(b, a.T)


            for fn in (
                naive_transpose_coalesced_read,
                naive_transpose_coalesced_write,
            ):
                check_naive(fn, 64, 96)
                check_naive(fn, 1003, 1507)
            print("both naive mappings are correct")
            """
        ),
        markdown(
            """
            ### Benchmark Challenge A

            Benchmark both direct mappings on the same large, regular matrix.
            Effective bandwidth counts one required Float32 input read and one
            output write:

            ```text
            traffic bytes = 2 * M * N * sizeof(Float32)
            GB/s = traffic bytes / (average_us * 1000)
            ```

            Compile outside the timing loop, perform warmup launches, and use
            `testing.benchmark` so compilation and Python setup are excluded.
            The reported H100 percentage uses the theoretical 3,350 GB/s SXM
            specification; it is not another measured value.
            """
        ),
        code(
            """
            H100_SXM_PEAK_GBPS = 3_350.0


            def benchmark_naive(
                name,
                transpose_fn,
                a,
                b,
                warmup=10,
                iterations=100,
            ):
                a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
                b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
                compiled = cute.compile(transpose_fn, a_, b_)

                compiled(a_, b_)
                torch.cuda.synchronize()
                torch.testing.assert_close(b, a.T)

                average_us = testing.benchmark(
                    compiled,
                    kernel_arguments=testing.JitArguments(a_, b_),
                    warmup_iterations=warmup,
                    iterations=iterations,
                )
                traffic_bytes = 2 * a.numel() * a.element_size()
                bandwidth_gbps = traffic_bytes / (average_us * 1_000)
                peak_percent = 100 * bandwidth_gbps / H100_SXM_PEAK_GBPS
                print(
                    f"{name:24s} {average_us:9.3f} us  "
                    f"{bandwidth_gbps:9.2f} GB/s  "
                    f"{peak_percent:6.2f}% of H100 SXM peak"
                )


            M, N = 8192, 8192
            benchmark_a = torch.randn(
                M, N, device="cuda", dtype=torch.float32
            )
            benchmark_b = torch.empty(
                N, M, device="cuda", dtype=torch.float32
            )

            benchmark_naive(
                "naive/coalesced-read",
                naive_transpose_coalesced_read,
                benchmark_a,
                benchmark_b,
            )
            benchmark_naive(
                "naive/coalesced-write",
                naive_transpose_coalesced_write,
                benchmark_a,
                benchmark_b,
            )
            """
        ),
        markdown(
            """
            ## 4. Challenge B starts with the same thread layout

            One CTA contains 256 threads. Interpret the linear ID as:

            ```python
            lane_col = tidx % 32
            lane_row = tidx // 32
            ```

            This creates 8 logical thread rows and 32 columns. Every thread
            handles four matrix rows:

            ```text
            local_row = lane_row + value * 8, value in [0,4)
            local_col = lane_col
            ```

            Hence `256 threads * 4 values = 1024`, exactly one `32 x 32` tile.
            Within each load iteration, a warp has fixed `local_row` while its
            lanes vary `local_col`, producing a coalesced input row load.
            """
        ),
        code(
            """
            owners = {}
            for tid in range(256):
                lane_col = tid % 32
                lane_row = tid // 32
                for value in range(4):
                    coord = (lane_row + value * 8, lane_col)
                    assert coord not in owners
                    owners[coord] = tid

            assert len(owners) == 32 * 32
            print("thread 0 owns:", [coord for coord, tid in owners.items() if tid == 0])
            print("thread 31 owns:", [coord for coord, tid in owners.items() if tid == 31])
            """
        ),
        markdown(
            """
            ## 5. Shared-memory progression

            This section is deliberately incremental. Each subsection ends in
            a focused challenge before the final CuTe kernel combines all four
            ideas.

            ### 5.1 Shared staging makes both global accesses coalesced

            Challenge B reuses Challenge A's layout workflow instead of
            calculating scalar coordinates manually. The load phase partitions
            tensors that share the input-tile domain:

            ```python
            tAgA = cute.local_partition(tile_a, thread_layout, tidx)
            tAsA = cute.local_partition(sA, thread_layout, tidx)
            cute.copy(copy_atom, tAgA, rA, pred=pred_a)
            cute.copy(copy_atom, rA, tAsA, pred=pred_a)
            ```

            `sAT` is a second layout over the same shared pointer:

            ```python
            sAT = cute.make_tensor(sA.iterator, smem_layout_t)
            ```

            After the barrier, partition `sAT` and the output tile in their
            common output domain and copy again. Global loads and stores are
            both coalesced; the transposed address mapping exists entirely in
            the shared layout alias.

            The CTA barrier is mandatory: without it, one warp may read a
            transposed shared location before another warp has written it.
            """
        ),
        markdown(
            """
            #### Challenge 5.1: naive shared-memory transpose

            Your completed unpadded CuTe kernel is preserved below. It:

            1. Select the input/output CTA tiles.
            2. Allocate `sA` and create its transposed alias `sAT`.
            3. Partition and copy global A to shared memory.
            4. Synchronize the CTA.
            5. Partition and copy shared memory to global B.
            6. Build and launch the common tiled kernel.
            7. Supply the unpadded `(32,1)` / `(1,32)` layouts.

            This version is correct but intentionally retains the stride-32
            bank conflict that Day 06 measures with NCU.
            """
        ),
        code(
            shared_naive_puzzle_scaffold()
            + '\n\ncheck_transpose_correctness(\n'
            + '    "Challenge 5.1 shared/unpadded", transpose_unpadded\n'
            + ")\n"
            + "benchmark_transpose_bandwidth(\n"
            + '    "Challenge 5.1 shared/unpadded", transpose_unpadded\n'
            + ")\n"
        ),
        markdown(
            """
            ### 5.2 Padding changes addresses, not logical shape

            Keep the logical tensor shape `(32,32)` but use physical strides
            `(33,1)`:

            ```python
            smem_layout = cute.make_layout((32, 32), stride=(33, 1))
            ```

            The padded word between logical rows is not part of the matrix. It
            shifts each next row by one bank:

            ```text
            bank(lane * 33 + fixed_column)
              = (lane + fixed_column) mod 32
            ```

            The warp now visits all 32 banks once. CuTe's compact extent ends
            at the final logical element, so this layout adds 31 Float32 words,
            only 124 bytes per CTA.
            """
        ),
        markdown(
            """
            #### Challenge 5.2: derive the padded layout

            Derive the physical row stride and compact allocation extent. The
            logical shape must stay `32 x 32`, every lane must land in a
            different bank, and the allocation must grow by exactly 124 bytes.
            Your completed CuTe extension then launches the same kernel with
            the padded layout pair.
            """
        ),
        code(
            """
            def banks_for_column(row_stride, column=0):
                return [
                    (lane * row_stride + column) % 32
                    for lane in range(32)
                ]


            def padded_layout_model(tile_rows=32, tile_cols=32):
                return 33, (tile_rows - 1) * 33 + tile_cols


            padded_stride, allocated_words = padded_layout_model()
            padded = banks_for_column(padded_stride)
            print("padded banks:", padded)
            assert sorted(padded) == list(range(32))

            logical_bytes = 32 * 32 * 4
            allocated_bytes = allocated_words * 4
            assert allocated_bytes - logical_bytes == 124
            print(
                f"logical tile={logical_bytes} B, "
                f"padded allocation={allocated_bytes} B"
            )
            print("Challenge 5.2 padding model: correctness OK")
            """
        ),
        code(
            padded_puzzle_scaffold()
            + '\n\ncheck_transpose_correctness(\n'
            + '    "Challenge 5.2 shared/padded", transpose_padded\n'
            + ")\n"
            + "benchmark_transpose_bandwidth(\n"
            + '    "Challenge 5.2 shared/padded", transpose_padded\n'
            + ")\n"
        ),
        markdown(
            """
            ### 5.3 Swizzling changes bank bits without padding

            Padding changes the affine stride and slightly increases shared
            storage. A swizzle instead XORs selected address bits while
            preserving allocation size:

            ```python
            swizzle = cute.make_swizzle(5, 0, 5)
            smem_layout = cute.make_composed_layout(
                swizzle,
                0,
                cute.make_layout((32, 32), stride=(32, 1)),
            )
            ```

            CuTe layouts use element offsets. In a 32-element row, bits 0-4
            select the column and bits 5-9 select the row. `S<5,0,5>` XORs
            those row bits into the five bank-selecting low bits.
            """
        ),
        markdown(
            """
            #### Challenge 5.3: derive `S<5,0,5>`

            Implement the equivalent offset transformation. Verify that the
            32 addresses in one column remain unique and visit all 32 banks.
            Your completed CuTe extension composes `S<5,0,5>` with both
            unpadded layouts and launches the common kernel.
            """
        ),
        code(
            """
            def swizzle_5_0_5(offset):
                right = (offset >> 5) & 0x1F
                return offset ^ right


            column_offsets = [lane * 32 for lane in range(32)]
            swizzled_offsets = [swizzle_5_0_5(x) for x in column_offsets]
            swizzled_banks = [x % 32 for x in swizzled_offsets]
            assert len(set(swizzled_offsets)) == 32
            assert sorted(swizzled_banks) == list(range(32))
            print("swizzled banks:", swizzled_banks)
            print("Challenge 5.3 swizzle model: correctness OK")
            """
        ),
        code(
            swizzled_puzzle_scaffold()
            + '\n\ncheck_transpose_correctness(\n'
            + '    "Challenge 5.3 shared/swizzled", transpose_swizzled\n'
            + ")\n"
            + "benchmark_transpose_bandwidth(\n"
            + '    "Challenge 5.3 shared/swizzled", transpose_swizzled\n'
            + ")\n"
        ),
        markdown(
            """
            ## 6. Independent predicates for input and output

            The launch uses ceiling division, so edge CTAs contain logical
            coordinates outside the matrices. The input phase checks:

            ```text
            (global_row, global_col) < (M,N)
            ```

            The output phase uses a different thread-to-coordinate mapping and
            checks:

            ```text
            (output_row, output_col) < (N,M)
            ```

            These predicates are mathematically equivalent under transpose,
            but they belong to different thread iterations. Reusing the input
            lane predicate for the output access is wrong because the output
            thread partitions `sAT`, not the original `sA` view.

            As in Day 04, `cute.elem_less` performs componentwise coordinate
            comparison and combines the result.
            """
        ),
        code(
            """
            def tile_grid(m, n, tile=32):
                return ((n + tile - 1) // tile, (m + tile - 1) // tile)


            assert tile_grid(1003, 1507) == (48, 32)
            print("irregular-shape grid (x,y):", tile_grid(1003, 1507))
            """
        ),
        markdown(
            """
            ## 7. Benchmark all five variants

            Transpose performs one required input read and one output write:

            ```text
            bytes = 2 * M * N * sizeof(Float32)
            GB/s  = bytes / (average_us * 1000)
            ```

            This is effective algorithmic bandwidth. It does not count shared
            traffic because the metric asks how quickly the required global
            operation completes.

            Compile each variant once and benchmark the compiled callable on
            the same large, regular shape. The two naive results show the cost
            of choosing one coalesced global direction; the padded result shows
            the benefit of decoupling ownership through shared memory. The
            three shared variants isolate shared-memory address mapping.

            The H100 SXM specification is 3,350 GB/s. `peak efficiency` below
            compares measured effective bandwidth with that theoretical value;
            it is not a second measurement.
            """
        ),
        code(
            """
            H100_SXM_PEAK_GBPS = 3_350.0


            def benchmark_transpose(
                name,
                transpose_fn,
                a,
                b,
                warmup=10,
                iterations=100,
            ):
                a_ = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
                b_ = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
                compiled = cute.compile(transpose_fn, a_, b_)
                compiled(a_, b_)
                torch.cuda.synchronize()
                torch.testing.assert_close(b, a.T)

                average_us = testing.benchmark(
                    compiled,
                    kernel_arguments=testing.JitArguments(a_, b_),
                    warmup_iterations=warmup,
                    iterations=iterations,
                )
                # One required HBM read plus one required HBM write.
                traffic_bytes = 2 * a.numel() * a.element_size()
                bandwidth_gbps = traffic_bytes / (average_us * 1_000)
                peak_efficiency = 100 * bandwidth_gbps / H100_SXM_PEAK_GBPS
                print(
                    f"{name:24s} {average_us:9.3f} us  "
                    f"{bandwidth_gbps:9.2f} GB/s  "
                    f"{peak_efficiency:6.2f}% of H100 SXM peak"
                )
                return average_us, bandwidth_gbps


            M, N = 8192, 8192
            a = torch.randn(M, N, device="cuda", dtype=torch.float32)
            b = torch.empty(N, M, device="cuda", dtype=torch.float32)

            variants = {
                "naive/coalesced-read": naive_transpose_coalesced_read,
                "naive/coalesced-write": naive_transpose_coalesced_write,
                "shared/unpadded": transpose_unpadded,
                "shared/padded": transpose_padded,
                "shared/swizzled": transpose_swizzled,
            }
            for name, transpose_fn in variants.items():
                benchmark_transpose(name, transpose_fn, a, b)
            """
        ),
        markdown(
            """
            ### Measure a practical HBM copy ceiling

            The 3,350 GB/s specification is theoretical. The code below measures
            a device-to-device copy baseline on the same GPU. The `8192 x 8192`
            Float32 buffers are each 256 MiB, much larger than H100's L2 cache,
            so repeated copies must move data through HBM.

            CUDA events measure GPU time without including Python launch setup.
            Effective copy traffic also counts one source read and one
            destination write. The transpose cannot necessarily equal this
            baseline because it performs address permutation and shared-memory
            work.
            """
        ),
        code(
            """
            def benchmark_hbm_copy(src, dst, warmup=10, iterations=100):
                for _ in range(warmup):
                    dst.copy_(src)
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    dst.copy_(src)
                end.record()
                end.synchronize()

                total_ms = start.elapsed_time(end)
                average_us = total_ms * 1_000 / iterations
                traffic_bytes = (
                    2 * src.numel() * src.element_size() * iterations
                )
                bandwidth_gbps = traffic_bytes / (total_ms / 1_000) / 1e9
                peak_efficiency = 100 * bandwidth_gbps / H100_SXM_PEAK_GBPS
                print(
                    f"HBM copy baseline        {average_us:9.3f} us  "
                    f"{bandwidth_gbps:9.2f} GB/s  "
                    f"{peak_efficiency:6.2f}% of H100 SXM peak"
                )
                return average_us, bandwidth_gbps


            copy_src = torch.randn(8192, 8192, device="cuda", dtype=torch.float32)
            copy_dst = torch.empty_like(copy_src)
            benchmark_hbm_copy(copy_src, copy_dst)
            """
        ),
        markdown(
            """
            ## 8. Debugging guide

            | Symptom | Likely cause |
            | --- | --- |
            | Output equals input blocks | `sAT` does not use the transposed layout |
            | Correct square, wrong rectangle | Block origins were not swapped |
            | Edge contains garbage | Missing or mismatched output predicate |
            | Nondeterministic values | Missing CTA barrier |
            | Correct but unexpectedly slow | Shared mapping conflicts or global access is strided |
            | Layout allocation error | Padded layout's physical cosize was not respected |

            A transpose has three coordinate systems at once: input global,
            shared tile, and output global. Name coordinates by space instead
            of reusing ambiguous `x` and `y` variables.
            """
        ),
        markdown(
            """
            ## What you should know afterward

            - Transpose is a coordinate permutation expressed through layouts.
            - Direct coalesced-read and coalesced-write mappings move, but do
              not remove, the strided global access.
            - Shared memory decouples the input and output thread mappings.
            - A CTA barrier separates tile production from consumption.
            - Transposed shared access creates a 32-way conflict for stride 32.
            - Padding to stride 33 distributes a Float32 column across 32 banks.
            - `S<5,0,5>` swizzling removes the conflict without padding.
            - Input and output edge predicates must follow their own mappings.
            - Swizzling can replace padding, but its bit mapping must be derived
              and profiled rather than guessed.

            Compare every stage with `solution.py` only after the divisible and
            irregular checks pass.
            """
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
    destination = DAY_ROOT / "day05.ipynb"
    destination.write_text(
        json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
