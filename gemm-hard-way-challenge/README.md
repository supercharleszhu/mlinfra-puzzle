# GEMM the Hard Way Standalone Challenge

This folder is a standalone exercise path for learning GEMM and fused GPU
kernels in four chapters:

1. Days 1-12 build a CUDA GEMM from naive FP32 through asynchronous Tensor Cores.
2. Days 13-18 form Chapter 2: handwritten Hopper optimization without CUTLASS, based on fast.cu-style TMA/WGMMA kernels. https://www.aleksagordic.com/blog/matmul
3. Day 19 forms Chapter 3: CUTLASS 3.x Hopper configuration and tuning, following https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way-2/
4. Days 20-28 form Chapter 4: low-level CuTe GEMM tutorials, progressing from
   layouts and tiled operations to Hopper WGMMA/TMA and Blackwell UMMA/TMEM.

References:

- CUTLASS intro blog: https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/
- CUTLASS Hopper blog: https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way-2/
- Inside nvidia gpu: https://www.aleksagordic.com/blog/matmul
- explore-gemm source: https://github.com/gpusgobrr/explore-gemm/tree/main/cuda
- CUTLASS official SM90 kernel headers: https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel
- Hopper fast.cu-style blog notes: BF16 H100 matmul path with WGMMA, TMA, persistent scheduling, clusters, and L2-aware scheduling.

CUDA files `01` through `28` live under `cuda/`. Days 20-28 are flat, complete
C++ sources from NVIDIA's `examples/cute/tutorial` tree. The source-level
blanks sit directly in CuTe layouts, tilers, copy atoms, MMA atoms, pipelines,
clusters, and epilogues. Python is orchestration only: it compiles the selected
local `.cu` file with NVCC and never imports an example or kernel.

## Day map

### Days 1-12: CUDA GEMM foundations

| Path | Purpose |
| --- | --- |
| `cuda/01_naive.cu` | Naive FP32 GEMM baseline. |
| `cuda/02_kernel_global_mem_coalesce.cu` | Coalesced thread-to-output mapping. |
| `cuda/03_kernel_shared_mem.cu` | Shared-memory A/B tiling. |
| `cuda/04_kernel_blocktiling_1d.cu` | 1D register tiling. |
| `cuda/05_kernel_blocktiling_2d.cu` | 2D per-thread register tile. |
| `cuda/06_kernel_vectorize.cu` | Vectorized `float4` loads and SMEM layout. |
| `cuda/07_kernel_warptiling.cu` | CTA/warp/thread tiling hierarchy. |
| `cuda/08_kernel_warptiling_all_dtypes.cu` | FP32/FP16/BF16 warp tiling. |
| `cuda/09_kernel_tensorcore_naive.cu` | Naive WMMA Tensor Core baseline. |
| `cuda/10_kernel_tensorcore_warptiled.cu` | WMMA plus block/warp tiling. |
| `cuda/11_kernel_tensorcore_double_buffered.cu` | Tensor Core double buffering. |
| `cuda/12_kernel_tensorcore_async.cu` | Async pipeline variant. |

### Days 13-18: Chapter 2 - handwritten Hopper optimization

These days avoid CUTLASS and focus on the mechanics hidden by libraries: TMA tensor maps, barriers, WGMMA descriptors/instructions, cached tensor maps, TMA store, and Hilbert scheduling.

| Path | Handwritten Hopper focus |
| --- | --- |
| `cuda/13_kernel_fastcu_matmul2_manual_tma_wgmma.cu` | fast.cu matmul_2 analogue: manually apply 2D TMA loads, barriers, WGMMA m64n64k16, and C^T stores. |
| `cuda/14_kernel_fastcu_matmul3_big_tile.cu` | fast.cu matmul_3: larger BM/BN output tiles and wider WGMMA_N shapes. |
| `cuda/15_kernel_fastcu_matmul4_warp_specialized.cu` | fast.cu matmul_4: warp-specialized producer/consumer overlap for TMA loads and WGMMA compute. |
| `cuda/16_kernel_fastcu_matmul6_persistent.cu` | fast.cu matmul_6: persistent tile processing and steady-state overlap. |
| `cuda/17_kernel_fastcu_matmul10_tma_store.cu` | fast.cu matmul_10: accumulator conversion, stmatrix staging, and TMA store. |
| `cuda/18_kernel_fastcu_matmul12_final.cu` | fast.cu matmul_12: final combined handwritten kernel and benchmark analysis. |
| `LICENSE.fastcu` | MIT license for the fast.cu-derived source. |

The handwritten fast.cu ports use fast.cu's B^T/C^T layout convention: the Python wrapper passes `B.t().contiguous()` to the kernel and returns a transposed view of the output buffer.

### Day 19: Chapter 3 - CUTLASS 3 Hopper

The CUTLASS chapter intentionally skips the CUTLASS 2.x baseline. One tunable
SM90 `CollectiveBuilder` lesson covers the Hopper blog's full progression:
TMA/WGMMA warp specialization, stage count, thread-block clusters, persistent
cooperative and ping-pong schedules, Stream-K, rasterization, swizzle, and
decomposition tuning.

| Path | CUTLASS focus |
| --- | --- |
| `cuda/19_kernel_cutlass3_hopper_tunable.cu` | CUTLASS 3 SM90 kernel with blanks for schedule, stages, tile/cluster shape, raster order, decomposition, swizzle, and splits. |

### Days 20-28: Chapter 4 - CuTe GEMM from layouts to Blackwell

These lessons use CUTLASS v4.6.1 at commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3`. They follow NVIDIA's
[`examples/cute/tutorial`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial)
and the [CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/cute/0x_gemm_tutorial.html).
Each lesson contains the complete tutorial `.cu`; the pinned checkout supplies
CuTe/CUTLASS headers and `example_utils.hpp`. The runner compiles the selected
local challenge or solution directly. Hopper-only lessons require SM90 and
Blackwell lessons require SM100.

| Day | Official example | Main concepts |
| --- | --- | --- |
| 20 | `cute/tutorial/sgemm_1.cu` | Tensor shapes/strides, `local_tile`, `local_partition`, and scalar CuTe `gemm`. |
| 21 | `cute/tutorial/sgemm_2.cu` | `Copy_Atom`, `TiledCopy`, `MMA_Atom`, and `TiledMMA` partitioning. |
| 22 | `cute/tutorial/sgemm_sm80.cu` | Tensor Core MMA, swizzled SMEM, and multistage `cp.async`. |
| 23 | `cute/tutorial/hopper/wgmma_sm90.cu` | Descriptor-sourced WGMMA and warpgroup synchronization. |
| 24 | `cute/tutorial/hopper/wgmma_tma_sm90.cu` | TMA partitioning and mbarrier producer/consumer pipelines. |
| 25 | `cute/tutorial/blackwell/01_mma_sm100.cu` | 1-SM UMMA, TMEM allocation, and TMEM-to-register copies. |
| 26 | `cute/tutorial/blackwell/02_mma_tma_sm100.cu` | 1-SM UMMA with TMA mainloop loads. |
| 27 | `cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu` | Peer-CTA 2-SM UMMA and multicast TMA. |
| 28 | `cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu` | 2-SM multicast mainloop plus tiled TMA load/store epilogue. |

Each challenge `.cu` file explains the algorithm and asks you to fill real
compile-time aliases in the complete implementation. Its matching
`solutions/cuda/` file is fully filled and directly compilable.

## Layout

| Path | Purpose |
| --- | --- |
| `cuda/challenge_todo.cuh` | Placeholder macros used by the exercise blanks. |
| `BLANKS.md` | Greppable TODO list grouped by file. |
| `solutions/cuda/` | Complete solution kernels copied/adapted from the upstream implementation and later Hopper exercises. |
| `solutions/python/benchmark_solution.py` | Correctness and benchmark runner for the complete solutions. |
| `cuda/20_*.cu` ... `cuda/28_*.cu` | Complete flat C++ implementations with source-level challenge blanks and explanations. |
| `solutions/cuda/20_*.cu` ... `solutions/cuda/28_*.cu` | Completed full-source advanced C++ lessons. |
| `python/benchmark_advanced_day.py` | Direct NVCC build, architecture, correctness, and benchmark runner for Days 20-28. |
| `scripts/setup_advanced_cutlass.sh` | Fetch and verify the pinned official CUTLASS v4.6.1 checkout. |
| `scripts/upload_run_advanced_day.sh` | Upload one advanced lesson and its official source dependencies to a GPU pod. |
| `scripts/upload_to_pod.sh` | Upload this standalone challenge to the configured Kubernetes pod. |
| `scripts/run_remote_day.sh` | Run correctness + benchmark for one or more days on the pod. |
| `scripts/run_remote_matrix.sh` | Convenience wrapper that uploads, then runs selected days. |
| `REMOTE_POD_GUIDE.md` | How to copy the challenge to a Kubernetes GPU pod and run correctness, benchmark, and profiling. |

## How to use

1. Open one file at a time, starting with `cuda/01_naive.cu`.
2. Search for `GEMM_TODO`.
3. Replace the placeholder with the real expression from the blog concept.
4. Run correctness against `torch.matmul`.
5. Benchmark and profile before moving to the next file.

The placeholder macros currently return dummy values so the blanks are easy to find. The exercise kernels are not expected to be correct until you replace the placeholders. Each exercise file also has a small fill-in checklist near the top so you can write down the mapping, load, compute, store, and profiling blanks for that stage.

For CUTLASS days, remember that `GemmShape<M, N, K>` still follows GEMM dimensions: `M` rows of C/A, `N` cols of C/B, and `K` reduction depth. In Day 13, `ThreadBlockShape = GemmShape<BM, BN, BK>` is the CTA tile, not the full problem size.

## Solutions

The `solutions/` folder contains complete implementations and a minimal runner:

```bash
cd gemm-hard-way-challenge
python3 solutions/python/benchmark_solution.py --sizes 128 256 --dtype float32
python3 solutions/python/benchmark_solution.py --sizes 128 256 --dtype float16
```

By default the solution runner builds files 01-12. Pass `--include-cutlass` to build the Hopper/CUTLASS chapters, days 13-19; on Hopper GPUs these build with `sm_90a`.

For remote GPU execution through Kubernetes, see [`REMOTE_POD_GUIDE.md`](./REMOTE_POD_GUIDE.md).

Quick remote examples:

```bash
cd gemm-hard-way-challenge
scripts/run_remote_matrix.sh --day 1
scripts/run_remote_matrix.sh --day 9 --sizes "128 256" --iters 50
scripts/run_remote_matrix.sh --day 14 --with-cutlass
scripts/upload_run_one_day.sh --day 15 --with-cutlass
scripts/upload_run_one_day.sh --day 18 --with-cutlass --sizes "1024 2048 4096"
scripts/upload_run_one_day.sh --day 19 --with-cutlass --sizes "1024 2048 4096"
scripts/upload_run_advanced_day.sh --day 20 --solution
scripts/upload_run_advanced_day.sh --day 23 --solution
scripts/upload_run_advanced_day.sh --day 24 --solution
scripts/upload_run_advanced_day.sh --day 25 --solution --pod "$B200_POD"
scripts/upload_run_advanced_day.sh --day 28 --solution --pod "$B200_POD"
scripts/run_remote_matrix.sh --all --skip-upload
```

For a challenge, omit `--solution`. The runner will list every remaining
`GEMM_TODO` blank before invoking NVCC. `--dry-run` validates a completed
source and prints the selected local source and helper directory.
Day 20 checks its full output against a pedantic-FP32 cuBLAS reference before
starting the timing loop and exits with an error on any mismatch.

### Code navigation for Days 20-28

Generate a compilation database containing the real NVCC architecture and
include flags for every challenge and solution:

```bash
scripts/setup_code_navigation.sh
```

When this repository is opened at its root, the local VS Code C/C++ settings
read `build/gemm-hard-way-navigation/compile_commands.json`. This enables
go-to-definition and symbol completion for CuTe and CUTLASS headers. Rerun the
script only when a lesson is added, renamed, or its compiler flags change.

## Attribution

The original explore-gemm implementation is licensed under Apache License 2.0; a copy is included as `LICENSE.upstream`.
Days 13-18 include code or exercises adapted from the MIT-licensed fast.cu repository; the license is included as `LICENSE.fastcu`.
Days 20-28 contain NVIDIA CuTe tutorial sources under CUTLASS's BSD-3-Clause
license; their license headers are preserved. Adjacent helper headers remain
in the pinned official checkout created by the setup script.
