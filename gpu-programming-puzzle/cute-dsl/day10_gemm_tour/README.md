# Day 10 - GEMM tour from SIMT to H100 and B300

**Goal:** assemble the abstractions from Days 01–07 into an end-to-end GEMM
and learn how the same FP16 dataflow evolves toward H100 TMA/WGMMA and
B300/SM103 `tcgen05.mma`.

This lesson mirrors the staged teaching style of NVIDIA's
[`12_tour_to_sol_gemm.ipynb`](../notebooks/12_tour_to_sol_gemm.ipynb), but not
its Blackwell implementation. The runnable notebook develops three standalone
FP16-input, FP32-accumulation SIMT kernels, then adds a Day 21-style FP32
`TiledCopy`/`TiledMma` bridge before the architecture-specific tour:

Start with [`day10.ipynb`](day10.ipynb). It derives the GEMM work and memory
convention `A[M,N] @ B[N,K] -> C[M,K]`, then compares:

- one thread per output with direct global-memory reads;
- 32x32 CTA tiling through shared memory;
- 128x128 CTA tiling with CuTe `(32,8)` copy layouts, a `(16,16)` compute
  layout, and an 8x8 register tile per thread, following
  [`20_cute_sgemm_1.cu`](../../cuda-cpp/solutions/20_cute_sgemm_1.cu).
- the same 128x128x8 CTA structure packaged as 128-bit `TiledCopy` objects and
  a `(16,16,1)` `TiledMma`, with global-to-register prefetch, following
  [`21_cute_sgemm_2.cu`](../../cuda-cpp/challenge/21_cute_sgemm_2.cu).

The later sections explain two hardware-specific optimization paths:

- H100/SM90: FP32 accumulation, TMA descriptors, WGMMA, staged pipelines,
  warp specialization, and optimized stores.
- B300/SM103: `tcgen05.mma`, TMEM accumulators, TMA/UMMA pipelines, subtiled
  FP16 epilogues, and one- versus two-CTA groups.

The architecture sections are a staged tour linked to the complete tracked
C++ references. Each standalone implementation performs its own correctness
check and CUDA-event benchmark before the notebook transitions to the FP16
tensor-core output contract.

## The tour

| Stage | Question answered | CuTe mechanism |
| --- | --- | --- |
| 1. Problem | Which logical tensors implement `C = A @ B`? | Shape and stride |
| 2. CTA tiling | Which M/K output region belongs to this block? | Grid coordinates and CuTe layouts |
| 3. Movement | How do threads cooperatively load A and B? | Copy thread layout, `local_partition` |
| 4. Compute | Which output region belongs to each thread? | Projected compute layout, register tensor |
| 5. Packaged ownership | How do copy and MMA atoms encode the same mapping? | `TiledCopy`, `TiledMma`, register fragments |
| 6. Mainloop | How is reduction over N expressed? | Shared-memory tiles, scalar FMA, or `cute.gemm` |
| 7. Epilogue | How do accumulators become output values? | register-to-global copy |
| 8. Evaluation | Is it correct, and how fast is it? | PyTorch reference, CUDA events |

## Kernel dataflow

```text
global A/B
    │ copy-layout local partitions
    ▼
shared-memory A/B tiles
    │ projected compute-layout partitions
    ▼
register fragments
    │ scalar FP32 FMA or TiledMma over N tiles
    ▼
register accumulator
    │ epilogue copy
    ▼
global C
```

The solution is intentionally **single-stage**: it waits for each N tile
before computing it. Day 07 taught the ring-buffer protocol needed to overlap
future loads with current compute; production kernels combine that protocol
with the dataflow above.

## Optional class-based exercise

`puzzle.py` and `solution.py` retain the earlier FP32 class-based exercise for
learners who want to practice `local_tile`, `TiledCopy`, and `TiledMma`
separately from the notebook's standalone progression. Fill three connected
blocks:

1. Slice global tensors into per-CTA tiles.
2. Allocate and partition shared memory, copies, and MMA fragments.
3. Implement the K mainloop and epilogue.

```bash
python puzzle.py --M 256 --N 256 --K 64
python solution.py --M 256 --N 256 --K 64
python solution.py --M 1024 --N 1024 --K 256 --iterations 100
```

For this optional exercise, `M` and `N` must be multiples of 64 and `K` must
be a multiple of 8.

## From this kernel to H100 and B300

The algorithmic roles line up even though the hardware mechanisms differ:

| Standalone warp-tiled/TiledMma kernel | H100 / SM90 | B300 / SM103 |
| --- | --- | --- |
| cooperative scalar shared-memory loads | TMA | TMA |
| FP32 SIMT FMA | FP16 WGMMA, FP32 accumulation | FP16 `tcgen05.mma`, FP32 accumulation |
| register accumulator | register accumulator | TMEM accumulator |
| CTA barriers | `PipelineTmaAsync` | `PipelineTmaUmma` |
| direct output copy | vectorized FP16 epilogue | subtiled TMEM-to-FP16 epilogue |

The H100 pod can run the baseline and SM90 experiments. B300 has compute
capability 10.3 and requires an SM103 CUDA/CuTe DSL toolchain; the notebook
therefore treats the checked-in SM100 Notebook 12 as a conceptual/API bridge
and clearly marks B300-only execution steps.

WGMMA is an SM90a-specific implementation path, not the B300 compatibility
path. Compile a separate SM103 `tcgen05.mma` kernel for B300, even if both
kernels expose the same host-side callable interface.
