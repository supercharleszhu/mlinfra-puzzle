# Day 08 - Tour to a SIMT GEMM

**Goal:** assemble the abstractions from Days 01–07 into an end-to-end GEMM
and learn how to reason about its structure, correctness, and performance.

This lesson mirrors the staged teaching style of NVIDIA's
[`12_tour_to_sol_gemm.ipynb`](../notebooks/12_tour_to_sol_gemm.ipynb), but not
its Blackwell implementation. The official notebook uses SM100-only TMA,
UMMA, and tensor memory. This challenge deliberately uses an SM80-compatible
FP32 SIMT MMA so every stage runs on the configured H100.

## The tour

| Stage | Question answered | CuTe mechanism |
| --- | --- | --- |
| 1. Problem | Which logical tensors implement `C = A @ B^T`? | Shape and stride |
| 2. CTA tiling | Which M/N/K region belongs to this block? | `local_tile` |
| 3. Movement | How do threads cooperatively load A and B? | `TiledCopy`, `cp.async` |
| 4. Compute | Which fragments belong to each thread? | `TiledMma`, `partition_A/B/C` |
| 5. Mainloop | How is reduction over K expressed? | `autovec_copy`, `cute.gemm` |
| 6. Epilogue | How do accumulators become output values? | register-to-global copy |
| 7. Evaluation | Is it correct, and how fast is it? | PyTorch reference, CUDA events |

## Kernel dataflow

```text
global A/B
    │ cooperative 128-bit cp.async
    ▼
shared-memory A/B tiles
    │ per-thread partition + autovec_copy
    ▼
register fragments
    │ repeated cute.gemm over K blocks
    ▼
register accumulator
    │ epilogue copy
    ▼
global C
```

The solution is intentionally **single-stage**: it waits for each K tile
before computing it. Day 05 taught the ring-buffer protocol needed to overlap
future loads with current compute; production kernels combine that protocol
with the dataflow above.

## Task

Fill the three connected blocks in `puzzle.py`:

1. Slice global tensors into per-CTA tiles.
2. Allocate and partition shared memory, copies, and MMA fragments.
3. Implement the K mainloop and epilogue.

```bash
python puzzle.py --M 256 --N 256 --K 64
python solution.py --M 256 --N 256 --K 64
python solution.py --M 1024 --N 1024 --K 256 --iterations 100
```

`M` and `N` must be multiples of 64; `K` must be a multiple of 8.

## From this kernel to the official SOL GEMM

The algorithmic roles line up even though the hardware mechanisms differ:

| This H100-compatible lesson | Notebook 12 on Blackwell |
| --- | --- |
| `cp.async` cooperative loads | TMA bulk tensor loads |
| FP32 `MmaUniversalOp` | UMMA tensor-core operations |
| register accumulator | tensor-memory accumulator |
| CTA barriers | transaction barriers and specialized pipelines |
| direct output copy | tiled, vectorized epilogue |

The progression is therefore conceptual rather than source-compatible. After
understanding this lesson, Notebook 12 shows how Blackwell assigns the same
roles to dedicated hardware.
