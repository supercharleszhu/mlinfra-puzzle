# Day 05 — CuTe DSL Single-Stage SIMT GEMM

**Goal:** put together everything from days 01–04 to write a real (if naive)
SGEMM: `C = A @ B^T` for FP32 column-major operands, with a real MMA atom,
real shared memory, and a real K-mainloop.

This is a *single-stage* mainloop — no software pipelining, no
producer-consumer split. That's the natural next step after day 2 (which is
the C++ multi-stage version): you get to write the same algorithm without all
the pipeline bookkeeping, and verify that it works.

## Background

A CUTLASS-style GEMM kernel has four phases:

| Phase   | What happens                                                  |
|---------|---------------------------------------------------------------|
| Prologue| Allocate smem, partition tiled-copies and tiled-MMA per thread.|
| Mainloop| For each K-tile: `gmem → smem`, `smem → rmem`, `mma`.          |
| Sync    | `barrier()` so all threads see the new smem tile.              |
| Epilogue| Write the per-thread accumulator out to `C`.                   |

The DSL exposes each of those steps as a single function call. The puzzle
gives you the *host*-side machinery (smem layouts, tiled copies, tiled MMA,
launch) and asks you to fill in the kernel body — three numbered blocks
(A) → (B) → (C).

### New primitives this puzzle introduces

| Call                                     | What it does                              |
|------------------------------------------|-------------------------------------------|
| `cute.local_tile(t, tiler, coord, proj)` | slice a per-CTA `(BLK_M, BLK_K, k)` view  |
| `cutlass.utils.SmemAllocator`            | smem arena, hands out tensors             |
| `cute.nvgpu.MmaUniversalOp(Float32)`     | a 1x1x1 FP32 FMA "atom"                   |
| `cute.make_tiled_mma(op, atoms, perm)`   | tile the atom over threads/elements       |
| `thr_mma.partition_A/B/C(sX_or_gX)`      | per-thread view of the MMA operand        |
| `tiled_mma.make_fragment_X(view)`        | register fragment matching that view      |
| `cute.gemm(mma, D, A_frg, B_frg, C)`     | the inner K-block MMA                     |
| `cute.autovec_copy(src, dst)`            | smem → rmem with auto-vectorization       |
| `cute.arch.cp_async_commit_group()`      | mark async copies as a group              |
| `cute.arch.cp_async_wait_group(0)`       | wait for all in-flight async copies       |
| `cute.arch.barrier()`                    | CTA-wide barrier                          |

### Why a single-stage mainloop?

It compiles fast, fits in your head, and is correct. It is *not* fast.
Production CUTLASS kernels prefetch 2–3 K-tiles ahead so the cp.async pipeline
keeps the SM busy while compute runs (`examples/python/CuTeDSL/ampere/sgemm.py`
shows the full version). For pedagogy, single-stage is the right starting point.

## Task

Open `puzzle.py` and fill in TODOs (A), (B), (C). Each block has hint code
in the comments — you can paste it in and tweak.

## Run

```bash
python puzzle.py --M 256 --N 256 --K 64    # raises NotImplementedError
python solution.py --M 256 --N 256 --K 64  # reference
python solution.py --M 1024 --N 1024 --K 256
```

`M`, `N` must be multiples of 64; `K` a multiple of 8 (the CTA tiler).

## What you should walk away knowing

- A GEMM kernel is *not* a giant mass of indexing code — it's four lines of
  partitioning + a five-line mainloop, once you let CuTe do the layout math.
- The MMA atom abstracts "which thread reads which register from sA / sB" —
  `partition_A` / `partition_B` is the only function that needs to know.
- `cute.gemm(mma, D, A, B, C)` is a per-thread call. The "tile-wide" effect
  emerges because every thread runs it with the right partitioned fragments.
- Once this works, the upgrade path is: (1) prefetch N K-tiles before the
  loop, (2) split warps into producer/consumer, (3) swap `MmaUniversalOp` for
  a tensor-core atom (`MmaSm80Op` on A100/H100/B200, WGMMA on H100/B200,
  tcgen05 on B200). Day 2's C++ puzzle and the official `sgemm.py` are your
  references.

## Hardware

Runs on any **sm_80+** GPU — A100, H100, B200, or recent RTX consumer cards.
No Hopper- or Blackwell-specific primitives are used.
