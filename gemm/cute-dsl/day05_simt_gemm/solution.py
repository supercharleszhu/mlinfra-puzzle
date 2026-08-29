#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 05 — CuTe DSL SIMT GEMM, single-stage (solution).

A column-major SGEMM (FP32, ``C[m,n] = sum_k A[m,k] * B[n,k]``) using a
deliberately *un*-pipelined mainloop, so the code stays under ~120 lines and
the focus is on the new primitives:

  - ``cute.local_tile``            slice an MxKxL tensor into per-CTA tiles
  - ``cute.SmemAllocator``         per-CTA shared-memory arena
  - ``cute.make_tiled_mma`` + ``cute.nvgpu.MmaUniversalOp``
                                   build a tiled MMA "atom"
  - ``thr_mma.partition_A/B/C``    map smem/gmem -> per-thread fragments
  - ``cute.gemm``                  the actual MMA over (M,N,K) fragments
  - ``cute.autovec_copy``          smem -> register copy

Compared to the production-grade ``examples/python/CuTeDSL/ampere/sgemm.py``,
this version:

  - Assumes M, N, K are multiples of the tile (BLK_M, BLK_N, BLK_K). No predication.
  - Has only one shared-memory stage (no software pipelining).
  - Uses synchronous cp.async (commit + wait_group(0) every iteration).

To keep things tractable, A is column-major (M-major), B is column-major (N-major),
C is column-major (M-major). These are the easiest cases for vectorized
gmem -> smem copies.
"""
import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


class SimpleSGemm:
    def __init__(self, cta_tiler: Tuple[int, int, int] = (64, 64, 8),
                 num_threads: int = 128):
        self.cta_tiler = cta_tiler
        self.bM, self.bN, self.bK = cta_tiler
        self.num_threads = num_threads
        assert num_threads % 16 == 0
        assert self.bM % 16 == 0 and self.bN % 16 == 0

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        # ----- smem layouts: M-major / N-major to match MMA expectations -----
        sA_layout = cute.make_layout((self.bM, self.bK), stride=(1, self.bM))
        sB_layout = cute.make_layout((self.bN, self.bK), stride=(1, self.bN))

        # ----- gmem -> smem tiled copy (vectorized 128b loads of FP32) -----
        # 4 FP32 per thread = 16B = 128b vector load on the M dim (A) / N dim (B).
        num_vec: cutlass.Constexpr = 4
        major_mode_A = self.bM // num_vec
        tA = cute.make_layout(
            (major_mode_A, self.num_threads // major_mode_A),
            stride=(1, major_mode_A),
        )
        vA = cute.make_layout((num_vec, 1))
        atom_cpA = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vec,
        )
        tiled_copy_A = cute.make_tiled_copy_tv(atom_cpA, tA, vA)

        major_mode_B = self.bN // num_vec
        tB = cute.make_layout(
            (major_mode_B, self.num_threads // major_mode_B),
            stride=(1, major_mode_B),
        )
        vB = cute.make_layout((num_vec, 1))
        atom_cpB = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width * num_vec,
        )
        tiled_copy_B = cute.make_tiled_copy_tv(atom_cpB, tB, vB)

        # ----- tiled MMA: a simple universal FMA, 1x1x1 atom per thread -----
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        atoms_layout = cute.make_layout((self.num_threads // 16, 16, 1),
                                         stride=(16, 1, 0))
        permutation = (cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1)),
                       cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1)),
                       None)
        tiled_mma = cute.make_tiled_mma(op, atoms_layout,
                                         permutation_mnk=permutation)

        grid = (*cute.ceil_div(mC.shape, (self.bM, self.bN)), 1)
        self.kernel(mA, mB, mC, sA_layout, sB_layout,
                    tiled_copy_A, tiled_copy_B, tiled_mma).launch(
            grid=grid,
            block=(cute.size(atoms_layout), 1, 1),
        )

    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
               sA_layout: cute.Layout, sB_layout: cute.Layout,
               tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy,
               tiled_mma: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        thr_mma = tiled_mma.get_slice(tidx)

        # Slice the per-CTA tile out of the global tensors.
        # gA: (BLK_M, BLK_K, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)
        coord = (bidx, bidy, None)
        gA = cute.local_tile(mA, tiler=self.cta_tiler, coord=coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=self.cta_tiler, coord=coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=self.cta_tiler, coord=coord, proj=(1, 1, None))

        # Shared memory arena for A and B tiles.
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

        # Partition the tiled copy across threads.
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)   # (CPY, CPY_M, CPY_K, k)
        tAsA = thr_copy_A.partition_D(sA)   # (CPY, CPY_M, CPY_K)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        # Partition smem for the MMA + per-thread accumulator fragment.
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        k_tile_count = cute.size(tAgA, mode=[3])
        k_block_max = cute.size(tCrA, mode=[2])

        # Mainloop (no pipelining): for each K tile,
        #   gmem -> smem (cp.async),  smem -> rmem,  inner mma over k_block.
        for k_tile in range(k_tile_count):  # DSL 4.4+: traced as dynamic loop
            cute.copy(tiled_copy_A,
                      tAgA[None, None, None, k_tile],
                      tAsA[None, None, None])
            cute.copy(tiled_copy_B,
                      tBgB[None, None, None, k_tile],
                      tBsB[None, None, None])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            for k_block in cutlass.range_constexpr(k_block_max):
                cute.autovec_copy(tCsA[None, None, k_block], tCrA[None, None, k_block])
                cute.autovec_copy(tCsB[None, None, k_block], tCrB[None, None, k_block])
                cute.gemm(tiled_mma, tCrC,
                          tCrA[None, None, k_block],
                          tCrB[None, None, k_block],
                          tCrC)
            cute.arch.barrier()  # before next iter overwrites smem

        # Epilogue — just copy the accumulator out, unvectorized.
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                          mC.element_type)
        cute.copy(atom_store, tCrC, tCgC)


def run(M: int, N: int, K: int) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Day 05 needs a CUDA device.")
    cutlass.cuda.initialize_cuda_context()

    # Column-major (M-major) A: (K, M) row-major -> permute -> (M, K) col-major
    a = torch.empty(K, M, dtype=torch.float32).normal_().permute(1, 0).cuda()
    b = torch.empty(K, N, dtype=torch.float32).normal_().permute(1, 0).cuda()
    c = torch.empty(N, M, dtype=torch.float32).zero_().permute(1, 0).cuda()

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    print(f"\n=== SIMT GEMM (single-stage), M={M} N={N} K={K} ===")
    SimpleSGemm()(a_, b_, c_)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a, b)
    torch.testing.assert_close(c.cpu(), ref.cpu(), atol=1e-3, rtol=1e-4)
    print("  OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=256)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--K", type=int, default=64)
    a = p.parse_args()
    for x, name in [(a.M, "M"), (a.N, "N"), (a.K, "K")]:
        assert x % 64 == 0 or (name == "K" and x % 8 == 0), \
            f"{name}={x} must be a multiple of the tile dim (64 for M/N, 8 for K)"
    run(a.M, a.N, a.K)
    print("\nSuccess.")
