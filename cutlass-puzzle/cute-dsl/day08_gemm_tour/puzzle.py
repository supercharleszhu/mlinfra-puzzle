#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 08 - CuTe DSL tour to a single-stage SIMT GEMM (puzzle).

Three TODOs in the kernel:

  (A) Tile the global tensors into per-CTA views.
  (B) Allocate shared memory, partition the copies, partition the MMA.
  (C) The mainloop: gmem->smem, smem->rmem, mma, repeat.

The host-side setup (smem layouts, tiled copies, tiled MMA, launch) is filled
in for you.
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

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        # smem layouts: M-major / N-major so MMA can vectorize smem->rmem loads.
        sA_layout = cute.make_layout((self.bM, self.bK), stride=(1, self.bM))
        sB_layout = cute.make_layout((self.bN, self.bK), stride=(1, self.bN))

        # gmem -> smem: 128b vector cp.async on M (for A) and N (for B).
        num_vec: cutlass.Constexpr = 4
        def make_g2s(tiler_lead, dtype):
            t = cute.make_layout((tiler_lead, self.num_threads // tiler_lead),
                                  stride=(1, tiler_lead))
            v = cute.make_layout((num_vec, 1))
            atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(),
                                        dtype,
                                        num_bits_per_copy=dtype.width * num_vec)
            return cute.make_tiled_copy_tv(atom, t, v)

        tiled_copy_A = make_g2s(self.bM // num_vec, mA.element_type)
        tiled_copy_B = make_g2s(self.bN // num_vec, mB.element_type)

        # Tiled MMA: 1x1x1 universal FMA atoms laid out as (T/16, 16, 1).
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        atoms_layout = cute.make_layout((self.num_threads // 16, 16, 1),
                                         stride=(16, 1, 0))
        permutation = (cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1)),
                       cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1)),
                       None)
        tiled_mma = cute.make_tiled_mma(op, atoms_layout, permutation_mnk=permutation)

        grid = (*cute.ceil_div(mC.shape, (self.bM, self.bN)), 1)
        self.kernel(mA, mB, mC, sA_layout, sB_layout,
                    tiled_copy_A, tiled_copy_B, tiled_mma).launch(
            grid=grid, block=(cute.size(atoms_layout), 1, 1),
        )

    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
               sA_layout: cute.Layout, sB_layout: cute.Layout,
               tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy,
               tiled_mma: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        thr_mma = tiled_mma.get_slice(tidx)

        # ----- TODO (A): slice global tensors into per-CTA tiles -----
        # Use cute.local_tile.
        #   gA  shape:  (BLK_M, BLK_K, k)   proj=(1,    None, 1)
        #   gB  shape:  (BLK_N, BLK_K, k)   proj=(None, 1,    1)
        #   gC  shape:  (BLK_M, BLK_N)      proj=(1,    1,    None)
        #
        # coord = (bidx, bidy, None)
        #
        # gA = cute.local_tile(mA, tiler=self.cta_tiler, coord=coord, proj=(1, None, 1))
        # gB = cute.local_tile(mB, tiler=self.cta_tiler, coord=coord, proj=(None, 1, 1))
        # gC = cute.local_tile(mC, tiler=self.cta_tiler, coord=coord, proj=(1, 1, None))
        raise NotImplementedError("Day 08 (A): slice gmem tensors with local_tile")

        # ----- TODO (B): smem allocation + per-thread partitioning -----
        # smem = cutlass.utils.SmemAllocator()
        # sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        # sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        #
        # thr_copy_A = tiled_copy_A.get_slice(tidx)
        # thr_copy_B = tiled_copy_B.get_slice(tidx)
        # tAgA = thr_copy_A.partition_S(gA)
        # tAsA = thr_copy_A.partition_D(sA)
        # tBgB = thr_copy_B.partition_S(gB)
        # tBsB = thr_copy_B.partition_D(sB)
        #
        # tCsA = thr_mma.partition_A(sA)
        # tCsB = thr_mma.partition_B(sB)
        # tCgC = thr_mma.partition_C(gC)
        # tCrA = tiled_mma.make_fragment_A(tCsA)
        # tCrB = tiled_mma.make_fragment_B(tCsB)
        # tCrC = tiled_mma.make_fragment_C(tCgC)
        # tCrC.fill(0.0)

        # ----- TODO (C): mainloop (synchronous cp.async per K-tile) -----
        # k_tile_count = cute.size(tAgA, mode=[3])
        # k_block_max  = cute.size(tCrA, mode=[2])
        # for k_tile in range(k_tile_count):     # DSL traces this as dynamic
        #     cute.copy(tiled_copy_A,
        #               tAgA[None, None, None, k_tile],
        #               tAsA[None, None, None])
        #     cute.copy(tiled_copy_B,
        #               tBgB[None, None, None, k_tile],
        #               tBsB[None, None, None])
        #     cute.arch.cp_async_commit_group()
        #     cute.arch.cp_async_wait_group(0)
        #     cute.arch.barrier()
        #
        #     for k_block in cutlass.range_constexpr(k_block_max):
        #         cute.autovec_copy(tCsA[None, None, k_block], tCrA[None, None, k_block])
        #         cute.autovec_copy(tCsB[None, None, k_block], tCrB[None, None, k_block])
        #         cute.gemm(tiled_mma, tCrC,
        #                   tCrA[None, None, k_block],
        #                   tCrB[None, None, k_block],
        #                   tCrC)
        #     cute.arch.barrier()
        #
        # # Epilogue: copy accumulator to gmem (unvectorized).
        # atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        # cute.copy(atom_store, tCrC, tCgC)


def run(M: int, N: int, K: int) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Day 08 needs a CUDA device.")
    cutlass.cuda.initialize_cuda_context()

    a = torch.empty(K, M, dtype=torch.float32).normal_().permute(1, 0).cuda()
    b = torch.empty(K, N, dtype=torch.float32).normal_().permute(1, 0).cuda()
    c = torch.empty(N, M, dtype=torch.float32).zero_().permute(1, 0).cuda()

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    print(f"\n=== Day 08 SIMT GEMM, M={M} N={N} K={K} ===")
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
    run(a.M, a.N, a.K)
    print("\nSuccess.")
