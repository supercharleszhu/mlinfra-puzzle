#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 02 — CuTe DSL layout algebra (puzzle).

Fill in the four TODOs. No GPU kernels here — we're just learning the layout
algebra you'll lean on in every later puzzle.
"""
import argparse

import cutlass
import cutlass.cute as cute


# -----------------------------------------------------------------------------
# Part 1: row-major vs. column-major
# -----------------------------------------------------------------------------
@cute.jit
def part1_build_layouts() -> None:
    M, N = 4, 8
    row_major = cute.make_layout(shape=(4,8), stride=(8,1))
    col_major = cute.make_layout(shape=(4,8), stride=(8,1))

    # TODO(1): build two layouts of shape (M, N):
    #            - `row_major`  with strides   (N, 1)
    #            - `col_major`  with strides   (1, M)
    #          Then print both, and assert size(row_major) == M * N.
    #
    # Helpful:
    #   cute.make_layout((M, N), stride=(sM, sN))
    #   cute.size(layout)     -> Int32, the *number of elements*
    #   cute.cosize(layout)   -> Int32, one past the largest offset
    # raise NotImplementedError("Day 02 part 1")


# -----------------------------------------------------------------------------
# Part 2: zipped_divide — the bread-and-butter tiler
# -----------------------------------------------------------------------------
@cute.jit
def part2_zipped_divide() -> None:
    A = cute.make_layout((8, 16), stride=(16, 1))
    tile = (2, 4)

    # TODO(2): apply `cute.zipped_divide(A, tile)` and verify the result has
    #          shape ((2, 4), (4, 4))  — i.e. (per-tile, num-tiles).
    raise NotImplementedError("Day 02 part 2")


# -----------------------------------------------------------------------------
# Part 3: Thread-Value (TV) layout — maps (thread_id, value_id) -> tile coord
# -----------------------------------------------------------------------------
@cute.jit
def part3_tv_layout() -> None:
    # TODO(3): build:
    #   - thr_layout = (4, 32):(32, 1)  — 128 threads, contiguous on N
    #   - val_layout = (4, 4):(4, 1)    — each thread owns a 4x4 block on M
    # Then call:
    #   tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    # Print everything, assert cute.size(tv_layout) == 128 * 16.
    raise NotImplementedError("Day 02 part 3")


# -----------------------------------------------------------------------------
# Part 4: identity coordinate tensor — used for OOB predication later
# -----------------------------------------------------------------------------
@cute.jit
def part4_identity_tensor() -> None:
    # TODO(4): build an identity tensor for shape (3, 4), then print element
    #          cT[0], cT[1], cT[(2, 3)].  These should be coord tuples.
    #
    # Helpful:
    #   cute.make_identity_tensor(shape)
    raise NotImplementedError("Day 02 part 4")


def run() -> None:
    cutlass.cuda.initialize_cuda_context()
    print("\n=== part1_build_layouts ===")
    part1_build_layouts()
    print("\n=== part2_zipped_divide ===")
    part2_zipped_divide()
    print("\n=== part3_tv_layout ===")
    part3_tv_layout()
    print("\n=== part4_identity_tensor ===")
    part4_identity_tensor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
    print("\nSuccess.")
