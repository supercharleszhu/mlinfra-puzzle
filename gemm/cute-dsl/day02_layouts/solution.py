#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Day 02 — CuTe DSL layout algebra (solution).

No kernels. We're just learning the algebra: build layouts, compose them,
divide a logical tensor into per-CTA/per-thread tiles. Each part asserts the
resulting layout has a known shape/stride so it's a self-checking puzzle.
"""
import argparse

import cutlass
import cutlass.cute as cute


@cute.jit
def part1_build_layouts() -> None:
    """Construct a row-major (M, N) layout and a column-major one, print them."""
    M, N = 4, 8

    # Row-major: stride = (N, 1) means moving 1 along M jumps N elements.
    row_major = cute.make_layout((M, N), stride=(N, 1))
    # Column-major: stride = (1, M) is the transpose.
    col_major = cute.make_layout((M, N), stride=(1, M))

    print(f"[part1] row_major  = {row_major}")
    print(f"[part1] col_major  = {col_major}")
    print(f"[part1] size       = {cute.size(row_major)}  (== M*N)")
    print(f"[part1] cosize     = {cute.cosize(row_major)}  (one-past max offset)")

    assert cute.size(row_major) == M * N
    assert cute.cosize(row_major) == M * N


@cute.jit
def part2_zipped_divide() -> None:
    """Tile a (8, 16) tensor into (2, 4) tiles → ((2,4), (4,4)).

    `zipped_divide` is the canonical CuTe way to split a tensor into
    "per-tile shape" × "number of tiles". You'll use this in every kernel.
    """
    A = cute.make_layout((8, 16), stride=(16, 1))  # row-major
    tile = (2, 4)

    divided = cute.zipped_divide(A, tile)
    # Expected: ((TileM, TileN), (RestM, RestN)) = ((2, 4), (4, 4))
    print(f"[part2] A          = {A}")
    print(f"[part2] tile       = {tile}")
    print(f"[part2] divided    = {divided}")

    assert tuple(divided.shape[0]) == (2, 4)
    assert tuple(divided.shape[1]) == (4, 4)


@cute.jit
def part3_tv_layout() -> None:
    """Build a Thread-Value layout for 32 threads × 4 values, covering a (4, 32) tile.

    Each row of the (4, 32) tile is one warp; each thread of the warp owns 4
    contiguous elements on the *row* (M) dimension after the TV math.
    """
    # 32 threads laid out as one row of the (M=4, N=32) tile, contiguous on N.
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    # Each thread owns 4 contiguous elements on M dimension, 4 on N dimension.
    val_layout = cute.make_layout((4, 4), stride=(4, 1))

    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[part3] thr_layout = {thr_layout}")
    print(f"[part3] val_layout = {val_layout}")
    print(f"[part3] tiler_mn   = {tiler_mn}   (per-CTA tile size)")
    print(f"[part3] tv_layout  = {tv_layout}  ((tid, vid) -> tile coord)")

    # 128 threads * 16 values = 2048 elements per CTA tile.
    assert cute.size(thr_layout) == 128
    assert cute.size(val_layout) == 16
    assert cute.size(tv_layout) == 128 * 16


@cute.jit
def part4_identity_tensor() -> None:
    """An identity tensor maps coord -> coord. Used for OOB predication.

    For a (3, 4) shape, the identity tensor at coord (i, j) is just (i, j).
    """
    shape = (3, 4)
    cT = cute.make_identity_tensor(shape)
    print(f"[part4] identity tensor type = {cT.type}")

    # Element 0 in the linearized order should be coord (0, 0).
    print(f"[part4] cT[0]       = {cT[0]}")
    print(f"[part4] cT[1]       = {cT[1]}")
    print(f"[part4] cT[(2, 3)]  = {cT[(2, 3)]}")


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
