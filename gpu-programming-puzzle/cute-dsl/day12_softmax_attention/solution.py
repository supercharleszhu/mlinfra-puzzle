#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 12 - one-pass softmax attention in Triton and CuTe DSL."""

from __future__ import annotations

import argparse
import math

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.atom import Trait, make_atom
from cutlass.cute.core import _pack_shape
from cutlass.cute.runtime import from_dlpack


WARP_SIZE = 32
WARPS_PER_CTA = 4
CTA_THREADS = WARP_SIZE * WARPS_PER_CTA
QUERY_ROWS_PER_WARP = 16
QUERY_TILE = QUERY_ROWS_PER_WARP * WARPS_PER_CTA
KEY_TILE = 64


class MmaTF32CompatTrait(Trait):
    """Trait for the SM80 TF32 MMA atom missing from CuTe DSL 4.4.2."""


class MmaTF32CompatOp(cute.nvgpu.warp.WarpMmaOp):
    """Backport of warp.MmaTF32Op for CuTe DSL 4.4.2."""

    def __init__(self, shape_mnk: tuple[int, int, int]) -> None:
        if shape_mnk not in ((16, 8, 4), (16, 8, 8)):
            raise ValueError(
                "TF32 warp MMA shape must be (16, 8, 4) or (16, 8, 8)"
            )
        self.shape_mnk = shape_mnk

    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        shape_mnk = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
        atom_type = _cute_nvgpu_ir.MmaAtomSM80Type.get(
            shape_mnk.type.attribute,
            cutlass.TFloat32.mlir_type,
            cutlass.TFloat32.mlir_type,
            cutlass.Float32.mlir_type,
        )
        return MmaTF32CompatTrait(
            make_atom(atom_type, loc=loc, ip=ip)
        )

    def _verify_fragment_A(self, input, *, loc=None, ip=None):
        return True

    def _verify_fragment_B(self, input, *, loc=None, ip=None):
        return True


MmaTF32Op = getattr(
    cute.nvgpu.warp, "MmaTF32Op", MmaTF32CompatOp
)


def quad_reduce_sum(value: cutlass.Float32) -> cutlass.Float32:
    """Return a sum replicated within each four-lane MMA row group."""

    for offset in (2, 1):
        value += cute.arch.shuffle_sync_bfly(
            value, offset=offset, mask=-1, mask_and_clamp=31
        )
    return value


def quad_reduce_max(value: cutlass.Float32) -> cutlass.Float32:
    """Return a maximum replicated within each four-lane MMA row group."""

    for offset in (2, 1):
        value = cute.arch.fmax(
            value,
            cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=-1, mask_and_clamp=31
            ),
        )
    return value


@cute.jit
def simplified_flash_attention2(
    m_q: cute.Tensor,
    m_k: cute.Tensor,
    m_v: cute.Tensor,
    m_o: cute.Tensor,
) -> None:
    if cutlass.const_expr(
        m_q.element_type != cutlass.Float32
        or m_k.element_type != cutlass.Float32
        or m_v.element_type != cutlass.Float32
        or m_o.element_type != cutlass.Float32
    ):
        raise TypeError(
            "Day 12 CuTe uses Float32 Q/K/V/output storage"
        )

    query_length, head_dim = m_q.shape
    key_length, key_head_dim = m_k.shape
    value_length, value_head_dim = m_v.shape
    output_length, output_head_dim = m_o.shape
    if cutlass.const_expr(
        key_length != value_length
        or head_dim != key_head_dim
        or head_dim != value_head_dim
        or query_length != output_length
        or head_dim != output_head_dim
    ):
        raise ValueError(
            "expected Q/O [M, D] and K/V [N, D]"
        )
    if cutlass.const_expr(head_dim > 128 or head_dim % 4 != 0):
        raise ValueError(
            "the TF32 TiledMMA lesson requires head_dim <= 128 "
            "and divisible by 4"
        )

    if cutlass.const_expr(head_dim % 8 == 0):
        mma_op = MmaTF32Op((16, 8, 8))
    else:
        mma_op = MmaTF32Op((16, 8, 4))
    tiled_mma = cute.make_tiled_mma(
        mma_op, atom_layout_mnk=(WARPS_PER_CTA, 1, 1)
    )
    simplified_flash_attention2_kernel(
        m_q, m_k, m_v, m_o, tiled_mma
    ).launch(
        grid=(cute.ceil_div(query_length, QUERY_TILE), 1, 1),
        block=(CTA_THREADS, 1, 1),
    )


@cute.kernel
def simplified_flash_attention2_kernel(
    g_q: cute.Tensor,
    g_k: cute.Tensor,
    g_v: cute.Tensor,
    g_o: cute.Tensor,
    tiled_mma: cute.TiledMma,
) -> None:
    query_length, head_dim = g_q.shape
    key_length, _ = g_k.shape
    mma_head_dim = cute.ceil_div(head_dim, 8) * 8
    softmax_scale = 1.0 / math.sqrt(head_dim)
    thread_idx, _, _ = cute.arch.thread_idx()
    query_block, _, _ = cute.arch.block_idx()
    lane = thread_idx % WARP_SIZE

    q_layout = cute.make_layout(
        (QUERY_TILE, head_dim),
        stride=(head_dim, 1),
    )
    k_layout = cute.make_layout(
        (KEY_TILE, head_dim),
        stride=(1, KEY_TILE),
    )
    probability_layout = cute.make_layout(
        (QUERY_TILE, KEY_TILE), stride=(KEY_TILE, 1)
    )
    value_layout = cute.make_layout(
        (mma_head_dim, KEY_TILE),
        stride=(1, mma_head_dim),
    )
    output_layout = cute.make_layout(
        (QUERY_TILE, mma_head_dim),
        stride=(mma_head_dim, 1),
    )
    smem = cutlass.utils.SmemAllocator()
    s_q = smem.allocate_tensor(cutlass.Float32, q_layout, 16)
    s_k = smem.allocate_tensor(cutlass.Float32, k_layout, 16)
    s_v = smem.allocate_tensor(
        cutlass.Float32, value_layout, 16
    )
    s_tile_output = smem.allocate_tensor(
        cutlass.Float32, output_layout, 16
    )
    s_probability_view = smem.allocate_tensor(
        cutlass.Float32, probability_layout, 16
    )

    tile_elements = QUERY_TILE * head_dim
    items_per_thread = cute.ceil_div(tile_elements, CTA_THREADS)
    value_tile_elements = KEY_TILE * mma_head_dim
    value_items_per_thread = cute.ceil_div(
        value_tile_elements, CTA_THREADS
    )
    for item in cutlass.range_constexpr(items_per_thread):
        linear_idx = thread_idx + item * CTA_THREADS
        if linear_idx < tile_elements:
            row = linear_idx // head_dim
            dimension = linear_idx % head_dim
            query_idx = query_block * QUERY_TILE + row
            if query_idx < query_length:
                s_q[row, dimension] = g_q[query_idx, dimension]
            else:
                s_q[row, dimension] = cutlass.Float32.zero

    cute.arch.sync_threads()

    thr_mma = tiled_mma.get_slice(thread_idx)

    t_q = thr_mma.partition_A(s_q)
    t_k = thr_mma.partition_B(s_k)
    t_scores = thr_mma.partition_C(s_probability_view)
    r_q = tiled_mma.make_fragment_A(t_q)
    r_k = tiled_mma.make_fragment_B(t_k)
    r_scores = tiled_mma.make_fragment_C(t_scores)

    t_probabilities = thr_mma.partition_A(s_probability_view)
    t_v = thr_mma.partition_B(s_v)
    t_tile_output = thr_mma.partition_C(s_tile_output)
    r_probabilities = tiled_mma.make_fragment_A(t_probabilities)
    r_v = tiled_mma.make_fragment_B(t_v)
    r_output = tiled_mma.make_fragment_C(t_tile_output)
    r_output.fill(0.0)
    r_running_max = cute.make_rmem_tensor(
        (2,), cutlass.Float32
    )
    r_running_sum = cute.make_rmem_tensor(
        (2,), cutlass.Float32
    )
    for row_slot in cutlass.range_constexpr(2):
        r_running_max[row_slot] = -cutlass.Float32.inf
        r_running_sum[row_slot] = cutlass.Float32.zero

    score_k_blocks = cute.size(r_q, mode=[2])
    output_k_blocks = cute.size(r_probabilities, mode=[2])
    output_n_blocks = cute.size(r_output, mode=[2])
    for k_block in cutlass.range_constexpr(score_k_blocks):
        cute.autovec_copy(
            t_q[None, None, k_block],
            r_q[None, None, k_block],
        )

    key_tiles = cute.ceil_div(key_length, KEY_TILE)
    for key_tile in cutlass.range(key_tiles):
        key_start = key_tile * KEY_TILE
        for item in cutlass.range_constexpr(items_per_thread):
            linear_idx = thread_idx + item * CTA_THREADS
            if linear_idx < tile_elements:
                key_offset = linear_idx // head_dim
                dimension = linear_idx % head_dim
                key_idx = key_start + key_offset
                if key_idx < key_length:
                    s_k[key_offset, dimension] = g_k[
                        key_idx, dimension
                    ]
                else:
                    s_k[
                        key_offset, dimension
                    ] = cutlass.Float32.zero
        for item in cutlass.range_constexpr(
            value_items_per_thread
        ):
            linear_idx = thread_idx + item * CTA_THREADS
            if linear_idx < value_tile_elements:
                key_offset = linear_idx // mma_head_dim
                dimension = linear_idx % mma_head_dim
                key_idx = key_start + key_offset
                if key_idx < key_length and dimension < head_dim:
                    s_v[dimension, key_offset] = g_v[
                        key_idx, dimension
                    ]
                else:
                    s_v[
                        dimension, key_offset
                    ] = cutlass.Float32.zero
        cute.arch.sync_threads()

        r_scores.fill(0.0)
        for k_block in cutlass.range_constexpr(score_k_blocks):
            cute.autovec_copy(
                t_k[None, None, k_block],
                r_k[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                r_scores,
                r_q[None, None, k_block],
                r_k[None, None, k_block],
                r_scores,
            )
        lane_in_quad = lane % 4
        for row_slot in cutlass.range_constexpr(2):
            accumulator_base = row_slot * 2
            thread_max = -cutlass.Float32.inf
            for n_block in cutlass.range_constexpr(
                cute.size(r_scores, mode=[2])
            ):
                for value in cutlass.range_constexpr(2):
                    column = (
                        n_block * 8 + lane_in_quad * 2 + value
                    )
                    key_idx = key_start + column
                    score = (
                        r_scores[
                            accumulator_base + value, 0, n_block
                        ]
                        * softmax_scale
                    )
                    if key_idx >= key_length:
                        score = -cutlass.Float32.inf
                    r_scores[
                        accumulator_base + value, 0, n_block
                    ] = score
                    thread_max = cute.arch.fmax(
                        thread_max, score
                    )
            tile_max = quad_reduce_max(thread_max)
            old_max = r_running_max[row_slot]
            new_max = cute.arch.fmax(old_max, tile_max)
            previous_scale = cute.math.exp(old_max - new_max)
            thread_sum = cutlass.Float32.zero
            for n_block in cutlass.range_constexpr(
                cute.size(r_scores, mode=[2])
            ):
                for value in cutlass.range_constexpr(2):
                    probability = cute.math.exp(
                        r_scores[
                            accumulator_base + value, 0, n_block
                        ]
                        - new_max
                    )
                    r_scores[
                        accumulator_base + value, 0, n_block
                    ] = probability
                    thread_sum += probability
            tile_sum = quad_reduce_sum(thread_sum)
            r_running_max[row_slot] = new_max
            r_running_sum[row_slot] = (
                r_running_sum[row_slot] * previous_scale
                + tile_sum
            )
            for n_block in cutlass.range_constexpr(
                output_n_blocks
            ):
                for value in cutlass.range_constexpr(2):
                    r_output[
                        accumulator_base + value, 0, n_block
                    ] = (
                        r_output[
                            accumulator_base + value, 0, n_block
                        ]
                        * previous_scale
                    )
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float32
            ),
            r_scores,
            t_scores,
        )
        cute.arch.sync_threads()
        for k_block in cutlass.range_constexpr(output_k_blocks):
            cute.autovec_copy(
                t_probabilities[None, None, k_block],
                r_probabilities[None, None, k_block],
            )
            cute.autovec_copy(
                t_v[None, None, k_block],
                r_v[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                r_output,
                r_probabilities[None, None, k_block],
                r_v[None, None, k_block],
                r_output,
            )
        cute.arch.sync_threads()

    for n_block in cutlass.range_constexpr(output_n_blocks):
        for accumulator in cutlass.range_constexpr(4):
            r_output[accumulator, 0, n_block] = (
                r_output[accumulator, 0, n_block]
                / r_running_sum[accumulator // 2]
            )
    cute.copy(
        cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32
        ),
        r_output,
        t_tile_output,
    )
    cute.arch.sync_threads()

    for item in cutlass.range_constexpr(items_per_thread):
        linear_idx = thread_idx + item * CTA_THREADS
        if linear_idx < tile_elements:
            row = linear_idx // head_dim
            dimension = linear_idx % head_dim
            query_idx = query_block * QUERY_TILE + row
            if query_idx < query_length:
                g_o[query_idx, dimension] = s_tile_output[row, dimension]


def _validate_inputs(
    q,
    k,
    v,
    expected_dtype=None,
    expected_ndim: int = 3,
    allow_cross_attention: bool = False,
) -> None:
    import torch

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, and v must be CUDA tensors")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must use the same dtype")
    if expected_dtype is not None and q.dtype != expected_dtype:
        raise TypeError(f"q, k, and v must use {expected_dtype}")
    invalid_shape = (
        q.ndim != expected_ndim
        or k.ndim != expected_ndim
        or v.ndim != expected_ndim
    )
    if allow_cross_attention:
        invalid_shape = invalid_shape or (
            k.shape != v.shape
            or q.shape[:-2] != k.shape[:-2]
            or q.shape[-1] != k.shape[-1]
        )
    else:
        invalid_shape = (
            invalid_shape or k.shape != q.shape or v.shape != q.shape
        )
    if invalid_shape:
        shape_contract = (
            "[batch_heads, sequence_length, head_dim]"
            if expected_ndim == 3
            else "Q [M, D] and K/V [N, D]"
        )
        raise ValueError(
            f"invalid attention shapes; expected {shape_contract}"
        )
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous")
    if any(
        dimension == 0
        for tensor in (q, k, v)
        for dimension in tensor.shape
    ):
        raise ValueError("attention dimensions must be non-zero")


def prepare_cute_attention(q, k, v):
    """Compile the 2D CuTe kernel and return stable runtime arguments."""

    import torch

    _validate_inputs(
        q,
        k,
        v,
        torch.float32,
        expected_ndim=2,
        allow_cross_attention=True,
    )
    output = torch.empty_like(q)
    q_cute = from_dlpack(q, assumed_align=16)
    k_cute = from_dlpack(k, assumed_align=16)
    v_cute = from_dlpack(v, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    compiled = cute.compile(
        simplified_flash_attention2,
        q_cute,
        k_cute,
        v_cute,
        output_cute,
    )
    return output, compiled, (q_cute, k_cute, v_cute, output_cute)


def build_triton_attention():
    """Define a forward-only FP16 FlashAttention-2 Triton kernel."""

    import triton
    import triton.language as tl

    @triton.jit
    def triton_flash_attention2_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        softmax_scale,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        query_indices = (
            query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        )
        key_offsets = tl.arange(0, BLOCK_N)
        dimensions = tl.arange(0, HEAD_DIM)
        query_mask = query_indices < N_CTX
        tensor_base = batch_head * N_CTX * HEAD_DIM
        q_offsets = (
            tensor_base
            + query_indices[:, None] * HEAD_DIM
            + dimensions[None, :]
        )
        q = tl.load(
            q_ptr + q_offsets,
            mask=query_mask[:, None],
            other=0.0,
        )

        running_max = (
            tl.zeros((BLOCK_M,), dtype=tl.float32)
            - float("inf")
        )
        running_sum = tl.zeros(
            (BLOCK_M,), dtype=tl.float32
        ) + 1.0
        output_accumulator = tl.zeros(
            (BLOCK_M, HEAD_DIM), dtype=tl.float32
        )
        qk_scale = softmax_scale * 1.44269504

        for key_start in tl.range(0, N_CTX, BLOCK_N):
            key_indices = key_start + key_offsets
            key_mask = key_indices < N_CTX
            kv_offsets = (
                tensor_base
                + key_indices[:, None] * HEAD_DIM
                + dimensions[None, :]
            )
            keys = tl.load(
                k_ptr + kv_offsets,
                mask=key_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(keys))
            scores = tl.where(
                key_mask[None, :],
                scores * qk_scale,
                -1.0e6,
            )

            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            probabilities = tl.math.exp2(
                scores - new_max[:, None]
            )
            previous_scale = tl.math.exp2(
                running_max - new_max
            )
            tile_sum = tl.sum(probabilities, axis=1)

            values = tl.load(
                v_ptr + kv_offsets,
                mask=key_mask[:, None],
                other=0.0,
            )
            output_accumulator *= previous_scale[:, None]
            output_accumulator = tl.dot(
                probabilities.to(tl.float16),
                values,
                output_accumulator,
            )
            running_sum = running_sum * previous_scale + tile_sum
            running_max = new_max

        output = output_accumulator / running_sum[:, None]
        tl.store(
            o_ptr + q_offsets,
            output.to(tl.float16),
            mask=query_mask[:, None],
        )

    return triton, triton_flash_attention2_kernel


_TRITON_MODULE = None
_TRITON_KERNEL = None


def launch_triton_attention(q, k, v, output) -> None:
    global _TRITON_MODULE, _TRITON_KERNEL

    import torch

    _validate_inputs(q, k, v, torch.float16)
    if output.shape != q.shape or output.dtype != q.dtype:
        raise ValueError("output must match q")
    if _TRITON_KERNEL is None:
        _TRITON_MODULE, _TRITON_KERNEL = build_triton_attention()
    batch_heads, sequence_length, head_dim = q.shape
    if head_dim not in (32, 64, 128):
        raise ValueError("Triton attention supports head_dim 32, 64, or 128")
    block_m = 64
    block_n = min(64, head_dim)
    grid = (
        _TRITON_MODULE.cdiv(sequence_length, block_m),
        batch_heads,
    )
    _TRITON_KERNEL[grid](
        q,
        k,
        v,
        output,
        1.0 / math.sqrt(head_dim),
        N_CTX=sequence_length,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=3,
    )


def attention_reference(q, k, v):
    import torch

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (
        torch.matmul(q.float(), k.float().transpose(-1, -2))
        * scale
    )
    output = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    return output.to(q.dtype)


TEST_SHAPES = ((2, 17, 32), (3, 33, 64), (4, 64, 64))
BENCHMARK_SHAPES = ((16, 256, 64), (16, 512, 64))
CUTE_TEST_SHAPES = (
    (2, 3, 4),
    (17, 19, 32),
    (33, 35, 64),
    (64, 64, 64),
    (512, 256, 128),
)
CUTE_BENCHMARK_SHAPES = ((256, 64), (512, 64), (4096, 64))


def _make_inputs(shape, seed, dtype=None):
    import torch

    if dtype is None:
        dtype = torch.float32
    torch.manual_seed(seed)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def test_triton_attention() -> None:
    import torch

    for index, shape in enumerate(TEST_SHAPES):
        q, k, v = _make_inputs(
            shape, 20261015 + index, torch.float16
        )
        reference = attention_reference(q, k, v)
        output = torch.empty_like(q)
        launch_triton_attention(q, k, v, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, reference, atol=2e-2, rtol=2e-2
        )
    print("Day 12 FP16 Triton FlashAttention-2: correctness OK")


def test_cute_attention() -> None:
    import torch

    for index, (query_length, key_length, head_dim) in enumerate(
        CUTE_TEST_SHAPES
    ):
        torch.manual_seed(20261015 + index)
        q = torch.randn(
            (query_length, head_dim),
            device="cuda",
            dtype=torch.float32,
        )
        k = torch.randn(
            (key_length, head_dim),
            device="cuda",
            dtype=torch.float32,
        )
        v = torch.randn_like(k)
        reference = attention_reference(q, k, v)
        output, compiled, arguments = prepare_cute_attention(q, k, v)
        compiled(*arguments)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, reference, atol=3e-2, rtol=3e-2
        )
    print("Day 12 FP32/TF32 CuTe attention: correctness OK")


def test_attention() -> None:
    test_triton_attention()
    test_cute_attention()


def attention_flops(
    batch_heads: int,
    sequence_length: int,
    head_dim: int,
) -> int:
    return (
        4
        * batch_heads
        * sequence_length
        * sequence_length
        * head_dim
    )


def benchmark_triton_attention(
    batch_heads: int,
    sequence_length: int,
    head_dim: int,
    warmup: int = 25,
    repetitions: int = 100,
) -> float:
    import torch
    import triton

    shape = (batch_heads, sequence_length, head_dim)
    q, k, v = _make_inputs(shape, 20261015, torch.float16)
    output = torch.empty_like(q)
    launch_triton_attention(q, k, v, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        attention_reference(q, k, v),
        atol=2e-2,
        rtol=2e-2,
    )
    return triton.testing.do_bench(
        lambda: launch_triton_attention(q, k, v, output),
        warmup=warmup,
        rep=repetitions,
    )


def benchmark_cute_attention(
    sequence_length: int,
    head_dim: int,
    warmup: int = 25,
    repetitions: int = 100,
) -> float:
    import torch
    import triton

    shape = (sequence_length, head_dim)
    q, k, v = _make_inputs(shape, 20261015)
    output, compiled, arguments = prepare_cute_attention(q, k, v)
    compiled(*arguments)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        attention_reference(q, k, v),
        atol=3e-2,
        rtol=3e-2,
    )
    return triton.testing.do_bench(
        lambda: compiled(*arguments),
        warmup=warmup,
        rep=repetitions,
    )


def benchmark_naive_attention(
    sequence_length: int,
    head_dim: int,
    warmup: int = 25,
    repetitions: int = 100,
) -> float:
    import torch
    import triton

    q, k, v = _make_inputs(
        (sequence_length, head_dim), 20261015
    )
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        output = attention_reference(q, k, v)
        torch.cuda.synchronize()
        assert output.shape == q.shape
        return triton.testing.do_bench(
            lambda: attention_reference(q, k, v),
            warmup=warmup,
            rep=repetitions,
        )
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def triton_benchmark_suite() -> None:
    print("BH x N x D | Triton ms | Triton TFLOP/s")
    print("----------- | --------- | ---------------")
    for shape in BENCHMARK_SHAPES:
        elapsed_ms = benchmark_triton_attention(*shape)
        print(
            f"{shape[0]:2d}x{shape[1]:4d}x{shape[2]:2d} | "
            f"{elapsed_ms:9.4f} | "
            f"{attention_flops(*shape) / (elapsed_ms * 1e9):15.2f}"
        )


def cute_benchmark_suite() -> None:
    print(
        "N x D | Naive ms | Naive TFLOP/s | "
        "CuTe ms | CuTe TFLOP/s | CuTe speedup"
    )
    print(
        "----- | -------- | ------------- | "
        "------- | ------------- | ------------"
    )
    for shape in CUTE_BENCHMARK_SHAPES:
        naive_ms = benchmark_naive_attention(*shape)
        cute_ms = benchmark_cute_attention(*shape)
        flops = attention_flops(1, *shape)
        print(
            f"{shape[0]:4d}x{shape[1]:2d} | "
            f"{naive_ms:8.4f} | "
            f"{flops / (naive_ms * 1e9):13.2f} | "
            f"{cute_ms:7.4f} | "
            f"{flops / (cute_ms * 1e9):13.2f} | "
            f"{naive_ms / cute_ms:12.2f}x"
        )


def benchmark_suite() -> None:
    triton_benchmark_suite()
    print()
    cute_benchmark_suite()


def run(benchmark: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Day 12 needs a CUDA device")
    cutlass.cuda.initialize_cuda_context()
    test_attention()
    if benchmark:
        benchmark_suite()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    run(args.benchmark)
