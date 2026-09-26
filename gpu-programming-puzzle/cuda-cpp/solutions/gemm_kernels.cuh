#pragma once
#include <torch/torch.h>

// Naive SGEMM implementation
void sgemm_naive(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                 torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with global memory coalescing
void sgemm_global_mem_coalesce(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                               torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with shared memory tiling
void sgemm_shared_mem(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with 1D block tiling
void sgemm_blocktiling_1d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with 2D block tiling
void sgemm_blocktiling_2d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with vectorized memory access
void sgemm_vectorize(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                     torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with warp-level tiling (full templatization)
template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int WM = 64, const int WN = 64, const int WNITER = 4,
          const int TM = 8, const int TN = 4, const int NUM_THREADS = 128>
void sgemm_warptiling(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM warptiling with default parameters (for Python binding)
// FP32 version - uses the original warptiling kernel
void sgemm_warptiling_default(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                              torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM warptiling with multi-dtype support (FP16, BF16)
// Input/output use same dtype, like PyTorch behavior
void sgemm_warptiling_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta);

void sgemm_warptiling_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with Tensor Cores - Naive version
// Input/output use same dtype (FP16 or BF16), like PyTorch behavior
// Each warp processes a single 16x16 WMMA tile without block/warp tiling
void sgemm_tensorcore_naive_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta);

void sgemm_tensorcore_naive_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with Tensor Cores - Optimized version
// Input/output use same dtype (FP16 or BF16), like PyTorch behavior
// Block and warp-level tiling with shared memory padding to reduce bank conflicts
void sgemm_tensorcore_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta);

void sgemm_tensorcore_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with Tensor Cores and Double Buffering
// Input: FP16 or BF16, Output: FP32
// Overlaps memory loads with computation for better performance
void sgemm_tensorcore_double_buffered_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                           torch::Tensor &output_matrix, float alpha, float beta);

void sgemm_tensorcore_double_buffered_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                           torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with Tensor Cores and Async Pipeline (cp.async)
// Input: FP16 or BF16, Output: FP32
// Uses async memory copies with multi-stage pipeline for maximum overlap
// Requires SM 8.0+ (Ampere and newer)
void sgemm_tensorcore_async_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta);

void sgemm_tensorcore_async_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta);

// CUTLASS 3.x Hopper (SM90) TMA/WGMMA GEMM with tunable schedules.
void sgemm_cutlass3_hopper_tunable_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix);

void sgemm_fastcu_matmul2_manual_tma_wgmma_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed);

// Handwritten fast.cu-derived Hopper kernel.
// Expects B^T and writes C^T because the upstream kernel uses column-major B/C views.
void sgemm_fastcu_handwritten_tma_wgmma_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed);

void sgemm_fastcu_handwritten_cached_tma_maps_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed);

void sgemm_fastcu_final_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed);

void sgemm_fastcu_tma_store_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed);

void sgemm_fastcu_hilbert_final_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed);
