// GEMM the Hard Way challenge edition.
// Day 01 blank focus: one thread computes one C element, including row/col mapping, dot product, and alpha/beta epilogue.
// Replace GEMM_TODO_* placeholders with the real implementation after
// studying the matching blog section and upstream reference.
#include "challenge_todo.cuh"


// FILL-IN BLANKS FOR THIS FILE
//   [ ] Blank A: identify the thread/block tile mapping.
//   [ ] Blank B: fill the global/shared/register load expression.
//   [ ] Blank C: fill the compute or MMA accumulation expression.
//   [ ] Blank D: fill the store/epilogue or launch configuration.
//   [ ] Blank E: write down the Nsight metric that should improve.

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

template <const uint block_size>
__global__ void sgemm_naive_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                   float alpha, const float *matrix_a,
                                   const float *matrix_b, float beta, float *output_matrix)
{
    // BLANK A: map the 1D thread ID to the 2D C element this thread owns.
    const int output_row = blockIdx.x * block_size + GEMM_TODO_INT("Day01: row offset from threadIdx.x");
    const int output_col = blockIdx.y * block_size + GEMM_TODO_INT("Day01: col offset from threadIdx.x");

    // Boundary check for non-multiple of block size
    if (output_row < num_rows_a && output_col < num_cols_b)
    {
        float accumulator = 0.0f;
        for (int k_idx = 0; k_idx < num_cols_a; ++k_idx)
        {
            // BLANK B: load A[row,k] and B[k,col], then multiply them.
            accumulator += GEMM_TODO_FLOAT("Day01: A[row,k] * B[k,col]");
        }
        // C = α*(A@B)+β*C
        const int output_idx = output_row * num_cols_b + output_col;
        // BLANK C: apply the alpha/beta epilogue for the output element.
        output_matrix[output_idx] = GEMM_TODO_FLOAT("Day01: alpha * accumulator + beta * C");
    }
}

void sgemm_naive(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                 torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch: 1D blocks with block_size^2 threads (32x32 = 1024 threads per block)
    constexpr uint block_size = 32;
    dim3 block_dim(block_size * block_size);
    dim3 grid_dim(ceil_div(num_rows_a, block_size),
                  ceil_div(num_cols_b, block_size));

    // Launch kernel
    sgemm_naive_kernel<block_size><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}