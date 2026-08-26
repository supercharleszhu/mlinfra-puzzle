#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

inline bool
verify_sgemm_with_cublas(
    int m, int n, int k,
    char trans_a, char trans_b,
    float alpha,
    thrust::device_vector<float> const& d_a, int ld_a,
    thrust::device_vector<float> const& d_b, int ld_b,
    float beta,
    thrust::host_vector<float> const& initial_c, int ld_c,
    thrust::host_vector<float> const& actual)
{
  cublasHandle_t handle = nullptr;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CORRECTNESS: FAILED to create cuBLAS handle\n";
    return false;
  }

  status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
  thrust::device_vector<float> d_reference = initial_c;
  if (status == CUBLAS_STATUS_SUCCESS) {
    cublasOperation_t op_a = trans_a == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t op_b = trans_b == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    status = cublasSgemm(
        handle,
        op_a, op_b,
        m, n, k,
        &alpha,
        d_a.data().get(), ld_a,
        d_b.data().get(), ld_b,
        &beta,
        d_reference.data().get(), ld_c);
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CORRECTNESS: FAILED to run cuBLAS reference, status "
              << static_cast<int>(status) << "\n";
    cublasDestroy(handle);
    return false;
  }

  thrust::host_vector<float> reference = d_reference;
  cublasDestroy(handle);

  constexpr double abs_tolerance = 1.0e-3;
  constexpr double rel_tolerance = 1.0e-3;
  double max_abs_error = 0.0;
  double max_scaled_error = 0.0;
  int mismatch_index = -1;

  for (int index = 0; index < m * n; ++index) {
    double expected = static_cast<double>(reference[index]);
    double observed = static_cast<double>(actual[index]);
    double abs_error = std::abs(observed - expected);
    double error_limit = abs_tolerance + rel_tolerance * std::abs(expected);
    double scaled_error = abs_error / error_limit;
    max_abs_error = std::max(max_abs_error, abs_error);
    max_scaled_error = std::max(max_scaled_error, scaled_error);
    if (mismatch_index < 0 && scaled_error > 1.0) {
      mismatch_index = index;
    }
  }

  if (mismatch_index >= 0) {
    int row = mismatch_index % ld_c;
    int column = mismatch_index / ld_c;
    std::cerr << "CORRECTNESS: FAILED at C(" << row << "," << column << ")"
              << ", expected=" << reference[mismatch_index]
              << ", actual=" << actual[mismatch_index]
              << ", max_abs_error=" << max_abs_error
              << ", max_scaled_error=" << max_scaled_error << "\n";
    return false;
  }

  std::cout << "CORRECTNESS: PASSED"
            << ", max_abs_error=" << max_abs_error
            << ", max_scaled_error=" << max_scaled_error << "\n";
  return true;
}
