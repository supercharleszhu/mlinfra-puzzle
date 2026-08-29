#pragma once

// Placeholder helpers for the standalone challenge files.
// They intentionally produce harmless dummy values so the blanks are easy to
// grep. Replace each GEMM_TODO_* call with the real code for that day.
#define GEMM_TODO_INT(label) 0
#define GEMM_TODO_UINT(label) 1u
#define GEMM_TODO_FLOAT(label) 0.0f
#define GEMM_TODO_FLOAT4(label) make_float4(0.0f, 0.0f, 0.0f, 0.0f)
#define GEMM_TODO_WMMA_LOAD(label) ((void)0)
#define GEMM_TODO_WMMA_MMA(label) ((void)0)
