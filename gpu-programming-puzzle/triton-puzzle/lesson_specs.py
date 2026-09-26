"""Canonical metadata for the Triton challenge curriculum."""

LESSONS = [
    ("day01_vector_add", "Vector Addition", "01-vector-add.py", "vector_add_kernel"),
    ("day02_fused_softmax", "Fused Softmax", "02-fused-softmax.py", "softmax_kernel"),
    (
        "day03_matrix_multiplication",
        "Matrix Multiplication",
        "03-matrix-multiplication.py",
        "matmul_kernel",
    ),
    (
        "day04_low_memory_dropout",
        "Low-Memory Dropout",
        "04-low-memory-dropout.py",
        "dropout_kernel",
    ),
    ("day05_layer_norm", "Layer Normalization", "05-layer-norm.py", "layer_norm"),
    ("day06_fused_attention", "Fused Attention", "06-fused-attention.py", "attention_kernel"),
    ("day07_extern_functions", "External Functions", "07-extern-functions.py", "libdevice.asin"),
    ("day08_grouped_gemm", "Grouped GEMM", "08-grouped-gemm.py", "grouped_matmul_kernel"),
    ("day09_persistent_matmul", "Persistent Matmul", "09-persistent-matmul.py", "NUM_SMS"),
    (
        "day10_block_scaled_matmul",
        "Block-Scaled Matmul",
        "10-block-scaled-matmul.py",
        "scale_a",
    ),
    (
        "day11_programmatic_dependent_launch",
        "Programmatic Dependent Launch",
        "11-programmatic-dependent-launch.py",
        "gdc_wait",
    ),
    ("day12_matrix_transpose", "Matrix Transpose", None, "transpose_kernel"),
    (
        "day13_softmax_reduction",
        "Softmax Reduction Benchmark",
        None,
        "softmax_kernel",
    ),
]
