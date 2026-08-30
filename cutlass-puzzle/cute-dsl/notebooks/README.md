# Official NVIDIA CuTe DSL Notebooks

These notebooks are unchanged copies from NVIDIA CUTLASS, renamed with numeric
prefixes to make their prerequisite order explicit.

- Source: [`examples/python/CuTeDSL/cute/notebooks`](https://github.com/NVIDIA/cutlass/tree/ffa119a1255d78998536107466cc7097ecefa393/examples/python/CuTeDSL/cute/notebooks)
- Commit: `ffa119a1255d78998536107466cc7097ecefa393`
- License: [`LICENSE.nvidia`](LICENSE.nvidia), BSD-3-Clause

| Step | Notebook | Concepts | Practice | H100 |
| --- | --- | --- | --- | --- |
| 01 | [`01_hello_world.ipynb`](01_hello_world.ipynb) | JIT functions, kernels, and launches | Day 01 | Yes |
| 02 | [`02_data_types.ipynb`](02_data_types.ipynb) | Static and dynamic values, primitive types | Day 01 | Yes |
| 03 | [`03_print.ipynb`](03_print.ipynb) | Compile-time and device-side printing | Day 01 | Yes |
| 04 | [`04_tensor.ipynb`](04_tensor.ipynb) | Engines, layouts, DLPack, and memory spaces | Days 02–04 | Partial; TMEM is Blackwell-only |
| 05 | [`05_layout_algebra.ipynb`](05_layout_algebra.ipynb) | CuTe layout algebra | Day 02 | Yes |
| 06 | [`06_composed_layout.ipynb`](06_composed_layout.ipynb) | Composed layouts and coordinate transforms | Day 02 | Yes |
| 07 | [`07_tensor_ssa.ipynb`](07_tensor_ssa.ipynb) | Register-resident TensorSSA operations | Days 03–04 | Yes |
| 08 | [`08_elementwise_add.ipynb`](08_elementwise_add.ipynb) | Vectorized copies, TV layouts, and predicates | Days 03–04 | Yes |
| 09 | [`09_async_pipeline.ipynb`](09_async_pipeline.ipynb) | Warp specialization and multistage pipelines | [Day 05](../day05_async_pipeline/) | Yes |
| 10 | [`10_benchmark_autotune.ipynb`](10_benchmark_autotune.ipynb) | Benchmarking and configuration search | [Day 06](../day06_benchmark_autotune/) | Yes |
| 11 | [`11_cuda_graphs.ipynb`](11_cuda_graphs.ipynb) | Streams and CUDA graph capture | [Day 07](../day07_cuda_graphs/) | Yes |
| 12 | [`12_tour_to_sol_gemm.ipynb`](12_tour_to_sol_gemm.ipynb) | TMA, UMMA, TMEM, and Blackwell GEMM | [Day 08](../day08_gemm_tour/) | No; requires SM100 |

The `images/` directory is also copied from the same commit because Steps 09,
11, and 12 reference those assets by relative path.
