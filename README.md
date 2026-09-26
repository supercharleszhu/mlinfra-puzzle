# mlinfra-puzzle

Hands-on systems and language-model implementation exercises.

## Repository layout

```text
mlinfra-puzzle/
├── gpu-programming-puzzle/
│   ├── cuda-cpp/
│   │   ├── challenge/        # GEMM challenge Days 01–28
│   │   ├── solutions/        # Completed CUDA/CuTe C++ implementations
│   │   └── scripts/          # Local and Kubernetes helpers
│   ├── python/               # Correctness and benchmark runners
│   ├── cute-dsl/             # Independent CuTe DSL Days 01–11
│   └── triton-puzzle/        # Triton tutorials and local challenges
├── algorithm-coding-challenge/
│   ├── grid-infection-puzzle/ # Multi-stage grid simulation interview mock
│   └── memory-allocator-puzzle/ # O(log m) allocator interview mock
├── llm-from-scratch-puzzle/  # Six compact PyTorch chapters
└── blog/                     # Full C++ tutorial references
```

The main curriculum is documented in
[`gpu-programming-puzzle/README.md`](gpu-programming-puzzle/README.md).
The Python DSL curriculum is documented in
[`gpu-programming-puzzle/cute-dsl/README.md`](gpu-programming-puzzle/cute-dsl/README.md).

## LLM from-scratch track

[`llm-from-scratch-puzzle/`](llm-from-scratch-puzzle/README.md) contains six
explanation-rich Jupyter notebooks with 20 original, CPU-friendly exercises on
token windows, attention, GPT sizing, generation and checkpoints,
classification, instruction formatting, and LoRA. It is a learning companion,
not a reproduction of the book, and uses only synthetic toy inputs.

```bash
llm-from-scratch-puzzle/.venv/bin/python \
  llm-from-scratch-puzzle/scripts/check_solutions.py
```

## Algorithm coding challenges

[`algorithm-coding-challenge/grid-infection-puzzle/`](algorithm-coding-challenge/grid-infection-puzzle/README.md)
is a staged simulation interview covering multi-source BFS, immune walls,
recovery timing, threshold infection, death variants, candidate-written unit
tests, and randomized differential testing.

[`algorithm-coding-challenge/memory-allocator-puzzle/`](algorithm-coding-challenge/memory-allocator-puzzle/README.md)
provides pointer-only and explicit-size allocator APIs, first-fit allocation,
adjacent-block coalescing, bitmap oracles, and an expected `O(log m)`
augmented-treap reference solution.

```bash
python3 algorithm-coding-challenge/grid-infection-puzzle/solution.py
python3 algorithm-coding-challenge/memory-allocator-puzzle/solution.py
```

## GEMM CUDA/CuTe C++ track

The 28-day GEMM path progresses through:

1. CUDA memory coalescing, shared-memory tiling, register tiling, and WMMA.
2. Handwritten Hopper TMA, WGMMA, warp specialization, and persistence.
3. CUTLASS 3.x Hopper configuration.
4. Official NVIDIA CuTe tutorials through Hopper and Blackwell.

```bash
cd gpu-programming-puzzle
python3 python/benchmark_advanced_day.py --day 20 --source solution --dry-run
cuda-cpp/scripts/upload_run_advanced_day.sh --day 20 --solution
```

## CuTe Python DSL track

The DSL track begins with 12 official NVIDIA CUTLASS notebooks in
`gpu-programming-puzzle/cute-dsl/notebooks/`, ordered from first launch through
Blackwell GEMM.
Eleven local challenges reinforce the core topics:

| Day | Topic |
| --- | --- |
| 01 | Hello DSL, decorators, and thread IDs |
| 02 | Layout algebra |
| 03 | Elementwise scalar and vectorized kernels |
| 04 | TV layouts and predication |
| 05 | Matrix transpose from naive mappings to padding and swizzling |
| 06 | Nsight Compute analysis of transpose bottlenecks |
| 07 | Producer/consumer asynchronous pipelines |
| 08 | Benchmarking and autotuning |
| 09 | CUDA Graph capture and replay |
| 10 | Tour to a complete SIMT GEMM |
| 11 | Stable softmax with local and warp reductions |

```bash
python gpu-programming-puzzle/cute-dsl/day01_hello_world/solution.py
python gpu-programming-puzzle/cute-dsl/day11_softmax_reduction/solution.py
```

All eleven days have challenge notebooks. To prepare and verify the
official Notebook 01 on the configured H100 pod:

```bash
gpu-programming-puzzle/cute-dsl/scripts/setup_remote_notebook.sh --verify
```

## Triton track

[`gpu-programming-puzzle/triton-puzzle/`](gpu-programming-puzzle/triton-puzzle/README.md)
contains one local puzzle and solution for each numbered official Triton
tutorial, followed by matrix-transpose and matched softmax-reduction
challenges. Each day includes a generated notebook.

```bash
python gpu-programming-puzzle/triton-puzzle/scripts/generate_notebooks.py
python gpu-programming-puzzle/triton-puzzle/scripts/check_curriculum.py
```

## Dependencies

- CUDA Toolkit 12.4+
- Python 3.10+
- NVIDIA CUTLASS/CuTe headers for the C++ track
- `nvidia-cutlass-dsl>=4.6,<5` and PyTorch for the DSL track

Only pinned official NVIDIA CUTLASS/CuTe sources are used by the advanced
curriculum and setup helpers.
