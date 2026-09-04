# mlinfra-puzzle

Hands-on systems and language-model implementation exercises.

## Repository layout

```text
mlinfra-puzzle/
├── cutlass-puzzle/
│   ├── cuda/                 # GEMM challenge Days 01–28
│   ├── solutions/            # Completed CUDA/CuTe C++ implementations
│   ├── python/               # Correctness and benchmark runners
│   ├── scripts/              # Local and Kubernetes helpers
│   └── cute-dsl/             # Independent CuTe DSL Days 01–08
├── llm-from-scratch-puzzle/  # Six compact PyTorch chapters
└── blog/                     # Full C++ tutorial references
```

The main curriculum is documented in
[`cutlass-puzzle/README.md`](cutlass-puzzle/README.md).
The Python DSL curriculum is documented in
[`cutlass-puzzle/cute-dsl/README.md`](cutlass-puzzle/cute-dsl/README.md).

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

## GEMM CUDA/CuTe C++ track

The 28-day GEMM path progresses through:

1. CUDA memory coalescing, shared-memory tiling, register tiling, and WMMA.
2. Handwritten Hopper TMA, WGMMA, warp specialization, and persistence.
3. CUTLASS 3.x Hopper configuration.
4. Official NVIDIA CuTe tutorials through Hopper and Blackwell.

```bash
cd cutlass-puzzle
python3 python/benchmark_advanced_day.py --day 20 --source solution --dry-run
scripts/upload_run_advanced_day.sh --day 20 --solution
```

## CuTe Python DSL track

The DSL track begins with 12 official NVIDIA CUTLASS notebooks in
`cutlass-puzzle/cute-dsl/notebooks/`, ordered from first launch through
Blackwell GEMM.
Eight local challenges reinforce the core topics:

| Day | Topic |
| --- | --- |
| 01 | Hello DSL, decorators, and thread IDs |
| 02 | Layout algebra |
| 03 | Elementwise scalar and vectorized kernels |
| 04 | TV layouts and predication |
| 05 | Producer/consumer asynchronous pipelines |
| 06 | Benchmarking and autotuning |
| 07 | CUDA Graph capture and replay |
| 08 | Tour to a complete SIMT GEMM |

```bash
python cutlass-puzzle/cute-dsl/day01_hello_world/solution.py
python cutlass-puzzle/cute-dsl/day08_gemm_tour/solution.py
```

All eight days have shorter challenge notebooks. To prepare and verify the
official Notebook 01 on the configured H100 pod:

```bash
cutlass-puzzle/cute-dsl/scripts/setup_remote_notebook.sh --verify
```

## Dependencies

- CUDA Toolkit 12.4+
- Python 3.10+
- NVIDIA CUTLASS/CuTe headers for the C++ track
- `nvidia-cutlass-dsl>=4.6,<5` and PyTorch for the DSL track

Only pinned official NVIDIA CUTLASS/CuTe sources are used by the advanced
curriculum and setup helpers.
