# cutlass-puzzle

Hands-on CUDA, CUTLASS, CuTe C++, and CuTe Python DSL exercises.

## Repository layout

```text
cutlass-puzzle/
├── gemm/
│   ├── cuda/                 # GEMM challenge Days 01–28
│   ├── solutions/            # Completed CUDA/CuTe C++ implementations
│   ├── python/               # Correctness and benchmark runners
│   ├── scripts/              # Local and Kubernetes helpers
│   └── cute-dsl/             # Independent CuTe DSL Days 01–07
└── blog/                     # Full C++ tutorial references
```

The main curriculum is documented in [`gemm/README.md`](gemm/README.md).
The Python DSL curriculum is documented in
[`gemm/cute-dsl/README.md`](gemm/cute-dsl/README.md).

## GEMM CUDA/CuTe C++ track

The 28-day GEMM path progresses through:

1. CUDA memory coalescing, shared-memory tiling, register tiling, and WMMA.
2. Handwritten Hopper TMA, WGMMA, warp specialization, and persistence.
3. CUTLASS 3.x Hopper configuration.
4. Official NVIDIA CuTe tutorials through Hopper and Blackwell.

```bash
cd gemm
python3 python/benchmark_advanced_day.py --day 20 --source solution --dry-run
scripts/upload_run_advanced_day.sh --day 20 --solution
```

## CuTe Python DSL track

The DSL track begins with 12 official NVIDIA CUTLASS notebooks in
`gemm/cute-dsl/notebooks/`, ordered from first launch through Blackwell GEMM.
Seven local challenges reinforce the core topics:

| Day | Topic |
| --- | --- |
| 01 | Hello DSL, decorators, and thread IDs |
| 02 | Layout algebra |
| 03 | Elementwise scalar and vectorized kernels |
| 04 | TV layouts and predication |
| 05 | Single-stage SIMT GEMM |
| 06 | JIT/kernel calling conventions |
| 07 | Kernel launch configuration |

```bash
python gemm/cute-dsl/day01_hello_world/solution.py
python gemm/cute-dsl/day07_launch_config/solution.py
```

Days 01–05 also retain shorter challenge notebooks. To prepare and verify the
official Notebook 01 on the configured H100 pod:

```bash
gemm/cute-dsl/scripts/setup_remote_notebook.sh --verify
```

## Dependencies

- CUDA Toolkit 12.4+
- Python 3.10+
- NVIDIA CUTLASS/CuTe headers for the C++ track
- `nvidia-cutlass-dsl>=4.5,<5` and PyTorch for the DSL track

Only pinned official NVIDIA CUTLASS/CuTe sources are used by the advanced
curriculum and setup helpers.
