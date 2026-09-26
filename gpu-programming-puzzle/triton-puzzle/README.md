# Triton Tutorial Challenges

This track is parallel to [`../cute-dsl/`](../cute-dsl/). For Days 01-11, each
notebook starts with the official downloadable Triton tutorial and appends a
local TODO challenge with correctness checks. Each day also retains a
standalone `puzzle.py`, complete `solution.py`, and short guide. The sequence
covers every numbered module in Triton's official
[tutorial gallery](https://triton-lang.org/main/getting-started/tutorials/),
followed by matrix-transpose and softmax-reduction challenges that compare
directly with the matching CuTe lessons.

The official notebook snapshots and their MIT license are kept in
[`official-notebooks/`](./official-notebooks/). Day 11 is generated from its
official Triton source because the current website notebook URL is unavailable.
The generator preserves every official cell and only adds challenge cells at
the end.

| Day | Challenge | Official tutorial |
| --- | --- | --- |
| 01 | Vector addition, launch grids, offsets, and masks | [01 Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) |
| 02 | Row-wise fused softmax and reductions | [02 Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) |
| 03 | Tiled, grouped-order matrix multiplication | [03 Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) |
| 04 | Seeded low-memory dropout | [04 Low-Memory Dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html) |
| 05 | LayerNorm forward and backward reductions | [05 Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) |
| 06 | Fused scaled dot-product attention | [06 Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) |
| 07 | Calling `libdevice` functions | [07 External Functions](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html) |
| 08 | Device-scheduled grouped GEMM | [08 Grouped GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) |
| 09 | Persistent matrix multiplication | [09 Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) |
| 10 | Block-scaled matrix multiplication semantics | [10 Block-Scaled Matmul](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html) |
| 11 | Programmatic dependent launch | [11 Programmatic Dependent Launch](https://github.com/triton-lang/triton/blob/main/python/tutorials/11-programmatic-dependent-launch.py) |
| 12 | Coalesced, tiled matrix transpose | Original extension |
| 13 | Stable softmax tests and matched CuTe/Triton benchmarks | Original extension |

Days 01-09 and 12-13 run on CUDA GPUs supported by Triton. Day 10 includes a
portable scale-semantics kernel that runs on H100 and explains how Blackwell
maps the same model to `tl.dot_scaled`; native block-scaled MMA requires
compute capability 10 or newer. Day 11 requires CUDA compute capability 9 or
newer.

## Setup

Use the PyTorch-provided Triton installation when available:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python day01_vector_add/solution.py
```

Regenerate and check every notebook:

```bash
python scripts/generate_notebooks.py
python scripts/check_curriculum.py
```

Prepare the H100 workspace:

```bash
scripts/setup_remote_notebook.sh --verify
```
