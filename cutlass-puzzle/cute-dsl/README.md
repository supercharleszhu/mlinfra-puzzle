# CuTe DSL Tutorial and Challenges

Start with the 12 official NVIDIA notebooks in [`notebooks/`](notebooks/).
They are copied unchanged from a pinned CUTLASS commit and numbered in
prerequisite order:

| Steps | Topic |
| --- | --- |
| 01–03 | JIT/kernel mechanics, data types, and debugging |
| 04–07 | Tensors, layout algebra, composed layouts, and TensorSSA |
| 08–11 | Elementwise kernels, async pipelines, autotuning, and CUDA graphs |
| 12 | Blackwell TMA/UMMA/TMEM GEMM |

After the relevant notebook, reinforce it with the local puzzle/solution:

| Challenge | Topic | Read first |
| --- | --- | --- |
| 01 | `@cute.jit`, `@cute.kernel`, and thread IDs | Notebook 01 |
| 02 | Layout algebra | Notebooks 04–06 |
| 03 | Scalar and vectorized elementwise add | Notebooks 07–08 |
| 04 | Notebook 08's TV-layout elementwise stage plus OOB predication | Notebooks 07–08 |
| 05 | Producer/consumer async pipeline | Notebook 09 |
| 06 | Benchmarking and configuration autotuning | Notebook 10 |
| 07 | CUDA Graph capture and replay | Notebook 11 |
| 08 | Tour to a complete H100-compatible SIMT GEMM | Notebook 12 |

Each `dayNN_*` directory contains `puzzle.py`, a completed `solution.py`, and
a README. Every day also has a short, challenge-focused notebook.

Day 03 implements Notebook 08's scalar and per-thread vector elementwise
stages. Day 04 continues with CTA tiling and explicit `(thread, value)` layout
ownership, then adds coordinate predication and compile-once bandwidth
benchmarking. Read
[`day04_tv_layout_predicated/README.md`](day04_tv_layout_predicated/README.md)
for the full mapping and tiler derivation.

Notebook 12 requires an SM100 Blackwell GPU. The configured H100 pod can run
Steps 01–11, except for the TMEM section of Step 04.

## Local setup

CuTe DSL requires an SM80-or-newer GPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r cutlass-puzzle/cute-dsl/requirements.txt

python cutlass-puzzle/cute-dsl/day01_hello_world/solution.py
python cutlass-puzzle/cute-dsl/day08_gemm_tour/solution.py
jupyter notebook cutlass-puzzle/cute-dsl/notebooks/01_hello_world.ipynb
```

Select a `cuda-python` major compatible with the remote driver as documented
in `requirements.txt`.

## H100 pod notebook

The setup helper uploads the tutorial and challenges, creates an isolated
environment using official NVIDIA CuTe DSL packages, and verifies Notebook 01:

```bash
cutlass-puzzle/cute-dsl/scripts/setup_remote_notebook.sh --verify
```

It prints the `kubectl exec` command for starting Jupyter and the
`kubectl port-forward` command for opening it locally.
