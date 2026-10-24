# Autodiff Puzzles

Twenty progressively harder exercises for implementing Jacobians of tensor
operations: scalar transforms, reductions, indexing, sorting, convolution,
reshaping, matrix multiplication, and 2D convolution.

This is a local, output-free adaptation of Sasha Rush's MIT-licensed
[Autodiff Puzzles](https://github.com/srush/Autodiff-Puzzles), pinned to
verified upstream commit
[`ae4586b`](https://github.com/srush/Autodiff-Puzzles/commit/ae4586b052f3328031ad64024edabbf6fbd2e4bb).
The exercise cells are preserved. The upstream runtime GitHub package install
and animation helper are replaced by [`checker.py`](./checker.py), which
compares each hand-written Jacobian directly with `jax.jacrev`.

## Start

```bash
cd autodiff-puzzle
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab autodiff_puzzlers.ipynb
```

Replace each `# fill in` expression and run its cell. An incorrect Jacobian
raises an assertion showing whether its shape or values differ from JAX.

Validate the committed notebook without installing its runtime dependencies:

```bash
python3 scripts/check_notebook.py
```

See [`LICENSE`](./LICENSE) for the upstream MIT license.
