# Tensor Puzzles

Twenty-one compact PyTorch exercises for learning tensor programming from first
principles: broadcasting, masks, indexing, reductions, scatter-style updates,
reshaping, interpolation, repetition, and bucketization.

This is a local, output-free adaptation of Sasha Rush's MIT-licensed
[Tensor Puzzles](https://github.com/srush/Tensor-Puzzles), created with
Marcos Treviso and pinned to verified upstream commit
[`ad5069f`](https://github.com/srush/Tensor-Puzzles/commit/ad5069f7ef48523f733d445a7f94e93427d64f2d).
The upstream runtime package installation, downloaded helper, diagram
dependencies, and external reward video are replaced by
[`checker.py`](./checker.py). The local checker generates deterministic NumPy
inputs and compares each PyTorch implementation with the original loop-based
specification.

## Rules

Each solution should be one expression of fewer than 80 columns. You may use:

- `@`, arithmetic, and comparison operators;
- `shape`;
- indexing, including slices, `None`, and indices produced by `arange`;
- the provided `arange` and `where` helpers;
- functions solved in earlier puzzles.

Do not use convenience implementations such as `sum`, `view`, `take`,
`squeeze`, or constructors that directly produce the requested answer.

## Curriculum

| Puzzles | Topics |
| --- | --- |
| 1-6 | Ones, reduction, outer products, diagonals, identity, triangular masks |
| 7-13 | Prefix sums, differences, stacking, rolling, flipping, compression, padding |
| 14-17 | Sequence masks, bincount, scatter-add, flattening |
| 18-21 | Linspace, heaviside, repeat, bucketize |

## Start

```bash
cd tensor-puzzle
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab tensor_puzzlers.ipynb
```

Replace each `# fill in` stub, uncomment its `run_test(...)` line, and execute
the cell.

Validate the committed notebook without installing runtime dependencies:

```bash
python3 scripts/check_notebook.py
```

See [`LICENSE`](./LICENSE) for the upstream MIT license.
