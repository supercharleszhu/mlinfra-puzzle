# Day 06 - Benchmarking and Autotuning

**Goal:** separate kernel correctness from configuration search and reliable
measurement.

Read official Notebook
[`10_benchmark_autotune.ipynb`](../notebooks/10_benchmark_autotune.ipynb).
The exercise reuses Day 04's predicated elementwise kernel and tunes its copy
width between 64 and 128 bits.

Three APIs have different jobs:

- `cutlass.cute.testing.autotune_jit` caches the fastest specialization for
  changing problem dimensions.
- `cutlass.testing.tune` exposes the selected parameter dictionary explicitly.
- `cutlass.testing.benchmark` uses CUDA timing and rotating workspaces to
  reduce cache bias.

Fill six TODOs in `puzzle.py`, then compare with the solution:

```bash
python puzzle.py --M 512 --N 512
python solution.py --M 512 --N 512
```

Correctness is checked before timing. The measured operation moves three
tensors, so effective bandwidth is `3 * elements * sizeof(dtype) / time`.
