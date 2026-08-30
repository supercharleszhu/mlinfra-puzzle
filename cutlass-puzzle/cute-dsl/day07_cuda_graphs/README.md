# Day 07 - CUDA Graph Capture and Replay

**Goal:** precompile a CuTe DSL launch, capture repeated launches into a CUDA
Graph, and replay them with one host submission.

Read official Notebook
[`11_cuda_graphs.ipynb`](../notebooks/11_cuda_graphs.ipynb). The critical rule
is:

> Compile before capture; invoke only the already-compiled callable while
> capture is active.

Compilation may allocate memory or load a CUDA module, operations that are not
safe inside stream capture. The runtime callable must also receive the
`CUstream` corresponding to PyTorch's current capture stream.

Fill the five TODOs in `puzzle.py`:

```bash
python puzzle.py --operations 8
python solution.py --operations 8
```

The kernel increments one scalar. After two graph replays containing eight
launches each, the scalar must equal 16.
