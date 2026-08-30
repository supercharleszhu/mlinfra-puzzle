# Day 01 — CuTe DSL Hello World

**Goal:** launch your first CuTe-DSL kernel from Python, and have every thread
write its global linear ID into a tensor.

## Background

CuTe DSL is the Python-frontend cousin of CuTe C++. The same Layout/Tensor
algebra you learned in days 1–9, but expressed as decorators on Python
functions that get JIT-compiled to PTX/CUBIN.

The two decorators you need:

| Decorator     | Runs on | Purpose                                              |
|---------------|---------|------------------------------------------------------|
| `@cute.jit`   | Host    | A regular Python function that may *launch* kernels. |
| `@cute.kernel`| Device  | The GPU kernel body — one instance per thread.       |

Inside a `@cute.kernel`:

```python
tidx, tidy, tidz = cute.arch.thread_idx()   # like CUDA threadIdx
bidx, bidy, bidz = cute.arch.block_idx()    # like CUDA blockIdx
bdim, _,    _    = cute.arch.block_dim()    # like CUDA blockDim
cute.printf("fmt {} {}", a, b)              # device-side printf
gA[i] = value                               # tensor store
```

Two flavors of `if`:

- **Static** (compile-time):  `if cutlass.const_expr(x == 0):` — branches
  resolved during JIT trace.
- **Dynamic** (runtime):      a plain `if tidx == 0:` — DSL 4.5+ traces this
  as a dynamic branch automatically. (Older snippets in the wild use
  `cutlass.dynamic_expr(...)`, now deprecated.)

You'll want **dynamic** for any condition that depends on `thread_idx`.

## Task

Fill in the two TODOs in `puzzle.py`:

1. **`hello_kernel`** — only thread 0 of block 0 prints `"hello"`.
2. **`write_tid_kernel`** — each thread computes `gid = bidx*bdim + tidx`
   and stores it into `gA[gid]`. Cast `gid` to `gA.element_type` first.

## Build & run

```bash
# (one-time) install deps — see top-level README "CuTe DSL setup" section.

# Run the broken puzzle: should raise NotImplementedError.
python puzzle.py

# Reference answer:
python solution.py
```

Expected output:

```
=== Part 1: hello_world ===
hello from CuTe DSL: tidx=0 bidx=0

=== Part 2: write_thread_id ===
OK — first 8 = [0, 1, 2, 3, 4, 5, 6, 7]

Success.
```

## What you should walk away knowing

- The host/device split: `@cute.jit` traces Python and emits IR; `@cute.kernel`
  becomes a CUDA function with one invocation per thread.
- `cute.printf` is device-side and asynchronous — it may print *after* the
  Python interpreter has continued.
- `from_dlpack(torch_tensor)` is the bridge from PyTorch into CuTe.
- The unit test for a CuTe kernel is just `torch.testing.assert_close` against
  a reference torch expression. Use this aggressively.
