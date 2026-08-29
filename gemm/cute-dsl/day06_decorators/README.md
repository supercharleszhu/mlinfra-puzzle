# Day 06 -- CuTe DSL Decorator Boundaries

**Goal:** understand which functions execute in Python, on the JIT-compiled
host path, and on the GPU.

This challenge follows NVIDIA's
[CuTe DSL introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Call graph

```text
Python run()
  -> @cute.jit affine()                 compiled host function
       -> plain Python ceil_div()       inlined at compile time
       -> @cute.kernel affine_kernel()  launched on the GPU
            -> @cute.jit affine_value() inlined into the kernel
```

The reverse calls are not all legal: Python cannot directly invoke a
`@cute.kernel`, and one kernel cannot launch another kernel.

## Task

Fill the three TODOs in `puzzle.py`:

1. Implement the JIT-inlined affine helper.
2. Implement the bounds-checked GPU kernel.
3. Launch the kernel from the host-side `@cute.jit` function.

`scale` and `bias` are `cutlass.Constexpr` values. They specialize the
generated program and are not runtime arguments in the JIT executor.

## Run

```bash
python puzzle.py --n 1003 --scale 2.5 --bias -0.75
python solution.py --n 1003 --scale 2.5 --bias -0.75
```

The non-divisible size exercises the bounds check. PyTorch verifies every
output element.
