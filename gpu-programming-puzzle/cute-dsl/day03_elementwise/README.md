# Day 03 — CuTe DSL Elementwise Add

**Goal:** write your first real CuTe-DSL kernel — `C = A + B` for FP16 matrices,
first naively (1 element / thread) then vectorized (8 elements / thread → one
128-bit transaction).

## Background

Elementwise add is memory-bound. To saturate DRAM you need *coalesced* (warp
threads touch contiguous addresses) and *vectorized* (one wide load per
instruction, not eight scalars) memory accesses.

### Step 1: naive — 1 element per thread

The simplest possible mapping. Each thread computes its global linear id,
turns it into `(mi, ni)`, and does one scalar load + store. This is already
coalesced as long as `ni` is the contiguous (mode-1) dimension.

### Step 2: vectorized — 8 elements per thread

`cute.zipped_divide(mA, (1, 8))` reshapes the *logical* `(M, N)` tensor into
`((1, 8), (M, N//8))`. The inner mode `(1, 8)` is the per-thread tile; the
outer mode is the per-thread *index*. Then:

```python
a_vec = gA[(None, (mi, ni))].load()    # one ld.global.v8.f16
gC[(None, (mi, ni))] = a_vec + b_vec   # one st.global.v8.f16
```

The `None` keeps the inner mode whole; `(mi, ni)` slices the outer. The DSL
knows the inner mode is 8 contiguous halves and emits a 128-bit instruction.

## Task

Fill in the two `# TODO`s in `puzzle.py`. The host code, tensor allocation,
verification, and launch math are all written — you only need the kernel bodies.

## Run

```bash
python puzzle.py --M 512 --N 1024    # NotImplementedError until you fix it
python solution.py --M 512 --N 1024  # reference

# Try a tiny size to inspect printed layouts:
python solution.py --M 64 --N 64
```

`N` must be divisible by 8 for the vectorized variant.

## What you should walk away knowing

- The two-step pattern: host `@cute.jit` does layout math + launches; device
  `@cute.kernel` does per-thread work.
- `zipped_divide` is the lever you pull to control "how many elements does each
  thread own?" — the rest of the kernel is the same.
- `.load()` on a `cute.Tensor` view returns a register fragment; an assignment
  back to a view emits a store. There is no explicit "for v in range(8)" loop.
- 1 element/thread vs 8 elements/thread is one line of host code, and 4–8x
  bandwidth in practice.
