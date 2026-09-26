# Day 04 - Low-Memory Dropout

**Official topic:** [Low-Memory Dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html).

Saving a full random mask costs one byte or more per activation. Counter-based
randomness lets backward or a repeated invocation reconstruct the same mask
from only a seed and element offset. This challenge uses `tl.rand(seed,
offsets)` so each logical element has a deterministic random counter.

## Challenge

1. Reuse the vector-add offset and boundary-mask pattern.
2. Generate random values directly from the global offsets.
3. Keep values where `random > p`.
4. Apply inverted-dropout scaling `x / (1-p)` so the expected output is x.
5. Prove equal seeds reproduce the mask and different seeds change it.

Validate probability boundaries explicitly; `p == 1` would divide by zero and
must be rejected. A statistical check should use enough elements to tolerate
random variation while still detecting missing scaling.

The memory lesson matters more than the arithmetic: the operation stores only
the output, and deterministic counters eliminate the separately materialized
mask that a conventional implementation might retain.
