# Day 01 - Vector Addition

**Official topic:** [Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).

This first challenge establishes Triton's execution model. A program instance
owns one power-of-two block of offsets; `tl.arange` creates the vector of lane
offsets, and a mask protects the final partial block. Unlike a CUDA thread,
each Triton program expresses operations over a whole vector.

## Challenge

1. Compute `offsets = program_id * BLOCK_SIZE + arange`.
2. Build `offsets < n_elements` and pass it to both loads and the store.
3. Allocate an output tensor and launch enough programs with `triton.cdiv`.
4. Validate a size that is not divisible by the block size.

Run `python puzzle.py`, then compare with `python solution.py`. The important
invariant is that masked-off lanes never dereference out-of-range pointers.
After correctness, benchmark the kernel with `triton.testing.do_bench` and
report effective bandwidth as three tensor transfers: two reads and one write.

This lesson prepares every later challenge: program IDs choose tiles, vector
offsets describe data ownership, compile-time meta-parameters shape the tile,
and masks separate logical bounds from the physical launch grid.
