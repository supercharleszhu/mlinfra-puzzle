# Day 09 - Persistent Matrix Multiplication

**Official topic:** [Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html).

A conventional GEMM launches one program per output tile. A persistent kernel
launches at most one program per SM and lets each program consume multiple
tiles. This reduces launch/scheduling work and exposes a steady-state loop that
can later use TMA, warp specialization, or cluster launch control.

## Challenge

1. Compute the total logical tile count.
2. Cap the physical grid at the GPU's SM count.
3. Start each program at its own ID.
4. Iterate `tile_id += NUM_SMS` until every logical tile is covered.
5. Rebuild A/B pointers and accumulators for every claimed tile.
6. Mask irregular matrix boundaries and compare with PyTorch.

The solution is the pointer-based FP16 baseline. The official tutorial extends
the same scheduler with Hopper tensor descriptors, TMA, FP8, epilogue
subtiling, warp specialization, and Blackwell CLC. Those mechanisms change how
tiles move and how resident work is assigned, but the persistent invariant
remains: the physical grid is smaller than the logical tile grid.
