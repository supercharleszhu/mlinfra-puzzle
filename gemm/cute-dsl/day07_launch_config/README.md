# Day 07 -- CuTe DSL Kernel Launch Configuration

**Goal:** connect CuTe DSL's `.launch()` arguments to CUDA execution and
compiler metadata.

This challenge follows the kernel-launch section of NVIDIA's
[CuTe DSL introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

| Argument | Meaning |
| --- | --- |
| `grid` | Number of thread blocks launched in each dimension. |
| `block` | Threads actually launched per block. |
| `max_number_threads` | Compile-time `maxntid` upper bound. |
| `min_blocks_per_mp` | Compile-time `minctasm` occupancy hint. |

The remaining introduction options—clusters, fallback clusters, cooperative
launch, PDL, and shared-memory policies—matter for later Hopper/Blackwell
kernels and are intentionally not enabled in this architecture-neutral
exercise.

## Task

Fill two TODOs in `puzzle.py`:

1. Implement a bounds-checked scaling kernel.
2. Supply the explicit launch configuration.

The actual block size may be 64, 128, or 256, while the compiled maximum is
256 threads. Both the tensor length and block size are exercised by the
correctness check.

## Run

```bash
python puzzle.py --n 4099 --threads 128
python solution.py --n 4099 --threads 128
python solution.py --n 4099 --threads 256
```
