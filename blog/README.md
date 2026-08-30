# Blog: Reference Code

These are the **complete, unmodified** CUTLASS/CuTe tutorial programs referenced in the blog post
[*CUTLASS Deep Dive: From CuTe Layouts to Blackwell tcgen05*](https://supercharleszhu.github.io).

Each file is a single-translation-unit program you can build and run. They are numbered in the same
order they are introduced in the blog, so you can read top-down.

| # | File | What it teaches | Minimum arch |
|---|------|-----------------|--------------|
| 1 | `01_tiled_copy.cu` | Layouts, `tiled_divide`, `TiledCopy` with vectorized loads | Any |
| 2 | `02_sgemm_sm80.cu` | Software-pipelined GEMM with `cp.async`, `ldmatrix`, swizzles | SM80 (A100) |
| 3 | `03_wgmma_sm90.cu` | Hopper warp-group MMA (smem-source, async) without TMA | SM90 (H100) |
| 4 | `04_wgmma_tma_sm90.cu` | Hopper WGMMA + TMA + cluster launch + mbarrier pipeline | SM90 (H100) |
| 5 | `05_blackwell_mma_sm100.cu` | Blackwell `tcgen05.mma` + TMEM allocation + `tcgen05.ld` | SM100 (B200) |
| 6 | `06_blackwell_mma_tma_sm100.cu` | Add `SM90_TMA_LOAD` to the SM100 mainloop | SM100 |
| 7 | `07_blackwell_mma_tma_multicast_sm100.cu` | Cluster multicast TMA loads | SM100 |
| 8 | `08_blackwell_mma_tma_2sm_sm100.cu` | 2SM MMA (256×256) with leader/peer CTA coordination | SM100 |
| 9 | `09_blackwell_mma_tma_epi_sm100.cu` | TMA epilogue — C/D moved through smem via TMA | SM100 |

## Origin

Files 1–9 are copied verbatim from [NVIDIA/cutlass/examples/cute/tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial).
`example_utils.hpp` is the shared helper from the Blackwell tutorial directory and is used by files 5–9.

## Build

These compile against the top-level CMake project in this repository — see the root
[README](../README.md) for setup.

```bash
cd /path/to/mlinfra-puzzle
cmake -B build -GNinja
cmake --build build --target blog_01_tiled_copy
./build/blog/blog_01_tiled_copy
```

Build all blog examples at once:

```bash
cmake --build build --target blog_all
```

> **Note**: Each blog example requires a matching compute capability at build-and-run time. The
> CMake target will print a warning and skip if your GPU is too old. See the per-file `if (props.major ...)`
> guards at the top of each `main()`.
