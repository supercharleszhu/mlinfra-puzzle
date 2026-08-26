# GEMM Challenge Blanks

Use this as your checklist. Each item corresponds to a `GEMM_TODO_*` placeholder in `cuda/`.

| File | Fill-in focus |
| --- | --- |
| `01_naive.cu` | Non-coalesced row/column mapping, dot-product load, alpha/beta epilogue. |
| `02_kernel_global_mem_coalesce.cu` | Coalesced row/column mapping for warp lanes. |
| `03_kernel_shared_mem.cu` | A/B global-to-shared loads and shared-memory dot product. |
| `04_kernel_blocktiling_1d.cu` | Tile loads, B register cache, TM per-thread output update. |
| `05_kernel_blocktiling_2d.cu` | A/B tile loads, register fragments, TMxTN outer-product FMA. |
| `06_kernel_vectorize.cu` | `float4` A/B loads and shared-to-register loads. |
| `07_kernel_warptiling.cu` | Warp subtile constants and warp-fragment loads. |
| `08_kernel_warptiling_all_dtypes.cu` | Dtype-aware accumulator initialization and type reasoning. |
| `09_kernel_tensorcore_naive.cu` | WMMA tile mapping, fragment loads, and `mma_sync`. |
| `10_kernel_tensorcore_warptiled.cu` | WMMA loads/MMA from tiled shared-memory fragments. |
| `11_kernel_tensorcore_double_buffered.cu` | Read/write buffer selection and buffer-flip pipeline. |
| `12_kernel_tensorcore_async.cu` | Async pipeline compute placeholder. |
| `13_kernel_fastcu_matmul2_manual_tma_wgmma.cu` | Handwritten Hopper matmul_2: 2D TMA loads, barriers, WGMMA m64n64k16, and C^T stores. |
| `14_kernel_fastcu_matmul3_big_tile.cu` | fast.cu matmul_3: larger BM/BN output tiles and wider WGMMA_N shapes. |
| `15_kernel_fastcu_matmul4_warp_specialized.cu` | fast.cu matmul_4: warp-specialized producer/consumer overlap for TMA loads and WGMMA compute. |
| `16_kernel_fastcu_matmul6_persistent.cu` | fast.cu matmul_6: persistent tile processing and steady-state overlap. |
| `17_kernel_fastcu_matmul10_tma_store.cu` | fast.cu matmul_10: accumulator conversion, stmatrix staging, and TMA store. |
| `18_kernel_fastcu_matmul12_final.cu` | fast.cu matmul_12: final combined handwritten kernel and benchmark analysis. |
| `19_kernel_cutlass3_hopper_tunable.cu` | CUTLASS 3 Hopper: tune TMA warp-specialized schedule, stage count, tile/cluster shape, raster order, decomposition, swizzle, and split count. |

## CuTe GEMM tutorials

These blanks are C++ `GEMM_TODO_*` values inside the complete flat
`cuda/20_*.cu` through `cuda/28_*.cu` implementations, with completed mirrors
under `solutions/cuda/`. Run
`python3 python/benchmark_advanced_day.py --day DAY --source challenge
--cutlass-dir .cache/cutlass-v4.6.1 --dry-run` to check them.

| Day | Blanks | What the blanks teach |
| --- | --- | --- |
| 20 | CTA M/N/K and copy/compute ThreadLayouts | How `local_tile` and `local_partition` distribute tensors across CTAs and threads. |
| 21 | CTA tile, TiledCopy thread layout, TiledMMA replication | How CuTe packages copy and compute instructions with logical partitions. |
| 22 | CTA K, SMEM stages, cp.async partition, MMA replication | How GMEM-to-SMEM, SMEM-to-register, and Tensor Core work overlap. |
| 23 | WGMMA K tile, stages, copy layout, wait depth | How a 128-thread warpgroup consumes SMEM descriptors asynchronously. |
| 24 | TMA coordinate, CTA tile, TMA stage count | How one elected lane and transaction barriers drive a circular TMA/WGMMA pipeline. |
| 25 | UMMA M/N, K depth, copy width, cluster | How Blackwell moves accumulators into TMEM and returns them to registers. |
| 26 | UMMA M/N, K depth, TMA coordinate, cluster | How TMA replaces cooperative copies while UMMA/TMEM remains unchanged. |
| 27 | 2-SM UMMA tile, K depth, cluster M/N | How peer CTAs construct multicast masks and share one UMMA operation. |
| 28 | 2-SM UMMA tile, K depth, epilogue subdivision, cluster | How tiled C-load and D-store TMA operations optimize the Blackwell epilogue. |
