# Day 06 - Nsight Compute Profiling Analysis

**Goal:** collect comparable Nsight Compute reports for all five Day 05
variants, then explain both why coalesced writes beat coalesced reads and why
padded/swizzled shared memory beats the unpadded shared tile.

This is an evidence-driven optimization lab:

```text
hypothesis -> controlled variants -> profile -> compare -> explain
```

The profiling target exposes:

```text
Challenge A
  naive-read          coalesced global loads, scattered global stores
  naive-write         scattered global loads, coalesced global stores

Challenge B
  shared-unpadded     shared stride 32
  shared-padded       shared stride 33
  shared-swizzled     S<5,0,5> over stride 32
```

Problem shape, thread block, grid, input/output types, and instructions around
the shared layout remain the same.

## 1. Collect reports on the H100 pod

From the Day 06 directory:

```bash
PYTHON_BIN=../.venv/bin/python \
M=4096 N=4096 \
bash collect_profiles.sh
```

In the configured remote workspace:

```bash
cd /tmp/gpu-programming-puzzle-cute-dsl/day06_ncu_profiling
PYTHON_BIN=../.venv/bin/python bash collect_profiles.sh
```

The helper creates `.ncu-rep` and `.csv` files for:

```text
naive_read, naive_write,
shared_unpadded, shared_padded, shared_swizzled
```

Open `.ncu-rep` files in the Nsight Compute UI for interactive inspection. The
CSV files come from:

```bash
ncu --import REPORT.ncu-rep --csv --page raw
```

NCU may require GPU performance-counter permission. A system that reports
`ERR_NVGPUCTRPERM` must be configured by its administrator; do not work around
that restriction.

## 2. Metrics

| Evidence | Metric |
| --- | --- |
| Kernel duration | `gpu__time_duration.sum` |
| DRAM traffic | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| DRAM utilization | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| Global load/store requests | `l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st}.sum` |
| Global load/store sectors | `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum` |
| Coalescing quality | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_{ld,st}.ratio` |
| Shared load conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` |
| Shared store conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Register pressure | `launch__registers_per_thread` |
| Shared-memory cost | `launch__shared_mem_per_block_dynamic` |

The collection command selects the matching naive or shared kernel name and
captures one launch. This prevents PyTorch setup kernels from entering the
comparison.

For Challenge A, compare load and store sectors/request. Both variants have one
bad global direction, but scattered stores are especially costly because many
sectors must be merged and committed. The expected signature is:

```text
naive-read:  low load sectors/request, high store sectors/request
naive-write: high load sectors/request, low store sectors/request
```

## 3. Analysis challenge

Fill seven TODOs in `puzzle.py`:

1. Sum load and store conflicts.
2. Parse NCU number fields safely.
3. Find the wide raw CSV header, units, and kernel data row.
4. Normalize time and data units.
5. Calculate shared-memory speedup and resource/metric deltas.
6. Compare the naive mappings through their load/store sectors per request.
7. Print evidence-based diagnoses for all five profiles.

Start with the checked-in H100 snapshots:

```bash
python puzzle.py
python solution.py
```

Then analyze your own collection:

```bash
python solution.py \
  --naive-read reports/naive_read.csv \
  --naive-write reports/naive_write.csv \
  --unpadded reports/shared_unpadded.csv \
  --padded reports/shared_padded.csv \
  --swizzled reports/shared_swizzled.csv
```

The fixtures are a reproducible example, not a performance guarantee. Timing,
clocking, driver versions, and concurrent system activity can change exact
values.

## 4. Interpretation discipline

A useful shared-memory optimization explanation must connect four levels:

1. **Source:** row stride changes from 32 to 33.
2. **Address mapping:** a transposed warp read changes from one bank to 32 banks.
3. **Counter:** shared-load bank conflicts fall substantially.
4. **Outcome:** duration falls and DRAM utilization rises.

Occupancy and registers act as controls. If they remain nearly unchanged, the
speedup is unlikely to come from a different occupancy regime.

Do not infer causality from duration alone. Also verify matching block/grid
geometry and similar DRAM traffic, proving the reports represent the same
logical work.

The checked-in H100 snapshots show:

| Variant | Duration | Load/store sectors per request | Bank conflicts |
| --- | ---: | ---: | ---: |
| Naive coalesced read | 283.872 us | 4 / 32 | 0 |
| Naive coalesced write | 88.864 us | 32 / 4 | 0 |
| Shared unpadded | 104.192 us | 4 / 4 | 16.37 M |
| Shared padded | 44.256 us | 4 / 4 | 35.3 K |
| Shared `S<5,0,5>` | 44.640 us | 4 / 4 | 32.2 K |

Thus coalesced write is 3.19x faster than coalesced read in this capture.
Padding and swizzling reduce shared conflicts by about 99.8%; swizzling reaches
the same performance regime without padding's extra 124 bytes per CTA.

## 5. Profiling versus benchmarking

NCU may replay a kernel once per metric pass and perturbs execution while
collecting counters. Use NCU to explain bottlenecks and a normal benchmark to
report final application performance. Compare durations collected with the
same NCU metric set, but do not substitute profiled time for Day 05's
steady-state benchmark.
