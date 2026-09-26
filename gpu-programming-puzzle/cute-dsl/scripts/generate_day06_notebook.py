#!/usr/bin/env python3
"""Generate the explanation-rich Day 06 NCU analysis notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


DSL_ROOT = Path(__file__).resolve().parents[1]
DAY_ROOT = DSL_ROOT / "day06_ncu_profiling"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def puzzle_source() -> str:
    source = (DAY_ROOT / "puzzle.py").read_text(encoding="utf-8")
    return source[: source.index("\n\nif __name__")].strip()


def main() -> None:
    cells = [
        markdown(
            """
            # Day 06 - Nsight Compute profiling analysis

            Day 05 built five transpose variants. Day 06 asks two questions:

            > Why is the naive coalesced-write mapping much faster than the
            > naive coalesced-read mapping?

            > Do padding and `S<5,0,5>` swizzling remove the shared-memory bank
            > conflict in the partitioned Challenge B kernel?

            You will collect five controlled Nsight Compute reports and connect
            global sectors, shared bank conflicts, and duration to the layouts.
            """
        ),
        markdown(
            """
            ## 1. Begin with a falsifiable hypothesis

            Challenge A predicts:

            ```text
            naive-read:  few load sectors/request, many store sectors/request
            naive-write: many load sectors/request, few store sectors/request
            ```

            Challenge B predicts:

            ```text
            unpadded: (32,32):(32,1)          high shared-load conflicts
            padded:   (32,32):(33,1)          low shared-load conflicts
            swizzled: S<5,0,5> o (32,32)      low conflicts, no padding
            ```

            Before profiling Challenge B, predict:

            | Metric | Prediction |
            | --- | --- |
            | Shared-load bank conflicts | Large decrease |
            | Duration | Decrease |
            | DRAM throughput % | Increase because shared serialization falls |
            | Registers/thread | No material change |
            | Occupancy | Nearly unchanged |
            | Dynamic shared memory | Increase by about 124 bytes/CTA |

            This table makes the experiment falsifiable. If conflicts do not
            fall, either the bank model was wrong, the compiler changed the
            access, or the wrong kernel was captured.
            """
        ),
        markdown(
            """
            ## 2. Control every variable except the tested mapping

            A meaningful A/B profile requires:

            - identical matrix shape and dtype;
            - identical launch grid and block;
            - identical warmup and correctness behavior;
            - the same NCU version and metric list;
            - exactly one captured target launch;
            - no unrelated PyTorch kernels in the report.

            `profile_target.py` selects one maintained Day 05 JIT function per
            process. All variants use the same shape, dtype, and 256-thread
            block. Each process compiles and launches exactly one target kernel.
            """
        ),
        code(
            """
            from pathlib import Path

            day_root = Path.cwd()
            if not (day_root / "fixtures").exists():
                day_root = day_root / "day06_ncu_profiling"

            fixtures = {
                "naive read": day_root / "fixtures" / "naive_read.csv",
                "naive write": day_root / "fixtures" / "naive_write.csv",
                "shared unpadded": day_root / "fixtures" / "shared_unpadded.csv",
                "shared padded": day_root / "fixtures" / "shared_padded.csv",
                "shared swizzled": day_root / "fixtures" / "shared_swizzled.csv",
            }
            unpadded_path = fixtures["shared unpadded"]
            reports_dir = day_root / "reports"
            for name, path in fixtures.items():
                print(name, path)
            """
        ),
        markdown(
            """
            ## 3. Collect all five target kernels

            Run profiling from a terminal on the H100 pod, not inside a busy
            notebook kernel:

            ```bash
            cd /tmp/gpu-programming-puzzle-cute-dsl/day06_ncu_profiling
            PYTHON_BIN=../.venv/bin/python M=4096 N=4096 \
              bash collect_profiles.sh
            ```

            The helper creates:

            ```text
            naive_read.csv             naive_write.csv
            shared_unpadded.csv        shared_padded.csv
            shared_swizzled.csv
            ```

            The key NCU controls are:

            ```text
            --kernel-name regex:<matching naive/shared kernel>
            --launch-count 1
            --metrics <small explicit list>
            --force-overwrite
            ```

            Kernel filtering excludes allocations and PyTorch setup kernels.
            One launch keeps the report unambiguous. A small explicit metric
            list minimizes replay passes and collection overhead.
            """
        ),
        markdown(
            """
            ## 4. Know what NCU replay means

            Hardware counters cannot always be collected simultaneously. NCU
            may save state and replay a kernel for multiple passes. This has
            two consequences:

            1. Profiled duration is perturbed and is not your production
               benchmark.
            2. Metrics from one report still describe the same captured launch,
               provided replay is safe and deterministic.

            Transpose writes its complete output and has no cross-launch
            accumulation, so replay is safe. For kernels with atomics, random
            state, inter-kernel dependencies, or persistent side effects,
            replay requires additional care.
            """
        ),
        markdown(
            """
            ## 5. Export the raw page

            `.ncu-rep` is the authoritative report for the GUI. The challenge
            consumes a machine-readable raw page:

            ```bash
            ncu --import reports/shared_unpadded.ncu-rep --csv --page raw \
              > reports/shared_unpadded.csv
            ```

            NCU 2025 emits a **wide CSV**:

            1. header row with metric names;
            2. units row;
            3. one row per captured kernel.

            Do not hardcode column positions. NCU versions and requested metric
            sets can add columns. Find values by metric name.
            """
        ),
        code(
            """
            import csv

            rows = list(csv.reader(unpadded_path.read_text().splitlines()))
            header = rows[0]
            units = rows[1]
            data = rows[2]

            for metric in (
                "gpu__time_duration.sum",
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
                "launch__shared_mem_per_block_dynamic",
            ):
                index = header.index(metric)
                print(metric, data[index], units[index])
            """
        ),
        markdown(
            """
            ## 6. Normalize units before calculating

            NCU chooses human-readable units. A duration may be `ns`, `us`, or
            `ms`; memory can appear as `byte`, `Kbyte`, or `Mbyte`. Launch
            metrics can add suffixes such as `Kbyte/block`.

            Convert to one internal system:

            - duration: microseconds;
            - DRAM traffic: decimal megabytes;
            - shared storage: decimal kilobytes;
            - ratios: percent;
            - conflicts and registers: raw counts.

            Decimal prefixes match NCU's displayed units. Unit conversion is
            part of correctness: a 1000x mistake can produce a plausible-looking
            but false conclusion.
            """
        ),
        markdown(
            """
            ## 7. Metrics answer different questions

            | Metric family | Question |
            | --- | --- |
            | `gpu__time_duration` | Did the variant execute faster under the same collection? |
            | global requests and sectors | How many 32-byte sectors did each load/store request touch? |
            | average sectors/request | Which global direction is coalesced or scattered? |
            | shared bank conflicts | Did padding affect the intended mechanism? |
            | DRAM bytes | Did both reports perform comparable global work? |
            | DRAM throughput % | Could the kernel issue memory work more effectively? |
            | achieved occupancy | Did residency materially change? |
            | registers and shared memory | What resource tradeoff did padding introduce? |

            Duration is an outcome. Global sectors explain Challenge A; bank
            conflicts explain Challenge B. Occupancy and resources are controls.
            """
        ),
        markdown(
            """
            ## 8. Challenge: implement the report analyzer

            Fill seven TODOs:

            1. Compute total bank conflicts.
            2. Parse numeric fields including thousands separators.
            3. Locate raw CSV header, units, and data rows.
            4. Normalize units.
            5. Compute shared-memory speedup and metric/resource deltas.
            6. Compare the naive mappings through global sector ratios.
            7. Print diagnoses supported by several counters.

            Begin with the checked-in H100 fixtures, then point your analyzer at
            reports you collect yourself.
            """
        ),
        code(puzzle_source()),
        markdown(
            """
            ## 9. Run the completed reference analyzer

            This cell uses the H100 fixture snapshots so it is deterministic
            and does not require a live profiler.
            """
        ),
        code(
            """
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location(
                "day06_solution", day_root / "solution.py"
            )
            solution = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = solution
            spec.loader.exec_module(solution)

            profiles = {
                name: solution.load_profile(path, name)
                for name, path in fixtures.items()
            }
            for profile in profiles.values():
                solution.print_profile(profile)
            solution.print_naive_comparison(
                profiles["naive read"], profiles["naive write"]
            )
            solution.print_shared_comparison(
                profiles["shared unpadded"], profiles["shared padded"], "padding"
            )
            solution.print_shared_comparison(
                profiles["shared unpadded"],
                profiles["shared swizzled"],
                "swizzling",
            )
            """
        ),
        markdown(
            """
            ### Analyze your five live reports

            After running `collect_profiles.sh`, load Challenge A and Challenge
            B together. `print_naive_comparison` reports the speedup and the
            load/store sectors per request that identify the coalesced side.
            """
        ),
        code(
            """
            live_paths = {
                "naive read": reports_dir / "naive_read.csv",
                "naive write": reports_dir / "naive_write.csv",
                "shared unpadded": reports_dir / "shared_unpadded.csv",
                "shared padded": reports_dir / "shared_padded.csv",
                "shared swizzled": reports_dir / "shared_swizzled.csv",
            }
            if all(path.exists() for path in live_paths.values()):
                live = {
                    name: solution.load_profile(path, name)
                    for name, path in live_paths.items()
                }
                for profile in live.values():
                    solution.print_profile(profile)
                solution.print_naive_comparison(
                    live["naive read"], live["naive write"]
                )
            else:
                print("Run collect_profiles.sh to create all five live reports.")
            """
        ),
        markdown(
            """
            ## Why coalesced write can be much faster

            A warp's memory request is split into 32-byte sectors. Challenge A
            moves the scattered direction rather than removing it:

            ```text
            naive-read:  contiguous loads, scattered stores
            naive-write: scattered loads, contiguous stores
            ```

            NCU should show this swap in average sectors/request. Scattered
            stores are often more expensive: their sectors must be merged and
            committed through the cache/memory hierarchy, while strided loads
            can benefit from cache-line fetch and reuse behavior. The exact
            speedup is architecture- and shape-dependent, so treat sector
            counts plus duration as the evidence rather than assuming symmetry.
            """
        ),
        markdown(
            """
            ## Why `S<5,0,5>` matches the Float32 tile

            CuTe layouts map coordinates to **element offsets**. For a
            row-major `32 x 32` tile:

            ```text
            offset bits 0..4 = column
            offset bits 5..9 = row
            ```

            A Float32 word occupies one shared-memory bank unit, so the low
            five element-offset bits select one of 32 banks. `S<5,0,5>` XORs
            the five row bits into those five bank-selecting column bits. A
            transposed warp read therefore spreads across the banks without
            allocating a padded column.
            """
        ),
        markdown(
            """
            ## 10. Interpret the H100 evidence

            The captured example shows approximately:

            | Evidence | Unpadded | Padded | Swizzled |
            | --- | ---: | ---: | ---: |
            | Duration | 104.192 us | 44.256 us | 44.640 us |
            | Total bank conflicts | 16.37 M | 35.3 K | 32.2 K |
            | DRAM throughput | 33.60% | 77.73% | 77.21% |
            | Occupancy | 93.20% | 92.12% | 92.12% |
            | Registers/thread | 28 | 26 | 26 |
            | Dynamic shared memory | 4.096 KB | 4.220 KB | 4.096 KB |

            Challenge A's same capture shows:

            | Variant | Duration | Load sectors/request | Store sectors/request |
            | --- | ---: | ---: | ---: |
            | Coalesced read | 283.872 us | 4 | 32 |
            | Coalesced write | 88.864 us | 32 | 4 |

            This supports the causal chain:

            ```text
            stride 33
              -> lanes distribute across shared banks
              -> shared-load conflict counter collapses
              -> warps spend less time serializing shared reads
              -> DRAM pipeline stays busier
              -> kernel duration falls
            ```

            Store conflicts need not move identically to load conflicts. The
            transposed operation is the shared read, and total conflicts remain
            dominated by the large load reduction.
            """
        ),
        markdown(
            """
            ## 11. Sanity checks before accepting a comparison

            Reject or qualify the result if:

            - block or grid dimensions differ;
            - DRAM traffic differs substantially;
            - more than one kernel launch appears;
            - one report has unavailable metrics;
            - clocks or concurrent workloads changed drastically;
            - the profiled target failed correctness;
            - different NCU metric sets caused different replay behavior.

            Similar DRAM bytes do not need to be bit-identical because cache and
            compression behavior can vary. They should nevertheless represent
            the same order of work.
            """
        ),
        markdown(
            """
            ## 12. Profiling is not benchmarking

            Use NCU to answer **why**. Use Day 05's `testing.benchmark` path to
            answer **how fast in steady state**.

            NCU instrumentation can alter clocks, replay kernels, flush caches,
            and serialize execution. Its duration is valuable for controlled
            within-report comparisons, not as a replacement for application
            latency. A complete optimization result includes:

            1. correctness on edge shapes;
            2. steady-state benchmark with warmup;
            3. targeted profiler evidence;
            4. resource and regression checks.
            """
        ),
        markdown(
            """
            ## 13. Extension experiments

            After padding, profile one change at a time:

            1. **Different swizzles:** derive bit mappings, then compare conflicts.
            2. **128-bit global copies:** inspect global sectors and duration.
            3. **Different tile sizes:** compare occupancy, waves, and conflicts.
            4. **Irregular shapes:** quantify predicate and partial-tile cost.
            5. **Roofline/SOL sections:** determine whether the optimized kernel
               is now limited by DRAM, L2, L1/shared, or instruction issue.

            Never begin with `--set full` unless you know why every section is
            needed. Broad collection is slow and produces more data than a
            focused hypothesis requires.
            """
        ),
        markdown(
            """
            ## What you should know afterward

            - Form a metric prediction before opening the profiler.
            - Compare controlled kernel variants with matching launch geometry.
            - Filter NCU to the target kernel and request a small metric set.
            - Treat raw CSV units as data, not decoration.
            - Use mechanism counters, outcome metrics, and resource controls together.
            - Distinguish NCU replay duration from steady-state benchmark time.
            - Preserve `.ncu-rep` for interactive inspection and CSV for
              reproducible analysis.
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "CuTe DSL (H100)",
                "language": "python",
                "name": "cutlass-cute-dsl",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = DAY_ROOT / "day06.ipynb"
    destination.write_text(
        json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
