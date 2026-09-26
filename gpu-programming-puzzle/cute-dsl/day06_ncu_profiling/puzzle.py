#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 06: fill the Nsight Compute report-analysis TODOs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


METRICS = {
    "duration": "gpu__time_duration.sum",
    "dram_read": "dram__bytes_read.sum",
    "dram_write": "dram__bytes_write.sum",
    "dram_throughput": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "global_load_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "global_store_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "global_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "global_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "load_sectors_per_request": (
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
    ),
    "store_sectors_per_request": (
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"
    ),
    "load_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    ),
    "store_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    ),
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers": "launch__registers_per_thread",
    "shared_memory": "launch__shared_mem_per_block_dynamic",
}


@dataclass(frozen=True)
class Profile:
    name: str
    kernel_name: str
    block_size: str
    grid_size: str
    duration_us: float
    dram_read_mb: float
    dram_write_mb: float
    dram_throughput_pct: float
    global_load_requests: float
    global_store_requests: float
    global_load_sectors: float
    global_store_sectors: float
    load_sectors_per_request: float
    store_sectors_per_request: float
    load_bank_conflicts: float
    store_bank_conflicts: float
    occupancy_pct: float
    registers_per_thread: float
    dynamic_shared_kb: float

    @property
    def total_bank_conflicts(self) -> float:
        # TODO(1): Return load plus store bank conflicts.
        raise NotImplementedError("Day 06 TODO(1): total bank conflicts")


def parse_number(text: str) -> float:
    # TODO(2): Strip whitespace and thousands separators, reject empty/N/A,
    # and return a float.
    raise NotImplementedError("Day 06 TODO(2): parse NCU numeric field")


def load_raw_csv(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))

    # TODO(3): Find the wide raw-page header containing both "Kernel Name" and
    # METRICS["duration"]. The next row contains units and the following
    # non-empty row contains kernel values. Return:
    #
    #   dict(zip(header, data)), dict(zip(header, units))
    #
    # Pad short units/data rows with empty strings before zipping.
    raise NotImplementedError("Day 06 TODO(3): parse NCU raw CSV")


def convert(value: float, unit: str, target: str) -> float:
    # TODO(4): Convert time to microseconds (`us`), byte counts to decimal
    # megabytes (`mb`), and shared memory to decimal kilobytes (`kb`).
    #
    # Strip suffixes such as `/block` before conversion.
    # Time units: ns, us, ms, s
    # Data units: byte, Kbyte, Mbyte, Gbyte
    raise NotImplementedError("Day 06 TODO(4): normalize metric units")


def load_profile(path: Path, name: str) -> Profile:
    values, units = load_raw_csv(path)

    def metric(key: str, target: str | None = None) -> float:
        column = METRICS[key]
        value = parse_number(values[column])
        return convert(value, units[column], target) if target else value

    return Profile(
        name=name,
        kernel_name=values["Kernel Name"],
        block_size=values.get("Block Size", ""),
        grid_size=values.get("Grid Size", ""),
        duration_us=metric("duration", "us"),
        dram_read_mb=metric("dram_read", "mb"),
        dram_write_mb=metric("dram_write", "mb"),
        dram_throughput_pct=metric("dram_throughput"),
        global_load_requests=metric("global_load_requests"),
        global_store_requests=metric("global_store_requests"),
        global_load_sectors=metric("global_load_sectors"),
        global_store_sectors=metric("global_store_sectors"),
        load_sectors_per_request=metric("load_sectors_per_request"),
        store_sectors_per_request=metric("store_sectors_per_request"),
        load_bank_conflicts=metric("load_conflicts"),
        store_bank_conflicts=metric("store_conflicts"),
        occupancy_pct=metric("occupancy"),
        registers_per_thread=metric("registers"),
        dynamic_shared_kb=metric("shared_memory", "kb"),
    )


def compare(unpadded: Profile, padded: Profile) -> dict[str, float]:
    # TODO(5): First reject mismatched block/grid geometry or DRAM traffic
    # differing by more than 10%. Then return:
    #   speedup = unpadded duration / padded duration
    #   conflict_reduction_pct
    #   dram_throughput_delta_points
    #   occupancy_delta_points
    #   shared_memory_delta_bytes
    #
    # Percent conflict reduction is:
    #   100 * (unpadded_total - padded_total) / unpadded_total
    raise NotImplementedError("Day 06 TODO(5): compare profiles")


def compare_naive(read: Profile, write: Profile) -> dict[str, float]:
    # TODO(6): Verify matching launch geometry, then return:
    #   write_speedup = read duration / write duration
    #   read_load_sectors_per_request
    #   read_store_sectors_per_request
    #   write_load_sectors_per_request
    #   write_store_sectors_per_request
    #
    # These four sector ratios show that thread remapping moves the scattered
    # direction from stores to loads rather than eliminating it.
    raise NotImplementedError("Day 06 TODO(6): compare naive mappings")


def main() -> int:
    parser = argparse.ArgumentParser()
    fixtures = Path(__file__).with_name("fixtures")
    parser.add_argument(
        "--naive-read",
        type=Path,
        default=fixtures / "naive_read.csv",
    )
    parser.add_argument(
        "--naive-write",
        type=Path,
        default=fixtures / "naive_write.csv",
    )
    parser.add_argument(
        "--unpadded",
        type=Path,
        default=fixtures / "shared_unpadded.csv",
    )
    parser.add_argument(
        "--padded",
        type=Path,
        default=fixtures / "shared_padded.csv",
    )
    parser.add_argument(
        "--swizzled",
        type=Path,
        default=fixtures / "shared_swizzled.csv",
    )
    args = parser.parse_args()

    naive_read = load_profile(args.naive_read, "naive coalesced read")
    naive_write = load_profile(args.naive_write, "naive coalesced write")
    unpadded = load_profile(args.unpadded, "unpadded stride 32")
    padded = load_profile(args.padded, "padded stride 33")
    swizzled = load_profile(args.swizzled, "swizzled S<5,0,5>")

    # TODO(7): Print all five profiles and evidence-based diagnoses:
    #
    # Challenge A:
    #   - compare_naive shows the 4/32 versus 32/4 sector swap;
    #   - explain why contiguous stores beat scattered stores.
    #
    # Challenge B:
    #   - same intended workload,
    #   - much lower total bank conflicts,
    #   - lower duration / higher DRAM utilization,
    #   - nearly unchanged occupancy and registers.
    #
    # Compare unpadded with both padded and swizzled. Also report that
    # swizzling avoids padding's extra shared-memory allocation.
    print(
        compare_naive(naive_read, naive_write),
        compare(unpadded, padded),
        compare(unpadded, swizzled),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
