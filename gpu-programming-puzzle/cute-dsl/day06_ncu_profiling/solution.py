#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Day 06: analyze and compare Nsight Compute raw CSV exports."""

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
    global_load_requests: float | None
    global_store_requests: float | None
    global_load_sectors: float | None
    global_store_sectors: float | None
    load_sectors_per_request: float | None
    store_sectors_per_request: float | None
    load_bank_conflicts: float
    store_bank_conflicts: float
    occupancy_pct: float
    registers_per_thread: float
    dynamic_shared_kb: float

    @property
    def total_bank_conflicts(self) -> float:
        return self.load_bank_conflicts + self.store_bank_conflicts


def parse_number(text: str) -> float:
    value = text.strip().replace(",", "")
    if value.lower() in {"", "n/a", "nan"}:
        raise ValueError(f"metric has no numeric value: {text!r}")
    return float(value)


def convert(value: float, unit: str, target: str) -> float:
    normalized = unit.strip().lower().split("/", 1)[0]
    if target == "us":
        factors = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
    elif target == "mb":
        factors = {
            "byte": 1e-6,
            "kbyte": 1e-3,
            "mbyte": 1.0,
            "gbyte": 1e3,
        }
    elif target == "kb":
        factors = {"byte": 1e-3, "kbyte": 1.0, "mbyte": 1e3}
    else:
        raise ValueError(f"unsupported target unit: {target}")
    if normalized not in factors:
        raise ValueError(f"cannot convert {unit!r} to {target}")
    return value * factors[normalized]


def load_raw_csv(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return one NCU raw page as column->value and column->unit mappings."""
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if "Kernel Name" in row and METRICS["duration"] in row
        ),
        None,
    )
    if header_index is None:
        raise ValueError(
            f"{path}: no Nsight Compute raw header; export with "
            "`ncu --import REPORT --csv --page raw`"
        )
    if header_index + 2 >= len(rows):
        raise ValueError(f"{path}: expected units and data after raw header")

    header = rows[header_index]
    units = rows[header_index + 1]
    data = next(
        (
            row
            for row in rows[header_index + 2 :]
            if row and any(field.strip() for field in row)
        ),
        None,
    )
    if data is None:
        raise ValueError(f"{path}: no kernel data row")

    if len(units) < len(header):
        units.extend([""] * (len(header) - len(units)))
    if len(data) < len(header):
        data.extend([""] * (len(header) - len(data)))
    return dict(zip(header, data)), dict(zip(header, units))


def read_metric(
    values: dict[str, str],
    units: dict[str, str],
    metric: str,
    target_unit: str | None = None,
) -> float:
    column = METRICS[metric]
    if column not in values:
        raise ValueError(f"report is missing required metric {column}")
    value = parse_number(values[column])
    return convert(value, units.get(column, ""), target_unit) if target_unit else value


def read_optional_metric(
    values: dict[str, str],
    units: dict[str, str],
    metric: str,
) -> float | None:
    column = METRICS[metric]
    if column not in values or values[column].strip().lower() in {"", "n/a", "nan"}:
        return None
    return parse_number(values[column])


def load_profile(path: Path, name: str) -> Profile:
    values, units = load_raw_csv(path)
    return Profile(
        name=name,
        kernel_name=values["Kernel Name"],
        block_size=values.get("Block Size", ""),
        grid_size=values.get("Grid Size", ""),
        duration_us=read_metric(values, units, "duration", "us"),
        dram_read_mb=read_metric(values, units, "dram_read", "mb"),
        dram_write_mb=read_metric(values, units, "dram_write", "mb"),
        dram_throughput_pct=read_metric(values, units, "dram_throughput"),
        global_load_requests=read_optional_metric(
            values, units, "global_load_requests"
        ),
        global_store_requests=read_optional_metric(
            values, units, "global_store_requests"
        ),
        global_load_sectors=read_optional_metric(
            values, units, "global_load_sectors"
        ),
        global_store_sectors=read_optional_metric(
            values, units, "global_store_sectors"
        ),
        load_sectors_per_request=read_optional_metric(
            values, units, "load_sectors_per_request"
        ),
        store_sectors_per_request=read_optional_metric(
            values, units, "store_sectors_per_request"
        ),
        load_bank_conflicts=read_metric(values, units, "load_conflicts"),
        store_bank_conflicts=read_metric(values, units, "store_conflicts"),
        occupancy_pct=read_metric(values, units, "occupancy"),
        registers_per_thread=read_metric(values, units, "registers"),
        dynamic_shared_kb=read_metric(values, units, "shared_memory", "kb"),
    )


def percent_reduction(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError("baseline must be positive for percent reduction")
    return 100.0 * (before - after) / before


def compare(unpadded: Profile, padded: Profile) -> dict[str, float]:
    if (unpadded.block_size, unpadded.grid_size) != (
        padded.block_size,
        padded.grid_size,
    ):
        raise ValueError("profiles use different launch geometries")

    unpadded_dram = unpadded.dram_read_mb + unpadded.dram_write_mb
    padded_dram = padded.dram_read_mb + padded.dram_write_mb
    dram_work_delta_pct = 100.0 * abs(padded_dram - unpadded_dram) / unpadded_dram
    if dram_work_delta_pct > 10.0:
        raise ValueError(
            "DRAM traffic differs by more than 10%; profiles may not represent "
            "the same workload"
        )

    return {
        "speedup": unpadded.duration_us / padded.duration_us,
        "conflict_reduction_pct": percent_reduction(
            unpadded.total_bank_conflicts,
            padded.total_bank_conflicts,
        ),
        "dram_throughput_delta_points": (
            padded.dram_throughput_pct - unpadded.dram_throughput_pct
        ),
        "occupancy_delta_points": padded.occupancy_pct - unpadded.occupancy_pct,
        "shared_memory_delta_bytes": (
            padded.dynamic_shared_kb - unpadded.dynamic_shared_kb
        )
        * 1_000,
        "dram_work_delta_pct": dram_work_delta_pct,
    }


def compare_naive(read: Profile, write: Profile) -> dict[str, float]:
    if (read.block_size, read.grid_size) != (
        write.block_size,
        write.grid_size,
    ):
        raise ValueError("profiles use different launch geometries")

    read_load = read.load_sectors_per_request
    read_store = read.store_sectors_per_request
    write_load = write.load_sectors_per_request
    write_store = write.store_sectors_per_request
    if (
        read_load is None
        or read_store is None
        or write_load is None
        or write_store is None
    ):
        raise ValueError("naive profiles are missing global sector metrics")

    return {
        "write_speedup": read.duration_us / write.duration_us,
        "read_load_sectors_per_request": read_load,
        "read_store_sectors_per_request": read_store,
        "write_load_sectors_per_request": write_load,
        "write_store_sectors_per_request": write_store,
    }


def print_profile(profile: Profile) -> None:
    print(f"\n{profile.name}")
    print(f"  launch: block={profile.block_size}, grid={profile.grid_size}")
    print(f"  duration: {profile.duration_us:.3f} us")
    print(f"  DRAM throughput: {profile.dram_throughput_pct:.2f}% of peak")
    if all(
        value is not None
        for value in (
            profile.load_sectors_per_request,
            profile.store_sectors_per_request,
            profile.global_load_sectors,
            profile.global_store_sectors,
        )
    ):
        print(
            "  global sectors/request: "
            f"load={profile.load_sectors_per_request:.2f}, "
            f"store={profile.store_sectors_per_request:.2f}"
        )
        print(
            "  global sectors: "
            f"load={profile.global_load_sectors:,.0f}, "
            f"store={profile.global_store_sectors:,.0f}"
        )
    print(
        "  bank conflicts: "
        f"load={profile.load_bank_conflicts:,.0f}, "
        f"store={profile.store_bank_conflicts:,.0f}, "
        f"total={profile.total_bank_conflicts:,.0f}"
    )
    print(f"  occupancy: {profile.occupancy_pct:.2f}%")
    print(f"  registers/thread: {profile.registers_per_thread:.0f}")
    print(f"  dynamic shared memory: {profile.dynamic_shared_kb:.3f} KB")


def print_naive_comparison(read: Profile, write: Profile) -> None:
    results = compare_naive(read, write)
    print("\nChallenge A: coalesced-read versus coalesced-write")
    print(f"  coalesced-write speedup: {results['write_speedup']:.2f}x")
    print(
        "  coalesced-read sectors/request: "
        f"load={results['read_load_sectors_per_request']:.2f}, "
        f"store={results['read_store_sectors_per_request']:.2f}\n"
        "  coalesced-write sectors/request: "
        f"load={results['write_load_sectors_per_request']:.2f}, "
        f"store={results['write_store_sectors_per_request']:.2f}"
    )
    print(
        "  diagnosis: thread remapping moves the expensive strided direction. "
        "The faster variant makes stores contiguous, sharply reducing store "
        "sectors/request; scattered stores are harder to merge and commit than "
        "the corresponding strided loads."
    )


def print_shared_comparison(
    baseline: Profile,
    optimized: Profile,
    optimization: str,
) -> None:
    results = compare(baseline, optimized)
    print(f"\nChallenge B: unpadded versus {optimization}")
    print(f"  speedup: {results['speedup']:.2f}x")
    print(
        "  total bank-conflict reduction: "
        f"{results['conflict_reduction_pct']:.2f}%"
    )
    print(
        "  DRAM throughput change: "
        f"{results['dram_throughput_delta_points']:+.2f} percentage points"
    )
    print(
        "  occupancy change: "
        f"{results['occupancy_delta_points']:+.2f} percentage points"
    )
    print(
        "  dynamic shared-memory cost: "
        f"{results['shared_memory_delta_bytes']:+.0f} bytes/CTA"
    )

    if (
        results["conflict_reduction_pct"] > 90
        and results["speedup"] > 1.1
        and abs(results["occupancy_delta_points"]) < 5
    ):
        print(
            f"  diagnosis: {optimization} removed the dominant shared-load "
            "bank conflict and improved throughput without materially "
            "changing occupancy."
        )
    else:
        print(
            f"  diagnosis: the expected {optimization} signature is not "
            "decisive; inspect launch matching, metric availability, and "
            "run-to-run variance."
        )


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

    print_profile(naive_read)
    print_profile(naive_write)
    print_profile(unpadded)
    print_profile(padded)
    print_profile(swizzled)
    print_naive_comparison(naive_read, naive_write)
    print_shared_comparison(unpadded, padded, "padding")
    print_shared_comparison(unpadded, swizzled, "swizzling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
