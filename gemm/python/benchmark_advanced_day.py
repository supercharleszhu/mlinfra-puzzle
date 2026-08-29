#!/usr/bin/env python3
"""Build and run one flattened NVIDIA CuTe C++ tutorial lesson."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path


CUTLASS_TAG = "v4.6.1"
CUTLASS_COMMIT = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
BUILD_CACHE_VERSION = "cute-tutorial-v4"
ROOT = Path(__file__).resolve().parents[1]

LESSONS = {
    20: {
        "name": "CuTe SGEMM 1: tensor and thread partitioning",
        "config": "20_cute_sgemm_1.cu",
        "target": "cute_tutorial_sgemm_1",
        "source": "examples/cute/tutorial/sgemm_1.cu",
        "min_sm": 70,
        "success_marker": "CORRECTNESS: PASSED",
    },
    21: {
        "name": "CuTe SGEMM 2: TiledCopy and TiledMMA",
        "config": "21_cute_sgemm_2.cu",
        "target": "cute_tutorial_sgemm_2",
        "source": "examples/cute/tutorial/sgemm_2.cu",
        "min_sm": 70,
        "success_marker": "CUTE_GEMM:",
    },
    22: {
        "name": "CuTe SM80 SGEMM: tensor cores and cp.async pipeline",
        "config": "22_cute_sgemm_sm80.cu",
        "target": "cute_tutorial_sgemm_sm80",
        "source": "examples/cute/tutorial/sgemm_sm80.cu",
        "min_sm": 80,
        "success_marker": "CUTE_GEMM:",
    },
    23: {
        "name": "CuTe Hopper WGMMA with cp.async staging",
        "config": "23_cute_hopper_wgmma.cu",
        "target": "cute_tutorial_wgmma_sm90",
        "source": "examples/cute/tutorial/hopper/wgmma_sm90.cu",
        "exact_sm": 90,
        "success_marker": "CUTE_GEMM:",
    },
    24: {
        "name": "CuTe Hopper WGMMA with TMA pipeline",
        "config": "24_cute_hopper_wgmma_tma.cu",
        "target": "cute_tutorial_wgmma_tma_sm90",
        "source": "examples/cute/tutorial/hopper/wgmma_tma_sm90.cu",
        "exact_sm": 90,
        "success_marker": "CUTE_GEMM:",
    },
    25: {
        "name": "CuTe Blackwell UMMA and TMEM",
        "config": "25_cute_blackwell_mma.cu",
        "target": "cute_tutorial_01_mma_sm100",
        "source": "examples/cute/tutorial/blackwell/01_mma_sm100.cu",
        "exact_sm": 100,
        "success_marker": "Execution is successful.",
    },
    26: {
        "name": "CuTe Blackwell UMMA with TMA",
        "config": "26_cute_blackwell_mma_tma.cu",
        "target": "cute_tutorial_02_mma_tma_sm100",
        "source": "examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu",
        "exact_sm": 100,
        "success_marker": "Execution is successful.",
    },
    27: {
        "name": "CuTe Blackwell 2-SM UMMA and multicast TMA",
        "config": "27_cute_blackwell_mma_tma_2sm.cu",
        "target": "cute_tutorial_04_mma_tma_2sm_sm100",
        "source": "examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu",
        "exact_sm": 100,
        "success_marker": "Execution is successful.",
    },
    28: {
        "name": "CuTe Blackwell 2-SM GEMM with TMA epilogue",
        "config": "28_cute_blackwell_mma_tma_epilogue.cu",
        "target": "cute_tutorial_05_mma_tma_epi_sm100",
        "source": "examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu",
        "exact_sm": 100,
        "success_marker": "Execution is successful.",
    },
}


def cutlass_root(cli_value: str | None) -> Path:
    candidates = [
        Path(cli_value) if cli_value else None,
        Path(os.environ["CUTLASS_ADVANCED_DIR"])
        if os.environ.get("CUTLASS_ADVANCED_DIR")
        else None,
        ROOT / ".cache" / "cutlass-v4.6.1",
        ROOT / "official-cutlass",
    ]
    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Pinned CUTLASS sources were not found. Run "
        "scripts/setup_advanced_cutlass.sh or pass --cutlass-dir."
    )


def verify_cutlass_version(root: Path) -> None:
    marker = root / ".cutlass-advanced-commit"
    if marker.is_file():
        commit = marker.read_text().strip()
    elif (root / ".git").is_dir():
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    else:
        raise RuntimeError(f"{root} has no version marker or git metadata")
    if commit != CUTLASS_COMMIT:
        raise RuntimeError(
            f"Expected CUTLASS {CUTLASS_TAG} ({CUTLASS_COMMIT}), found {commit}"
        )


def config_path(day: int, source: str) -> Path:
    if source == "challenge":
        return ROOT / "cuda" / LESSONS[day]["config"]
    return ROOT / "solutions" / "cuda" / LESSONS[day]["config"]


def check_completion(path: Path) -> None:
    lines = path.read_text().splitlines()
    blanks = [
        f"{path}:{line_number}: {line.strip()}"
        for line_number, line in enumerate(lines, start=1)
        if "GEMM_TODO" in line
    ]
    if blanks:
        raise ValueError("Lesson still has blanks:\n  " + "\n  ".join(blanks))


def detect_sm() -> int:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as error:
        raise RuntimeError("Could not detect GPU compute capability") from error
    major, minor = output.strip().split(".", maxsplit=1)
    return int(major) * 10 + int(minor)


def architecture_skip(lesson: dict[str, object], sm: int) -> str | None:
    exact_sm = lesson.get("exact_sm")
    if exact_sm is not None and sm != exact_sm:
        return f"requires exactly SM{exact_sm}; current device is SM{sm}"
    min_sm = lesson.get("min_sm")
    if min_sm is not None and sm < min_sm:
        return f"requires SM{min_sm}+; current device is SM{sm}"
    return None


def compile_source(
    source: Path,
    cutlass: Path,
    helper_source: Path,
    sm: int,
    jobs: int,
    verbose: bool,
    use_fast_math: bool,
) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc was not found")

    local_headers = b"".join(
        header.read_bytes() for header in sorted((ROOT / "cuda").glob("*.cuh"))
    )
    digest = hashlib.sha1(
        source.read_bytes()
        + local_headers
        + str(source.resolve()).encode()
        + CUTLASS_COMMIT.encode()
        + BUILD_CACHE_VERSION.encode()
        + str(sm).encode()
    ).hexdigest()[:12]
    output_dir = ROOT / "build" / "advanced" / f"sm{sm}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source.stem}_{digest}"
    if output.is_file():
        return output

    arch = f"sm_{sm}a" if sm in {90, 100} else f"sm_{sm}"
    command = [
        nvcc,
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--threads",
        str(jobs),
        f"-arch={arch}",
        "-I",
        str(ROOT / "cuda"),
        "-I",
        str(cutlass / "include"),
        "-I",
        str(cutlass / "tools" / "util" / "include"),
        "-I",
        str(cutlass / "examples" / "common"),
        "-I",
        str(helper_source.parent),
    ]
    if use_fast_math:
        command.append("--use_fast_math")
    command.extend([str(source), "-o", str(output), "-lcublas"])
    if verbose:
        print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return output


def example_arguments(day: int, iterations: int) -> list[str]:
    arguments: dict[int, list[str]] = {
        20: ["512", "512", "512", "N", "T"],
        21: ["512", "512", "512", "N", "T"],
        22: ["2048", "2048", "2048", "T", "N"],
        23: ["4096", "4096", "4096", "N", "T"],
        24: ["4096", "4096", "4096", "N", "T"],
        25: [],
        26: [],
        27: [],
        28: [],
    }
    del iterations  # NVIDIA's tutorial mains use a fixed timing loop.
    return arguments[day]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True, type=int, choices=LESSONS)
    parser.add_argument(
        "--source", choices=["challenge", "solution"], default="challenge"
    )
    parser.add_argument("--cutlass-dir")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if args.iters < 1:
        parser.error("--iters must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    lesson = LESSONS[args.day]
    config = config_path(args.day, args.source)
    check_completion(config)
    root = cutlass_root(args.cutlass_dir)
    verify_cutlass_version(root)
    source = root / str(lesson["source"])
    if not source.is_file():
        raise FileNotFoundError(f"Official source is missing: {source}")

    print(f"Day {args.day}: {lesson['name']}")
    print(f"CUTLASS: {CUTLASS_TAG} ({CUTLASS_COMMIT[:12]})")
    print(f"Local source: {config}")
    print(f"Example helper directory: {source.parent}")
    print("Note: NVIDIA's tutorial main controls its own timing iteration count.")
    if args.dry_run:
        return 0

    sm = detect_sm()
    skip_reason = architecture_skip(lesson, sm)
    if skip_reason:
        print(f"SKIP: {skip_reason}.")
        return 0

    executable = compile_source(
        config,
        root,
        source,
        sm,
        args.jobs,
        args.verbose_build,
        False,
    )
    if args.build_only:
        print(f"Built: {executable}")
        return 0

    command = [str(executable), *example_arguments(args.day, args.iters)]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"{lesson['target']} exited with status {result.returncode}"
        )

    lowered = result.stdout.lower()
    failure_markers = (
        "disposition: failed",
        "[fail]",
        "failed validation",
        "verification failed",
        "execution is failed.",
        "error: failed",
        "not supported on",
        "requires a gpu",
    )
    for marker in failure_markers:
        if marker in lowered:
            raise RuntimeError(
                f"{lesson['target']} reported failure marker: {marker}"
            )

    success_marker = str(lesson["success_marker"])
    required_count = int(lesson.get("success_count", 1))
    actual_count = result.stdout.count(success_marker)
    if actual_count < required_count:
        raise RuntimeError(
            f"{lesson['target']} did not report {required_count} "
            f"expected success marker(s) {success_marker!r}; found {actual_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
