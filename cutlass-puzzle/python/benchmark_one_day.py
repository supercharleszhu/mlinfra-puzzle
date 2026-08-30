"""Build and benchmark one GEMM hard-way challenge day."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Callable

import torch


DAY_SPECS = {
    1: ("01_naive.cu", "float32", "01_naive", "sgemm_naive", "fp32_out"),
    2: ("02_kernel_global_mem_coalesce.cu", "float32", "02_coalesced", "sgemm_global_mem_coalesce", "fp32_out"),
    3: ("03_kernel_shared_mem.cu", "float32", "03_shared_mem", "sgemm_shared_mem", "fp32_out"),
    4: ("04_kernel_blocktiling_1d.cu", "float32", "04_blocktiling_1d", "sgemm_blocktiling_1d", "fp32_out"),
    5: ("05_kernel_blocktiling_2d.cu", "float32", "05_blocktiling_2d", "sgemm_blocktiling_2d", "fp32_out"),
    6: ("06_kernel_vectorize.cu", "float32", "06_vectorize", "sgemm_vectorize", "fp32_out"),
    7: ("07_kernel_warptiling.cu", "float32", "07_warptiling", "sgemm_warptiling_default", "same_dtype_out"),
    8: ("08_kernel_warptiling_all_dtypes.cu", "float16", "08_warptiling_fp16", "sgemm_warptiling_fp16", "same_dtype_out"),
    9: ("09_kernel_tensorcore_naive.cu", "float16", "09_tensorcore_naive_fp16", "sgemm_tensorcore_naive_fp16", "fp32_then_cast"),
    10: ("10_kernel_tensorcore_warptiled.cu", "float16", "10_tensorcore_fp16", "sgemm_tensorcore_fp16", "fp32_then_cast"),
    11: ("11_kernel_tensorcore_double_buffered.cu", "float16", "11_tensorcore_db_fp16", "sgemm_tensorcore_double_buffered_fp16", "fp32_then_cast"),
    12: ("12_kernel_tensorcore_async.cu", "float16", "12_tensorcore_async_fp16", "sgemm_tensorcore_async_fp16", "fp32_then_cast"),
    13: ("13_kernel_fastcu_matmul2_manual_tma_wgmma.cu", "bfloat16", "13_fastcu_matmul2_manual_tma_wgmma_bf16", "sgemm_fastcu_matmul2_manual_tma_wgmma_bf16", "fastcu_handwritten_bf16_out"),
    14: ("14_kernel_fastcu_matmul3_big_tile.cu", "bfloat16", "14_fastcu_matmul3_big_tile_bf16", "sgemm_fastcu_handwritten_tma_wgmma_bf16", "fastcu_handwritten_bf16_out"),
    15: ("15_kernel_fastcu_matmul4_warp_specialized.cu", "bfloat16", "15_fastcu_matmul4_warp_specialized_bf16", "sgemm_fastcu_handwritten_cached_tma_maps_bf16", "fastcu_cached_bf16_out"),
    16: ("16_kernel_fastcu_matmul6_persistent.cu", "bfloat16", "16_fastcu_matmul6_persistent_bf16", "sgemm_fastcu_final_bf16", "fastcu_cached_bf16_out"),
    17: ("17_kernel_fastcu_matmul10_tma_store.cu", "bfloat16", "17_fastcu_matmul10_tma_store_bf16", "sgemm_fastcu_tma_store_bf16", "fastcu_cached_bf16_out"),
    18: ("18_kernel_fastcu_matmul12_final.cu", "bfloat16", "18_fastcu_matmul12_final_bf16", "sgemm_fastcu_hilbert_final_bf16", "fastcu_cached_bf16_out"),
    19: ("19_kernel_cutlass3_hopper_tunable.cu", "bfloat16", "19_cutlass3_hopper_tunable_bf16", "sgemm_cutlass3_hopper_tunable_bf16", "bf16_out"),
}

EXPLORE_GEMM_SIZES = [64, 96, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192]
_CACHE_FLUSH_SIZE = 4 * 1024 * 1024
_cache_flush_buffer: torch.Tensor | None = None
_fastcu_b_transpose_cache: dict[tuple[int, tuple[int, ...], torch.dtype], torch.Tensor] = {}
_fastcu_c_transpose_cache: dict[tuple[int, int, int, torch.dtype], torch.Tensor] = {}


def read_without_local_includes(path: Path) -> str:
    lines = []
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("#include"):
            continue
        if line.startswith("#pragma once"):
            continue
        lines.append(line)
    return "".join(lines)


def find_cutlass_paths(root: Path) -> list[str]:
    candidates = []
    if os.environ.get("CUTLASS_DIR"):
        candidates.append(Path(os.environ["CUTLASS_DIR"]))
    if os.environ.get("CUTLASS_INCLUDE_DIR"):
        candidates.append(Path(os.environ["CUTLASS_INCLUDE_DIR"]).parent)
    candidates.extend(
        [
            root / "cutlass",
            root / "solutions" / "third-party" / "cutlass",
            root.parent / "LeetCUDA" / "cutlass",
        ]
    )

    for base in candidates:
        include = base / "include"
        util_include = base / "tools" / "util" / "include"
        if (include / "cutlass").exists():
            paths = [str(include)]
            if util_include.exists():
                paths.append(str(util_include))
            return paths
    return []


def torch_include_paths() -> list[Path]:
    torch_root = Path(torch.__file__).resolve().parent
    return [
        Path(sysconfig.get_paths()["include"]),
        torch_root / "include",
        torch_root / "include" / "torch" / "csrc" / "api" / "include",
    ]


def torch_library_path() -> Path:
    return Path(torch.__file__).resolve().parent / "lib"


def import_shared_object(module_name: str, shared_object: Path):
    spec = importlib.util.spec_from_file_location(module_name, shared_object)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import compiled module: {shared_object}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_extension(day: int, root: Path, source_dir: Path, arch: str, verbose: bool):
    file_name, _, _, function_name, _ = DAY_SPECS[day]
    cuda_dir = root / "cuda"
    source_path = source_dir / file_name
    source_text = source_path.read_text()
    source_hash = hashlib.sha1(source_text.encode("utf-8")).hexdigest()[:10]

    module_name = f"gemm_hard_way_day{day:02d}_{source_hash}"
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    build_dir = root / "build" / f"day{day:02d}_{source_hash}_nvcc"
    generated_source = build_dir / f"{module_name}.cu"
    shared_object = build_dir / f"{module_name}{extension_suffix}"
    build_dir.mkdir(parents=True, exist_ok=True)

    if shared_object.exists() and shared_object.stat().st_mtime >= source_path.stat().st_mtime:
        return import_shared_object(module_name, shared_object)

    cuda_header = r"""
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif

#include <cassert>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <torch/extension.h>
namespace cg = cooperative_groups;
"""

    extra_include_paths: list[str] = []
    functions = [function_name]
    if day >= 13:
        extra_include_paths = find_cutlass_paths(root)
        if not extra_include_paths:
            raise RuntimeError("CUTLASS headers were not found. Set CUTLASS_DIR or upload with --with-cutlass.")
        cuda_header += r"""
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
"""
    header_code = read_without_local_includes(cuda_dir / "gemm_kernels.cuh")
    utils_code = read_without_local_includes(cuda_dir / "utils.cuh")
    todo_code = read_without_local_includes(cuda_dir / "challenge_todo.cuh")
    cuda_sources = (
        todo_code + "\n" + utils_code + "\n" +
        read_without_local_includes(source_path)
    )
    bindings = "\n".join(f'    m.def("{name}", &{name});' for name in functions)
    generated_source.write_text(
        cuda_header
        + "\n"
        + header_code
        + "\n"
        + cuda_sources
        + f"""

PYBIND11_MODULE({module_name}, m) {{
{bindings}
}}
""",
    )

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc was not found on PATH")

    abi_flag = int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False))
    torch_lib = torch_library_path()
    cmd = [
        nvcc,
        "-shared",
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-Xcompiler",
        "-fPIC",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
        f"-DPYBIND11_MODULE_NAME={module_name}",
        "-o",
        str(shared_object),
        str(generated_source),
    ]
    if arch.endswith("a"):
        compute = f"compute_{arch[3:]}"
        cmd.extend(["-gencode", f"arch={compute},code={arch}"])
    else:
        cmd.append(f"-arch={arch}")
    for include_path in torch_include_paths():
        cmd.extend(["-I", str(include_path)])
    for include_path in extra_include_paths:
        cmd.extend(["-I", include_path])
    cmd.extend(
        [
            "-L",
            str(torch_lib),
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            str(torch_lib),
            "-lc10",
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_python",
            "-lc10_cuda",
            "-ltorch_cuda",
            "-lcuda",
        ]
    )

    if verbose:
        print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return import_shared_object(module_name, shared_object)


def fp32_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(a, b, c, 1.0, 0.0)
    return c


def same_dtype_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    kernel(a, b, c, 1.0, 0.0)
    return c


def fp32_then_cast(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(a, b, c, 1.0, 0.0)
    return c.to(a.dtype)


def autotune_fp32_then_cast(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(0, a, b, c, 1.0, 0.0)
    return c.to(a.dtype)


def bf16_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)
    kernel(a, b, c)
    return c


def hopper_autotune_bf16_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)
    kernel(0, 2, 0, 1, 1, a, b, c)
    return c


def fastcu_handwritten_bf16_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    key = (b.data_ptr(), tuple(b.shape), b.dtype)
    b_t = _fastcu_b_transpose_cache.get(key)
    if b_t is None:
        b_t = b.t().contiguous()
        _fastcu_b_transpose_cache[key] = b_t
    c_t = torch.empty((b.shape[1], a.shape[0]), device=a.device, dtype=torch.bfloat16)
    kernel(a, b_t, c_t)
    return c_t.t()


def fastcu_cached_bf16_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b_key = (b.data_ptr(), tuple(b.shape), b.dtype)
    b_t = _fastcu_b_transpose_cache.get(b_key)
    if b_t is None:
        b_t = b.t().contiguous()
        _fastcu_b_transpose_cache[b_key] = b_t

    c_key = (a.data_ptr(), b.data_ptr(), b.shape[1], torch.bfloat16)
    c_t = _fastcu_c_transpose_cache.get(c_key)
    if c_t is None:
        c_t = torch.empty((b.shape[1], a.shape[0]), device=a.device, dtype=torch.bfloat16)
        _fastcu_c_transpose_cache[c_key] = c_t
    kernel(a, b_t, c_t)
    return c_t.t()


def flush_l2_cache() -> None:
    global _cache_flush_buffer
    if _cache_flush_buffer is None:
        _cache_flush_buffer = torch.empty(_CACHE_FLUSH_SIZE, dtype=torch.int8, device="cuda")
    _cache_flush_buffer.zero_()
    torch.cuda.synchronize()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def time_kernel(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
    flush_cache: bool,
) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        if flush_cache:
            flush_l2_cache()
        out = fn()

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for idx in range(iters):
        if flush_cache:
            flush_l2_cache()
        start_events[idx].record()
        out = fn()
        end_events[idx].record()
    torch.cuda.synchronize()

    times_ms = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    trim_count = max(1, iters // 10)
    if len(times_ms) > 2 * trim_count:
        times_ms = sorted(times_ms)[trim_count:-trim_count]
    return out, median(times_ms)


def default_tolerance(dtype_name: str) -> float:
    if dtype_name == "float32":
        return 1e-3
    if dtype_name == "float16":
        return 1e-1
    return 1.0


def default_arch(day: int) -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if day >= 13:
        if major < 9:
            raise RuntimeError("Days 13-19 require a Hopper GPU (SM90+) and should be built for sm_90a.")
        return "sm_90a"
    return f"sm_{major}{minor}"


def default_sizes(day: int, dtype_name: str) -> list[int]:
    return EXPLORE_GEMM_SIZES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, required=True, choices=sorted(DAY_SPECS))
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-flush-cache", dest="flush_cache", action="store_false")
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    root = Path(__file__).resolve().parents[1]
    source_dir = args.source_dir or root / "cuda"
    _, default_dtype, kernel_label, function_name, wrapper_name = DAY_SPECS[args.day]
    dtype_name = args.dtype or default_dtype
    sizes = args.sizes or default_sizes(args.day, dtype_name)
    tolerance = args.tolerance if args.tolerance is not None else default_tolerance(dtype_name)

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]

    arch = args.arch or default_arch(args.day)
    ext = create_extension(args.day, root, source_dir, arch, args.verbose_build)
    kernel = getattr(ext, function_name)
    wrapper = {
        "fp32_out": fp32_out,
        "same_dtype_out": same_dtype_out,
        "fp32_then_cast": fp32_then_cast,
        "autotune_fp32_then_cast": autotune_fp32_then_cast,
        "bf16_out": bf16_out,
        "hopper_autotune_bf16_out": hopper_autotune_bf16_out,
        "fastcu_handwritten_bf16_out": fastcu_handwritten_bf16_out,
        "fastcu_cached_bf16_out": fastcu_cached_bf16_out,
    }[wrapper_name]

    worst_error = 0.0
    print("| kernel | size | run | ms | TFLOPS | max_error |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    summaries = []
    for size in sizes:
        a = torch.randn((size, size), device="cuda", dtype=dtype)
        b = torch.randn((size, size), device="cuda", dtype=dtype)
        ref = torch.matmul(a, b)
        ops = 2.0 * size * size * size
        run_results = []
        for run_idx in range(1, args.runs + 1):
            out, ms = time_kernel(lambda: wrapper(kernel, a, b), args.warmup, args.iters, args.flush_cache)
            err = (out.float() - ref.float()).abs().max().item()
            tflops = ops / (ms * 1e9)
            worst_error = max(worst_error, err)
            run_results.append((ms, tflops, err))
            print(f"| {kernel_label} | {size} | {run_idx} | {ms:.4f} | {tflops:.2f} | {err:.4e} |")
        best_ms = min(result[0] for result in run_results)
        avg_ms = sum(result[0] for result in run_results) / len(run_results)
        ms_stddev = math.sqrt(sum((result[0] - avg_ms) ** 2 for result in run_results) / len(run_results))
        avg_tflops = sum(result[1] for result in run_results) / len(run_results)
        best_tflops = max(result[1] for result in run_results)
        max_error = max(result[2] for result in run_results)
        summaries.append((size, best_ms, avg_ms, ms_stddev, best_tflops, avg_tflops, max_error))

    if args.runs > 1:
        print()
        print("| kernel | size | best_ms | avg_ms | stddev_ms | best_TFLOPS | avg_TFLOPS | max_error |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for size, best_ms, avg_ms, ms_stddev, best_tflops, avg_tflops, max_error in summaries:
            print(
                f"| {kernel_label} | {size} | {best_ms:.4f} | {avg_ms:.4f} | {ms_stddev:.4f} | "
                f"{best_tflops:.2f} | {avg_tflops:.2f} | {max_error:.4e} |"
            )

    if worst_error > tolerance:
        raise SystemExit(f"FAIL: max_error {worst_error:.4e} exceeded tolerance {tolerance:.4e}")
    print(f"PASS: max_error {worst_error:.4e} <= tolerance {tolerance:.4e}")


if __name__ == "__main__":
    main()
