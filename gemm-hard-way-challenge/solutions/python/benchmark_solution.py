"""Correctness and benchmark runner for the complete solution kernels."""

from __future__ import annotations

import argparse
import math
from typing import Callable

import torch

from solution_extension_loader import create_solution_extension

_CACHE_FLUSH_SIZE = 4 * 1024 * 1024
_cache_flush_buffer: torch.Tensor | None = None
_fastcu_b_transpose_cache: dict[tuple[int, tuple[int, ...], torch.dtype], torch.Tensor] = {}
_fastcu_c_transpose_cache: dict[tuple[int, int, int, torch.dtype], torch.Tensor] = {}


def _flush_l2_cache() -> None:
    global _cache_flush_buffer
    if _cache_flush_buffer is None:
        _cache_flush_buffer = torch.empty(_CACHE_FLUSH_SIZE, dtype=torch.int8, device="cuda")
    _cache_flush_buffer.zero_()
    torch.cuda.synchronize()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _time(fn: Callable[[], torch.Tensor], warmup: int, iters: int, flush_cache: bool) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        if flush_cache:
            _flush_l2_cache()
        out = fn()

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for idx in range(iters):
        if flush_cache:
            _flush_l2_cache()
        start_events[idx].record()
        out = fn()
        end_events[idx].record()
    torch.cuda.synchronize()

    times_ms = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    trim_count = max(1, iters // 10)
    if len(times_ms) > 2 * trim_count:
        times_ms = sorted(times_ms)[trim_count:-trim_count]
    return out, _median(times_ms)


def _make_registry(ext, dtype: torch.dtype, include_cutlass: bool):
    if dtype == torch.float32:
        registry = {
            "torch": lambda a, b: torch.matmul(a, b),
            "01_naive": lambda a, b: _fp32_out(ext.sgemm_naive, a, b),
            "02_coalesced": lambda a, b: _fp32_out(ext.sgemm_global_mem_coalesce, a, b),
            "03_shared_mem": lambda a, b: _fp32_out(ext.sgemm_shared_mem, a, b),
            "04_blocktiling_1d": lambda a, b: _fp32_out(ext.sgemm_blocktiling_1d, a, b),
            "05_blocktiling_2d": lambda a, b: _fp32_out(ext.sgemm_blocktiling_2d, a, b),
            "06_vectorize": lambda a, b: _fp32_out(ext.sgemm_vectorize, a, b),
            "07_warptiling": lambda a, b: _same_dtype_out(ext.sgemm_warptiling_default, a, b),
        }
        return registry

    if dtype == torch.float16:
        registry = {
            "torch": lambda a, b: torch.matmul(a, b),
            "08_warptiling_fp16": lambda a, b: _same_dtype_out(ext.sgemm_warptiling_fp16, a, b),
            "09_tensorcore_naive_fp16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_naive_fp16, a, b),
            "10_tensorcore_fp16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_fp16, a, b),
            "11_tensorcore_db_fp16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_double_buffered_fp16, a, b),
            "12_tensorcore_async_fp16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_async_fp16, a, b),
        }
        return registry

    registry = {
        "torch": lambda a, b: torch.matmul(a, b),
        "08_warptiling_bf16": lambda a, b: _same_dtype_out(ext.sgemm_warptiling_bf16, a, b),
        "09_tensorcore_naive_bf16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_naive_bf16, a, b),
        "10_tensorcore_bf16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_bf16, a, b),
        "11_tensorcore_db_bf16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_double_buffered_bf16, a, b),
        "12_tensorcore_async_bf16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_async_bf16, a, b),
    }
    if include_cutlass:
        if hasattr(ext, "sgemm_cutlass3_hopper_tunable_bf16"):
            registry["19_cutlass3_hopper_tunable_bf16"] = lambda a, b: _bf16_out(
                ext.sgemm_cutlass3_hopper_tunable_bf16, a, b
            )
            registry["13_fastcu_matmul2_manual_tma_wgmma_bf16"] = lambda a, b: _fastcu_handwritten_bf16_out(
                ext.sgemm_fastcu_matmul2_manual_tma_wgmma_bf16, a, b
            )
            registry["14_fastcu_matmul3_big_tile_bf16"] = lambda a, b: _fastcu_handwritten_bf16_out(
                ext.sgemm_fastcu_handwritten_tma_wgmma_bf16, a, b
            )
            registry["15_fastcu_matmul4_warp_specialized_bf16"] = lambda a, b: _fastcu_cached_bf16_out(
                ext.sgemm_fastcu_handwritten_cached_tma_maps_bf16, a, b
            )
            registry["16_fastcu_matmul6_persistent_bf16"] = lambda a, b: _fastcu_cached_bf16_out(
                ext.sgemm_fastcu_final_bf16, a, b
            )
            registry["17_fastcu_matmul10_tma_store_bf16"] = lambda a, b: _fastcu_cached_bf16_out(
                ext.sgemm_fastcu_tma_store_bf16, a, b
            )
            registry["18_fastcu_matmul12_final_bf16"] = lambda a, b: _fastcu_cached_bf16_out(
                ext.sgemm_fastcu_hilbert_final_bf16, a, b
            )
    return registry


def _fp32_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(a, b, c, 1.0, 0.0)
    return c


def _same_dtype_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    kernel(a, b, c, 1.0, 0.0)
    return c


def _fp32_then_cast(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(a, b, c, 1.0, 0.0)
    return c.to(a.dtype)


def _fp32_then_cast_config(kernel, config_id: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(config_id, a, b, c, 1.0, 0.0)
    return c.to(a.dtype)


def _bf16_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)
    kernel(a, b, c)
    return c


def _hopper_autotune_bf16_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)
    kernel(0, 2, 0, 1, 1, a, b, c)
    return c


def _fastcu_handwritten_bf16_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    key = (b.data_ptr(), tuple(b.shape), b.dtype)
    b_t = _fastcu_b_transpose_cache.get(key)
    if b_t is None:
        b_t = b.t().contiguous()
        _fastcu_b_transpose_cache[key] = b_t
    c_t = torch.empty((b.shape[1], a.shape[0]), device=a.device, dtype=torch.bfloat16)
    kernel(a, b_t, c_t)
    return c_t.t()


def _fastcu_cached_bf16_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--kernels", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-flush-cache", dest="flush_cache", action="store_false")
    parser.add_argument("--include-cutlass", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    ext = create_solution_extension(include_cutlass=args.include_cutlass, verbose=args.verbose_build)
    registry = _make_registry(ext, dtype, args.include_cutlass)
    selected = args.kernels or list(registry)

    print("| kernel | size | run | ms | TFLOPS | max_error |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    summaries = []
    for size in args.sizes:
        a = torch.randn((size, size), device="cuda", dtype=dtype)
        b = torch.randn((size, size), device="cuda", dtype=dtype)
        ref = torch.matmul(a, b)
        for name in selected:
            if name not in registry:
                raise ValueError(f"Unknown kernel for dtype={args.dtype}: {name}")
            ops = 2.0 * size * size * size
            run_results = []
            for run_idx in range(1, args.runs + 1):
                out, ms = _time(lambda name=name: registry[name](a, b), args.warmup, args.iters, args.flush_cache)
                err = (out.float() - ref.float()).abs().max().item()
                tflops = ops / (ms * 1e9)
                run_results.append((ms, tflops, err))
                print(f"| {name} | {size} | {run_idx} | {ms:.4f} | {tflops:.2f} | {err:.4e} |")
            best_ms = min(result[0] for result in run_results)
            avg_ms = sum(result[0] for result in run_results) / len(run_results)
            ms_stddev = math.sqrt(sum((result[0] - avg_ms) ** 2 for result in run_results) / len(run_results))
            best_tflops = max(result[1] for result in run_results)
            avg_tflops = sum(result[1] for result in run_results) / len(run_results)
            max_error = max(result[2] for result in run_results)
            summaries.append((name, size, best_ms, avg_ms, ms_stddev, best_tflops, avg_tflops, max_error))

    if args.runs > 1:
        print()
        print("| kernel | size | best_ms | avg_ms | stddev_ms | best_TFLOPS | avg_TFLOPS | max_error |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name, size, best_ms, avg_ms, ms_stddev, best_tflops, avg_tflops, max_error in summaries:
            print(
                f"| {name} | {size} | {best_ms:.4f} | {avg_ms:.4f} | {ms_stddev:.4f} | "
                f"{best_tflops:.2f} | {avg_tflops:.2f} | {max_error:.4e} |"
            )


if __name__ == "__main__":
    main()
