"""Minimal benchmark scaffold for the standalone GEMM challenge.

This file intentionally does not auto-build the upstream extension. Fill
`load_extension()` and register kernels as you complete each CUDA file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch

_CACHE_FLUSH_SIZE = 4 * 1024 * 1024
_cache_flush_buffer: torch.Tensor | None = None


@dataclass
class BenchResult:
    name: str
    size: int
    ms: float
    tflops: float
    max_error: float


def load_extension():
    # TODO(student): load your completed CUDA extension with
    # torch.utils.cpp_extension.load or your preferred build system.
    return None


def register_kernels(ext) -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    kernels: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "torch": lambda a, b: torch.matmul(a, b),
    }

    # TODO(student): uncomment/add entries as files become correct.
    # kernels["01_naive"] = lambda a, b: ext.sgemm_naive(a, b)
    # kernels["02_coalesced"] = lambda a, b: ext.sgemm_global_mem_coalesce(a, b)
    # kernels["03_shared_mem"] = lambda a, b: ext.sgemm_shared_mem(a, b)

    return kernels


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


def bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int, flush_cache: bool) -> tuple[torch.Tensor, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-flush-cache", dest="flush_cache", action="store_false")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    ext = load_extension()
    kernels = register_kernels(ext)

    print("| kernel | size | ms | TFLOPS | max error |")
    print("| --- | ---: | ---: | ---: | ---: |")

    for size in args.sizes:
        a = torch.randn((size, size), device="cuda", dtype=dtype)
        b = torch.randn((size, size), device="cuda", dtype=dtype)
        ref = torch.matmul(a, b)

        for name, fn in kernels.items():
            out, ms = bench(lambda: fn(a, b), args.warmup, args.iters, args.flush_cache)
            ops = 2.0 * size * size * size
            err = (out.float() - ref.float()).abs().max().item()
            print(f"| {name} | {size} | {ms:.4f} | {ops / (ms * 1e9):.2f} | {err:.3e} |")


if __name__ == "__main__":
    main()
