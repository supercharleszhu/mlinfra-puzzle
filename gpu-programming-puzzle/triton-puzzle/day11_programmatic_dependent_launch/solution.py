#!/usr/bin/env python3
"""Day 11 solution: CUDA programmatic dependent launch (PDL)."""

import torch
import triton
import triton.language as tl


@triton.jit
def pdl_add_kernel(x_ptr, y_ptr, out_ptr, n, USE_GDC: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + offsets, x + y, mask=mask)


def pdl_add(x: torch.Tensor, y: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("x and y must have equal shapes")
    if enabled and torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("PDL requires CUDA compute capability 9 or newer")
    out = torch.empty_like(x)
    grid = (triton.cdiv(out.numel(), 1024),)
    pdl_add_kernel[grid](
        x, y, out, out.numel(),
        USE_GDC=enabled, BLOCK=1024, launch_pdl=enabled,
    )
    return out


def run() -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        print("Day 11 skipped: PDL requires compute capability 9+")
        return
    x = torch.randn(100_003, device="cuda")
    y = torch.randn_like(x)
    torch.testing.assert_close(pdl_add(x, y, True), x + y)
    torch.testing.assert_close(pdl_add(x, y, False), x + y)
    print("Day 11 PDL: correctness OK")


if __name__ == "__main__":
    run()
