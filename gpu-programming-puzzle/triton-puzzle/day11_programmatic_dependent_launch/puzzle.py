#!/usr/bin/env python3
"""Day 11 puzzle: CUDA programmatic dependent launch (PDL)."""

import torch
import triton
import triton.language as tl


@triton.jit
def pdl_add_kernel(x_ptr, y_ptr, out_ptr, n, USE_GDC: tl.constexpr, BLOCK: tl.constexpr):
    # EXERCISE(1): use tl.extra.cuda.gdc_wait() before consuming prior output,
    # then announce dependent grids with gdc_launch_dependents before storing.
    pass


def pdl_add(x: torch.Tensor, y: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    raise NotImplementedError("EXERCISE(2): launch with USE_GDC and launch_pdl")


def run() -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        print("Day 11 skipped: PDL requires compute capability 9+")
        return
    x = torch.randn(100_003, device="cuda")
    y = torch.randn_like(x)
    torch.testing.assert_close(pdl_add(x, y), x + y)
    print("Day 11 PDL: correctness OK")


if __name__ == "__main__":
    run()
