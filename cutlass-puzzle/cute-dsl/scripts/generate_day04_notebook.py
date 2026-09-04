#!/usr/bin/env python3
"""Generate the explanation-rich Day 04 challenge notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


DSL_ROOT = Path(__file__).resolve().parents[1]
DAY_ROOT = DSL_ROOT / "day04_tv_layout_predicated"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def read_sections() -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    heading = "Introduction"
    sections[heading] = []
    for line in (DAY_ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:]
            sections[heading] = [line]
        else:
            sections[heading].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def puzzle_scaffold() -> str:
    source = (DAY_ROOT / "puzzle.py").read_text(encoding="utf-8")
    start = source.index("@cute.kernel")
    end = source.index("\ndef benchmark(")
    return source[start:end].strip()


def main() -> None:
    sections = read_sections()
    cells = [
        markdown(
            sections["Introduction"]
            + """

## How this challenge fits Notebook 08

Official [Notebook 08](../notebooks/08_elementwise_add.ipynb) builds one
elementwise operation through four representations:

| Stage | Ownership expression | Local lesson |
| --- | --- | --- |
| Scalar | one global thread ID selects one `(m, n)` | Day 03 part 1 |
| Vector | one thread selects a `(1, 8)` slice | Day 03 part 2 |
| TV layout | `(thread_id, value_id)` maps into a CTA tile | **Day 04** |
| Generic operator | the same mapping calls a compile-time operation | Extension after Day 04 |

The arithmetic does not become harder. The representation of ownership becomes
more explicit and reusable.
"""
        ),
        code(
            """import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

try:
    import cutlass.testing as testing
except ModuleNotFoundError as error:
    if error.name != "cutlass.testing":
        raise
    from cutlass.cute import testing

cutlass.cuda.initialize_cuda_context()"""
        ),
        markdown(sections["1. From thread IDs to TV layouts"]),
        code(
            """# Calculate the expected FP16 tile without compiling a kernel.
dtype_bits = 16
elts_per_vec = 128 // dtype_bits
thread_shape = (4, 32)
value_shape = (4, elts_per_vec)
tile_shape = (
    thread_shape[0] * value_shape[0],
    thread_shape[1] * value_shape[1],
)
threads = thread_shape[0] * thread_shape[1]
values_per_thread = value_shape[0] * value_shape[1]

assert threads == 128
assert values_per_thread == 32
assert tile_shape == (16, 256)
assert threads * values_per_thread == tile_shape[0] * tile_shape[1]
print(
    f"{threads} threads x {values_per_thread} values/thread "
    f"= tile {tile_shape}"
)"""
        ),
        markdown(sections["2. What `zipped_divide` does"]),
        code(
            """def ceil_div(a, b):
    return (a + b - 1) // b


M, N = 33, 513
TileM, TileN = 16, 256
RestM, RestN = ceil_div(M, TileM), ceil_div(N, TileN)
assert (RestM, RestN) == (3, 3)
print(f"CTA grid has {RestM} x {RestN} = {RestM * RestN} tiles")"""
        ),
        markdown(sections["3. The mapping chain inside the kernel"]),
        markdown(sections["4. Why the identity tensor is the predicate"]),
        markdown(sections["5. Puzzle TODO sequence"]),
        markdown(
            """## Fill the elementwise kernel

The host constructs `tiler_mn`, `tv_layout`, tiled data tensors, and the tiled
identity tensor. Fill the five TODOs in order. Running this cell only defines
the functions; the `NotImplementedError` is reached when the kernel is
compiled in the correctness cell."""
        ),
        code(puzzle_scaffold()),
        markdown(sections["6. Correctness before performance"]),
        code(
            """# Run after filling the five kernel TODOs.
M, N = 1023, 1025
a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros_like(a)

a_, b_, c_ = (
    from_dlpack(t, assumed_align=16).mark_layout_dynamic()
    for t in (a, b, c)
)
compiled = cute.compile(elementwise_add, a_, b_, c_)
compiled(a_, b_, c_)
torch.cuda.synchronize()
torch.testing.assert_close(c, a + b)
print("Correct on the partial-tile shape (1023, 1025)")"""
        ),
        markdown(sections["7. Benchmarking"]),
        code(
            """# Benchmark only after correctness passes. Use a large, regular shape.
M, N = 16384, 8192
a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.empty_like(a)
a_, b_, c_ = (
    from_dlpack(t, assumed_align=16).mark_layout_dynamic()
    for t in (a, b, c)
)

compiled = cute.compile(elementwise_add, a_, b_, c_)
compiled(a_, b_, c_)
torch.cuda.synchronize()
torch.testing.assert_close(c, a + b)

avg_time_us = testing.benchmark(
    compiled,
    kernel_arguments=testing.JitArguments(a_, b_, c_),
    warmup_iterations=5,
    iterations=100,
)
total_bytes = 3 * a.numel() * a.element_size()
bandwidth_gbps = total_bytes / (avg_time_us * 1_000)
print(f"time: {avg_time_us:.3f} us")
print(f"effective bandwidth: {bandwidth_gbps:.2f} GB/s")"""
        ),
        markdown(
            """## Extension: from add to a generic elementwise operation

Notebook 08's next step passes an operation as a compile-time argument. The
ownership pipeline stays unchanged:

```text
tile -> compose TV layout -> thread slice -> predicate -> load -> op -> store
```

Only the compute line changes conceptually from:

```python
result = frgA.load() + frgB.load()
```

to:

```python
result = op(frgA.load(), frgB.load())
```

After finishing addition, try multiplication and `relu(a * b)`. Keep the same
coordinate predicate and verify each operator against a PyTorch expression.
This demonstrates why layout code and mathematical operation should be
separate.

## Reference implementation

Use the reference only after completing the five TODOs:

```python
%run solution.py --M 1023 --N 1025
%run solution.py --M 16384 --N 8192 --benchmark
```
"""
        ),
        markdown(sections["What you should know afterward"]),
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
    destination = DAY_ROOT / "day04.ipynb"
    destination.write_text(
        json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
