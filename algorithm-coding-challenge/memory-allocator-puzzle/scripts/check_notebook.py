#!/usr/bin/env python3
"""Validate the generated mock notebook without executing candidate stubs."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def main() -> int:
    notebook = json.loads(
        (ROOT / "memory_allocator_mock.ipynb").read_text(encoding="utf-8")
    )
    cells = notebook.get("cells", [])
    if notebook.get("nbformat") != 4 or len(cells) != 3:
        raise ValueError("unexpected notebook structure")
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    for cell in code_cells:
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError("notebook contains execution state")
        ast.parse(source(cell))
    combined = "\n".join(source(cell) for cell in code_cells)
    if combined.count("raise NotImplementedError") != 10:
        raise ValueError("candidate method stubs are missing")
    if "run_all_tests()" not in source(code_cells[-1]):
        raise ValueError("final test cell is missing")
    print("PASS memory_allocator_mock.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
