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
        (ROOT / "grid_infection_mock.ipynb").read_text(encoding="utf-8")
    )
    cells = notebook.get("cells", [])
    if notebook.get("nbformat") != 4 or len(cells) != 9:
        raise ValueError("unexpected notebook structure")
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    for cell in code_cells:
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError("notebook contains execution state")
        ast.parse(source(cell))
    combined = "\n".join(source(cell) for cell in code_cells)
    if combined.count("raise NotImplementedError") != 7:
        raise ValueError("candidate function stubs are missing")
    if "def run_candidate_unit_tests()" not in combined:
        raise ValueError("candidate unit-test exercise is missing")
    if "TODO: add candidate-written unit tests" not in combined:
        raise ValueError("candidate unit-test placeholder is missing")
    final_cell = source(code_cells[-1])
    if (
        "run_candidate_unit_tests()" not in final_cell
        or "run_all_tests()" not in final_cell
    ):
        raise ValueError("final test cell is missing")
    print("PASS grid_infection_mock.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
