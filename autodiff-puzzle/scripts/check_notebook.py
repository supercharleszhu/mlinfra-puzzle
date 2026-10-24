#!/usr/bin/env python3
"""Validate the vendored autodiff challenge notebook without executing it."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "autodiff_puzzlers.ipynb"
UPSTREAM_COMMIT = "ae4586b052f3328031ad64024edabbf6fbd2e4bb"
FORBIDDEN = ("!pip", "%pip", "!curl", "!wget", "git+https://")


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("expected notebook format 4")

    metadata = notebook.get("metadata", {}).get("autodiff_puzzle", {})
    if metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("upstream commit metadata is missing or incorrect")

    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    code = "\n".join(source(cell) for cell in code_cells)
    if code.count("# fill in") != 20:
        raise ValueError("expected exactly 20 Jacobian blanks")
    if "from checker import Array, Shaped, check, np" not in code:
        raise ValueError("notebook does not use the local checker")
    for marker in FORBIDDEN:
        if marker in code:
            raise ValueError(f"forbidden runtime installer found: {marker}")

    for cell in code_cells:
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError("notebook contains execution state")
        ast.parse(source(cell))

    print(f"PASS {NOTEBOOK.name}: 20 puzzles, no outputs or runtime installers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
