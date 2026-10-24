#!/usr/bin/env python3
"""Validate the vendored tensor challenge notebook without executing it."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tensor_puzzlers.ipynb"
UPSTREAM_COMMIT = "ad5069f7ef48523f733d445a7f94e93427d64f2d"
FORBIDDEN = (
    "!pip",
    "%pip",
    "!curl",
    "!wget",
    "git+https://",
    "from lib import",
    "draw_examples",
)


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("expected notebook format 4")

    metadata = notebook.get("metadata", {}).get("tensor_puzzle", {})
    if metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("upstream commit metadata is missing or incorrect")

    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    code = "\n".join(source(cell) for cell in code_cells)
    markdown = "\n".join(
        source(cell) for cell in cells if cell.get("cell_type") == "markdown"
    )

    if code.count("# fill in") != 21:
        raise ValueError("expected exactly 21 tensor implementation blanks")
    if markdown.count("## Puzzle ") != 21:
        raise ValueError("expected exactly 21 numbered puzzle headings")
    if "from checker import TT, make_test, np, run_test, tensor, torch" not in code:
        raise ValueError("notebook does not use the local checker")
    if "# run_test(test_bucketize)" not in code:
        raise ValueError("bucketize test cell is missing")

    for marker in FORBIDDEN:
        if marker in code:
            raise ValueError(f"forbidden runtime dependency found: {marker}")

    for cell in code_cells:
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError("notebook contains execution state")
        ast.parse(source(cell))

    print(f"PASS {NOTEBOOK.name}: 21 puzzles, no outputs or runtime installers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
