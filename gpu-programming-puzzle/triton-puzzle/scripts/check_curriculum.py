#!/usr/bin/env python3
"""Validate Triton puzzle sources and generated notebooks."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "official-notebooks"
sys.path.insert(0, str(ROOT))

from lesson_specs import LESSONS


def cell_source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def normalized_official_cells(source_name: str) -> list[dict[str, object]]:
    path = OFFICIAL_ROOT / source_name.replace(".py", ".ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return cells


def main() -> int:
    failures: list[str] = []
    for index, (directory, _, official, marker) in enumerate(LESSONS, start=1):
        day_root = ROOT / directory
        try:
            for name in ("puzzle.py", "solution.py"):
                source = (day_root / name).read_text(encoding="utf-8")
                ast.parse(source)
                if marker not in source:
                    raise ValueError(f"{name} is missing marker {marker!r}")
            guide = (day_root / "README.md").read_text(encoding="utf-8")
            if len(guide) < 600 or "## Challenge" not in guide:
                raise ValueError("README is incomplete")
            notebook_path = day_root / f"day{index:02d}.ipynb"
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            if notebook.get("nbformat") != 4 or len(notebook.get("cells", [])) < 4:
                raise ValueError("notebook structure is incomplete")
            metadata = notebook.get("metadata", {}).get("triton_puzzle", {})
            challenge_start = metadata.get("challenge_start_cell")
            if official:
                expected = normalized_official_cells(official)
                if notebook["cells"][: len(expected)] != expected:
                    raise ValueError("official tutorial cells were modified")
                if challenge_start != len(expected):
                    raise ValueError("challenge does not follow official tutorial")
            elif challenge_start != 0:
                raise ValueError("original lesson has invalid challenge metadata")
            challenge_cells = notebook["cells"][challenge_start:]
            if len(challenge_cells) < 3 or any(
                "challenge" not in cell.get("metadata", {}).get("tags", [])
                for cell in challenge_cells
            ):
                raise ValueError("challenge cells are not grouped at the end")
            challenge_text = "\n".join(
                cell_source(cell)
                for cell in challenge_cells
                if isinstance(cell, dict)
            )
            if marker not in challenge_text:
                raise ValueError(f"challenge is missing marker {marker!r}")
            for cell in notebook["cells"]:
                if cell.get("cell_type") != "code":
                    continue
                if cell.get("execution_count") is not None or cell.get("outputs"):
                    raise ValueError("notebook contains execution output")
                ast.parse(cell_source(cell))
        except Exception as error:
            failures.append(directory)
            print(f"FAIL {directory}: {error}", file=sys.stderr)
        else:
            print(f"PASS {directory}")
    if failures:
        return 1
    print(f"All {len(LESSONS)} Triton lessons passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
