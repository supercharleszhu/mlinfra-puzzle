#!/usr/bin/env python3
"""Generate one challenge notebook for every Triton lesson."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "official-notebooks"
sys.path.insert(0, str(ROOT))

from lesson_specs import LESSONS


def markdown(text: str, *, challenge: bool = False) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": ["challenge"]} if challenge else {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str, *, challenge: bool = False) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["challenge"]} if challenge else {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def puzzle_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    marker = '\nif __name__ == "__main__":'
    return source.split(marker, 1)[0].strip()


def official_cells(source_name: str) -> list[dict[str, object]]:
    path = OFFICIAL_ROOT / source_name.replace(".py", ".ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return cells


def main() -> None:
    for index, (directory, title, official, _) in enumerate(LESSONS, start=1):
        day_root = ROOT / directory
        guide = (day_root / "README.md").read_text(encoding="utf-8")
        source = puzzle_source(day_root / "puzzle.py")
        if official:
            cells = official_cells(official)
            challenge_start = len(cells)
            challenge_guide = "## Challenge" + guide.split("## Challenge", 1)[1]
            cells.extend(
                [
                    markdown(
                        f"# Challenge: {title}\n\n"
                        "The cells above are the official Triton tutorial. Now "
                        "complete the local exercise below without changing the "
                        "official implementation.\n\n"
                        f"{challenge_guide}",
                        challenge=True,
                    ),
                    code(source, challenge=True),
                    code("run()", challenge=True),
                ]
            )
        else:
            challenge_start = 0
            workflow = (
                "Original extension: compare Triton transpose with CuTe Day 05. "
                "Fill each `TODO`, then run the scaffold and its correctness "
                "checks from top to bottom."
                if directory == "day12_matrix_transpose"
                else
                "Original extension: compare Triton program-level reductions "
                "with CuTe Day 11. Complete the stable softmax, run the fixed "
                "tests, then benchmark the matched shapes."
            )
            cells = [
                markdown(guide, challenge=True),
                markdown(
                    f"## Notebook workflow\n\n{workflow}",
                    challenge=True,
                ),
                code(source, challenge=True),
            ]
            if directory == "day13_softmax_reduction":
                cells.extend(
                    [
                        markdown(
                            "## Correctness\n\nRun fixed regular, irregular, "
                            "and extreme-logit cases before measuring speed.",
                            challenge=True,
                        ),
                        code("test_softmax()", challenge=True),
                        markdown(
                            "## Benchmark\n\nUse a preallocated Triton output "
                            "and report both latency and effective bandwidth.",
                            challenge=True,
                        ),
                        code("benchmark_suite(rows=4096)", challenge=True),
                    ]
                )
            else:
                cells.append(code("run()", challenge=True))
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Triton (GPU)",
                    "language": "python",
                    "name": "triton-puzzle",
                },
                "language_info": {"name": "python", "version": "3.12"},
                "triton_puzzle": {
                    "challenge_start_cell": challenge_start,
                    "official_source": official,
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        destination = day_root / f"day{index:02d}.ipynb"
        destination.write_text(
            json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
