from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "chapter01_cs336_transformer_e2e/chapter01.ipynb": set(),
    "chapter02_classification/chapter02.ipynb": {
        "LLM_TODO_CH02_01_PADDING",
        "LLM_TODO_CH02_02_UNFREEZE",
        "LLM_TODO_CH02_03_REPRESENTATION",
    },
    "chapter03_instruction_finetuning/chapter03.ipynb": {
        "LLM_TODO_CH03_01_FORMAT",
        "LLM_TODO_CH03_02_MASK",
        "LLM_TODO_CH03_03_PLAN",
        "LLM_TODO_CH03_04_LORA",
    },
}


def source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def check_notebook(relative_path: str, expected_todos: set[str]) -> None:
    path = ROOT / relative_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{relative_path}: expected nbformat 4")

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{relative_path}: notebook has no cells")

    markdown_cells = [
        source(cell)
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
    ]
    if not markdown_cells or sum(map(len, markdown_cells)) < 2_000:
        raise ValueError(f"{relative_path}: explanations are too short")

    all_source = "\n".join(
        source(cell) for cell in cells if isinstance(cell, dict)
    )
    actual_todos = {
        token
        for token in all_source.replace("`", " ").split()
        if token.startswith(("LLM_TODO_CH", "ATTENTION_TODO_CH"))
    }
    if actual_todos != expected_todos:
        raise ValueError(
            f"{relative_path}: TODOs {sorted(actual_todos)} != "
            f"{sorted(expected_todos)}"
        )

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError(f"{relative_path}: cell {index} contains output")
        code_source = source(cell)
        if code_source.lstrip().startswith(("%", "!")):
            continue
        try:
            ast.parse(code_source)
        except SyntaxError as error:
            raise SyntaxError(
                f"{relative_path}: invalid code in cell {index}: {error}"
            ) from error


def main() -> int:
    failures: list[str] = []
    for relative_path, expected_todos in EXPECTED.items():
        try:
            check_notebook(relative_path, expected_todos)
        except Exception as error:
            failures.append(relative_path)
            print(f"FAIL {relative_path}: {error}", file=sys.stderr)
        else:
            print(f"PASS {relative_path}")

    if failures:
        print(f"{len(failures)} notebook(s) failed", file=sys.stderr)
        return 1
    print(f"All {len(EXPECTED)} puzzle notebooks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
