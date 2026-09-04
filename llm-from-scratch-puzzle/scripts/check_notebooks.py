from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "chapter02_text_data/chapter02.ipynb": {
        "LLM_TODO_CH02_01_BYTES",
        "LLM_TODO_CH02_02_WINDOWS",
    },
    "chapter03_attention/chapter03.ipynb": {
        "ATTENTION_TODO_CH03_00_CAUSAL",
        "ATTENTION_TODO_CH03_00_MULTIHEAD",
        "ATTENTION_TODO_CH03_00_SIMPLE",
        "LLM_TODO_CH03_01_TRANSFER",
        "LLM_TODO_CH03_02_MULTIHEAD",
        "LLM_TODO_CH03_03_DIMENSIONS",
    },
    "chapter04_gpt_architecture/chapter04.ipynb": {
        "LLM_TODO_CH04_01_COUNTS",
        "LLM_TODO_CH04_02_ESTIMATE",
    },
    "chapter05_pretraining/chapter05.ipynb": {
        "LLM_TODO_CH05_01_TEMPERATURE",
        "LLM_TODO_CH05_02_PROFILES",
        "LLM_TODO_CH05_03_DETERMINISTIC",
        "LLM_TODO_CH05_04_CHECKPOINT",
        "LLM_TODO_CH05_05_CROSS_ENTROPY",
        "LLM_TODO_CH05_06_CONFIG",
    },
    "chapter06_classification/chapter06.ipynb": {
        "LLM_TODO_CH06_01_PADDING",
        "LLM_TODO_CH06_02_UNFREEZE",
        "LLM_TODO_CH06_03_REPRESENTATION",
    },
    "chapter07_instruction_finetuning/chapter07.ipynb": {
        "LLM_TODO_CH07_01_FORMAT",
        "LLM_TODO_CH07_02_MASK",
        "LLM_TODO_CH07_03_PLAN",
        "LLM_TODO_CH07_04_LORA",
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
