#!/usr/bin/env python3
"""Generate the Colab-style memory allocator mock notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    guide = (ROOT / "README.md").read_text(encoding="utf-8")
    source = (ROOT / "memory_allocator_mock.py").read_text(encoding="utf-8")
    source = source.split('\nif __name__ == "__main__":', 1)[0].rstrip()
    notebook = {
        "cells": [
            markdown(guide),
            code(source),
            code("run_all_tests()\n"),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = ROOT / "memory_allocator_mock.ipynb"
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
