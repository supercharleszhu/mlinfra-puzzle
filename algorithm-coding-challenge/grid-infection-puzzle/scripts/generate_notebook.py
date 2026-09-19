#!/usr/bin/env python3
"""Generate the Colab-style grid infection mock notebook."""

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


def split_source(source: str) -> tuple[str, str, str]:
    candidate_marker = (
        "# ---------------------------------------------------------------------------\n"
        "# Candidate functions\n"
        "# ---------------------------------------------------------------------------"
    )
    oracle_marker = (
        "# ---------------------------------------------------------------------------\n"
        "# Slow oracles\n"
        "# ---------------------------------------------------------------------------"
    )
    setup, remainder = source.split(candidate_marker, 1)
    candidates, harness = remainder.split(oracle_marker, 1)
    return (
        setup.rstrip(),
        candidates.strip(),
        f"{oracle_marker}\n{harness.strip()}",
    )


def main() -> None:
    guide = (ROOT / "README.md").read_text(encoding="utf-8")
    source = (ROOT / "grid_infection_mock.py").read_text(encoding="utf-8")
    source = source.split('\nif __name__ == "__main__":', 1)[0].rstrip()
    setup, candidates, harness = split_source(source)
    notebook = {
        "cells": [
            markdown(guide),
            code(setup),
            markdown(
                """## Your task: write unit tests first

Before implementing the algorithms, write your own unit tests in the next
cell. Use plain `assert` statements and test behavior rather than internal
implementation details.

Cover at least:

- Part 1: empty, no source, already infected, one row/column, multiple sources,
  and a case proving newly infected cells do not spread again on the same day.
- Part 2: a fully blocked healthy region and a partially reachable region.
- Part 3: `D=1` and one case that would expose a recovery/spread off-by-one.
- Part 4: one threshold case and one death/recovery tie.
- Invalid input: one ragged grid or invalid parameter.

Remove the placeholder failure only after adding the tests. The final cell runs
your tests first, followed by the provided deterministic oracle comparisons.
"""
            ),
            code(
                """def run_candidate_unit_tests() -> None:
    # TODO: write your unit tests before implementing the functions below.
    #
    # Example shape only; replace this placeholder with your own assertions:
    # assert time_to_full_infection([[1]]) == 0
    #
    # The provided helper can check invalid inputs:
    # _expect_value_error(time_to_stable_state, [[1]], 0)
    raise AssertionError("TODO: add candidate-written unit tests")
"""
            ),
            markdown(
                """## Implement the challenge

Fill the seven functions below. Run the final test cell after completing a
part; your focused unit tests should fail faster and explain the behavior more
clearly than the full randomized harness.
"""
            ),
            code(candidates),
            markdown(
                """## Provided oracle harness

This cell defines deliberately slow synchronous oracles, fixed regression
tests, and deterministic random differential tests. Do not edit it while
solving the challenge.
"""
            ),
            code(harness),
            code("run_candidate_unit_tests()\nrun_all_tests()\n"),
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
    destination = ROOT / "grid_infection_mock.ipynb"
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
