"""CoderPad-style grid infection interview mock.

Fill the candidate functions and run this file. The harness contains fixed
tests, deterministic random tests, and slow simultaneous-update oracles.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence


Grid = Sequence[Sequence[int]]
DIRS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _parse_grid(grid: Grid, allowed: set[int]) -> list[list[int]]:
    rows = [list(row) for row in grid]
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("grid must be rectangular")
    if width == 0:
        return []
    invalid = {cell for row in rows for cell in row} - allowed
    if invalid:
        raise ValueError(f"invalid grid states: {sorted(invalid)}")
    return rows


def _neighbors(r: int, c: int, rows: int, cols: int):
    for dr, dc in DIRS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


# ---------------------------------------------------------------------------
# Candidate functions
# ---------------------------------------------------------------------------


def time_to_full_infection(grid: list[list[int]]) -> int:
    """Part 1: return days to infect every cell, or -1 if impossible."""

    raise NotImplementedError


def time_to_full_infection_with_immunity(grid: list[list[int]]) -> int:
    """Part 2: 2 is an immune wall; return -1 if any 0 is unreachable."""

    raise NotImplementedError


def time_to_stable_state(grid: list[list[int]], D: int) -> int:
    """Part 3: recover before spreading when current_day - infection_day >= D."""

    raise NotImplementedError


def time_to_full_infection_threshold(
    grid: list[list[int]], K: int
) -> int:
    """Part 4A: a healthy cell needs at least K infected neighbors."""

    raise NotImplementedError


def time_to_end_with_death_countdown(
    grid: list[list[int]], D: int, K: int, N: int
) -> tuple[int, int]:
    """Part 4B: infection threshold 1 plus an independent death countdown."""

    raise NotImplementedError


def time_to_end_with_death_on_recovery(
    grid: list[list[int]], D: int, K: int
) -> tuple[int, int]:
    """Part 4B strict: surrounded-at-infection cells die at recovery time."""

    raise NotImplementedError


def time_to_end_composite(
    grid: list[list[int]],
    D: int,
    infection_threshold: int,
    death_threshold: int,
    death_countdown: int,
) -> tuple[int, int]:
    """Part 4C: immunity, threshold infection, recovery, and death combined."""

    raise NotImplementedError


# ---------------------------------------------------------------------------
# Slow oracles
# ---------------------------------------------------------------------------


def full_infection_oracle(grid: Grid, *, immune: bool) -> int:
    state = _parse_grid(grid, {0, 1, 2} if immune else {0, 1})
    if not state:
        return 0
    rows, cols = len(state), len(state[0])
    days = 0
    while True:
        healthy = sum(cell == 0 for row in state for cell in row)
        if healthy == 0:
            return days
        newly_infected = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if state[r][c] == 0
            and any(
                state[nr][nc] == 1
                for nr, nc in _neighbors(r, c, rows, cols)
            )
        ]
        if not newly_infected:
            return -1
        for r, c in newly_infected:
            state[r][c] = 1
        days += 1


def recovery_oracle(grid: Grid, D: int) -> int:
    if D < 1:
        raise ValueError("D must be positive")
    state = _parse_grid(grid, {0, 1})
    if not state:
        return 0
    rows, cols = len(state), len(state[0])
    infection_day = {
        (r, c): 0
        for r in range(rows)
        for c in range(cols)
        if state[r][c] == 1
    }
    current_day = 0

    while infection_day:
        current_day += 1
        recovering = {
            position
            for position, infected_on in infection_day.items()
            if current_day - infected_on >= D
        }
        for r, c in recovering:
            state[r][c] = 2
            del infection_day[(r, c)]

        active = set(infection_day)
        newly_infected = {
            (nr, nc)
            for r, c in active
            for nr, nc in _neighbors(r, c, rows, cols)
            if state[nr][nc] == 0
        }
        for r, c in newly_infected:
            state[r][c] = 1
            infection_day[(r, c)] = current_day

    return current_day


def threshold_oracle(grid: Grid, K: int) -> int:
    if not 1 <= K <= 4:
        raise ValueError("K must be between 1 and 4")
    state = _parse_grid(grid, {0, 1, 2})
    if not state:
        return 0
    rows, cols = len(state), len(state[0])
    days = 0
    while True:
        healthy = sum(cell == 0 for row in state for cell in row)
        if healthy == 0:
            return days
        newly_infected = []
        for r in range(rows):
            for c in range(cols):
                if state[r][c] != 0:
                    continue
                count = sum(
                    state[nr][nc] == 1
                    for nr, nc in _neighbors(r, c, rows, cols)
                )
                if count >= K:
                    newly_infected.append((r, c))
        if not newly_infected:
            return -1
        for r, c in newly_infected:
            state[r][c] = 1
        days += 1


def composite_oracle(
    grid: Grid,
    D: int,
    infection_threshold: int,
    death_threshold: int,
    death_countdown: int,
) -> tuple[int, int]:
    """Part 4C semantics; Part 4B is the infection_threshold=1 case."""

    if D < 1 or death_countdown < 1:
        raise ValueError("day counts must be positive")
    if not 1 <= infection_threshold <= 4:
        raise ValueError("infection_threshold must be between 1 and 4")
    if not 1 <= death_threshold <= 4:
        raise ValueError("death_threshold must be between 1 and 4")
    state = _parse_grid(grid, {0, 1, 2})
    if not state:
        return 0, 0
    rows, cols = len(state), len(state[0])
    infection_day = [
        [0 if state[r][c] == 1 else None for c in range(cols)]
        for r in range(rows)
    ]
    death_day = [[None] * cols for _ in range(rows)]
    deaths = 0
    current_day = 0

    while any(cell == 1 for row in state for cell in row):
        current_day += 1
        next_state = [row[:] for row in state]
        for r in range(rows):
            for c in range(cols):
                if state[r][c] != 1:
                    continue
                if death_day[r][c] == current_day:
                    next_state[r][c] = 3
                    infection_day[r][c] = None
                    death_day[r][c] = None
                    deaths += 1
                elif current_day - infection_day[r][c] >= D:
                    next_state[r][c] = 2
                    infection_day[r][c] = None
                    death_day[r][c] = None

        counts = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if next_state[r][c] != 1:
                    continue
                for nr, nc in _neighbors(r, c, rows, cols):
                    counts[nr][nc] += 1

        for r in range(rows):
            for c in range(cols):
                if (
                    next_state[r][c] == 1
                    and death_day[r][c] is None
                    and counts[r][c] >= death_threshold
                ):
                    death_day[r][c] = current_day + death_countdown
                elif (
                    next_state[r][c] == 0
                    and counts[r][c] >= infection_threshold
                ):
                    next_state[r][c] = 1
                    infection_day[r][c] = current_day

        state = next_state

    return current_day, deaths


def death_countdown_oracle(
    grid: Grid, D: int, K: int, N: int
) -> tuple[int, int]:
    return composite_oracle(grid, D, 1, K, N)


def death_on_recovery_oracle(
    grid: Grid, D: int, K: int
) -> tuple[int, int]:
    if D < 1:
        raise ValueError("D must be positive")
    if not 1 <= K <= 4:
        raise ValueError("K must be between 1 and 4")
    state = _parse_grid(grid, {0, 1, 2})
    if not state:
        return 0, 0
    rows, cols = len(state), len(state[0])
    infection_day = [
        [0 if state[r][c] == 1 else None for c in range(cols)]
        for r in range(rows)
    ]
    doomed = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if state[r][c] != 1:
                continue
            count = sum(
                state[nr][nc] == 1
                for nr, nc in _neighbors(r, c, rows, cols)
            )
            doomed[r][c] = count >= K
    deaths = 0
    current_day = 0

    while any(cell == 1 for row in state for cell in row):
        current_day += 1
        next_state = [row[:] for row in state]
        for r in range(rows):
            for c in range(cols):
                if (
                    state[r][c] == 1
                    and current_day - infection_day[r][c] >= D
                ):
                    next_state[r][c] = 3 if doomed[r][c] else 2
                    deaths += doomed[r][c]
                    infection_day[r][c] = None

        counts = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if next_state[r][c] != 1:
                    continue
                for nr, nc in _neighbors(r, c, rows, cols):
                    counts[nr][nc] += 1

        for r in range(rows):
            for c in range(cols):
                if next_state[r][c] == 0 and counts[r][c] >= 1:
                    next_state[r][c] = 1
                    infection_day[r][c] = current_day
                    doomed[r][c] = counts[r][c] >= K

        state = next_state

    return current_day, deaths


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


def _expect_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def _expect_value_error(fn, *args) -> None:
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__}{args!r} should raise ValueError")


def _fixed_tests(impl) -> None:
    basic_cases = [
        ([], 0),
        ([[]], 0),
        ([[1]], 0),
        ([[0]], -1),
        ([[1, 0, 0, 0, 0]], 4),
        ([[1], [0], [0], [0], [0]], 4),
        ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 2),
        ([[1, 0, 0], [0, 0, 0], [0, 0, 1]], 2),
    ]
    for grid, expected in basic_cases:
        _expect_equal(
            impl.time_to_full_infection(grid), expected, f"Part 1 {grid}"
        )

    immune_cases = [
        ([], 0),
        ([[2]], 0),
        ([[0, 2, 0]], -1),
        ([[1, 0, 2, 0]], -1),
        ([[1, 0], [2, 0]], 2),
        ([[1, 2, 0], [2, 2, 0], [0, 0, 0]], -1),
        ([[1, 0, 2, 0, 0]], -1),
        ([[1, 0, 2, 1, 0]], 1),
    ]
    for grid, expected in immune_cases:
        _expect_equal(
            impl.time_to_full_infection_with_immunity(grid),
            expected,
            f"Part 2 {grid}",
        )

    recovery_cases = [
        ([], 2, 0),
        ([[0, 0]], 2, 0),
        ([[1]], 1, 1),
        ([[1]], 3, 3),
        ([[1, 0]], 1, 1),
        ([[1, 0]], 2, 3),
        ([[1, 0, 0]], 2, 4),
        ([[0, 1, 0], [0, 0, 0]], 3, 5),
    ]
    for grid, D, expected in recovery_cases:
        _expect_equal(
            impl.time_to_stable_state(grid, D),
            expected,
            f"Part 3 {grid}, D={D}",
        )

    threshold_cases = [
        ([], 1, 0),
        ([[0]], 1, -1),
        ([[1, 0, 0]], 1, 2),
        ([[1, 0, 1]], 2, 1),
        ([[1, 0, 0], [1, 0, 0]], 2, -1),
        ([[1, 0, 1], [0, 0, 0]], 2, -1),
        ([[1, 0], [0, 1]], 2, 1),
        ([[1, 2, 0]], 1, -1),
    ]
    for grid, K, expected in threshold_cases:
        _expect_equal(
            impl.time_to_full_infection_threshold(grid, K),
            expected,
            f"Part 4A {grid}, K={K}",
        )

    countdown_cases = [
        ([], 2, 1, 1, (0, 0)),
        ([[1]], 3, 1, 1, (3, 0)),
        ([[1, 1]], 3, 1, 1, (2, 2)),
        ([[1, 1]], 3, 1, 2, (3, 2)),
        ([[1, 1]], 1, 1, 1, (1, 0)),
        ([[1, 1]], 2, 1, 3, (2, 0)),
        ([[1, 0]], 3, 2, 1, (4, 0)),
    ]
    for grid, D, K, N, expected in countdown_cases:
        _expect_equal(
            impl.time_to_end_with_death_countdown(grid, D, K, N),
            expected,
            f"Part 4B countdown {grid}",
        )

    strict_cases = [
        ([], 2, 1, (0, 0)),
        ([[1]], 2, 1, (2, 0)),
        ([[1, 1]], 2, 1, (2, 2)),
        ([[1, 0, 1]], 2, 2, (3, 1)),
        ([[1, 0]], 1, 1, (1, 0)),
    ]
    for grid, D, K, expected in strict_cases:
        _expect_equal(
            impl.time_to_end_with_death_on_recovery(grid, D, K),
            expected,
            f"Part 4B strict {grid}",
        )

    composite_cases = [
        ([], 2, 1, 1, 1, (0, 0)),
        ([[1]], 2, 1, 1, 1, (2, 0)),
        ([[1, 0, 1]], 3, 2, 2, 1, (3, 1)),
        ([[1, 2, 0]], 2, 1, 1, 1, (2, 0)),
    ]
    for grid, D, infection_K, death_K, countdown, expected in composite_cases:
        _expect_equal(
            impl.time_to_end_composite(
                grid, D, infection_K, death_K, countdown
            ),
            expected,
            f"Part 4C {grid}",
        )

    _expect_value_error(impl.time_to_full_infection, [[1, 0], [0]])
    _expect_value_error(impl.time_to_full_infection, [[], [1]])
    _expect_value_error(impl.time_to_full_infection, [[2]])
    _expect_value_error(impl.time_to_full_infection_with_immunity, [[-1]])
    _expect_value_error(impl.time_to_stable_state, [[1]], 0)
    _expect_value_error(impl.time_to_full_infection_threshold, [[1]], 0)
    _expect_value_error(
        impl.time_to_end_with_death_countdown, [[1]], 1, 1, 0
    )


def _random_grid(
    rng: random.Random, rows: int, cols: int, symbols: tuple[int, ...]
) -> list[list[int]]:
    if rows == 0 or cols == 0:
        return []
    return [
        [rng.choice(symbols) for _ in range(cols)]
        for _ in range(rows)
    ]


def _random_tests(impl, seed: int = 20260915) -> None:
    rng = random.Random(seed)
    for case in range(200):
        rows, cols = rng.randint(1, 5), rng.randint(1, 5)
        basic = _random_grid(rng, rows, cols, (0, 0, 0, 1))
        immune = _random_grid(rng, rows, cols, (0, 0, 0, 1, 2, 2))

        _expect_equal(
            impl.time_to_full_infection(basic),
            full_infection_oracle(basic, immune=False),
            f"random Part 1 case {case}",
        )
        _expect_equal(
            impl.time_to_full_infection_with_immunity(immune),
            full_infection_oracle(immune, immune=True),
            f"random Part 2 case {case}",
        )

        D = rng.randint(1, 4)
        _expect_equal(
            impl.time_to_stable_state(basic, D),
            recovery_oracle(basic, D),
            f"random Part 3 case {case}",
        )

        K = rng.randint(1, 4)
        _expect_equal(
            impl.time_to_full_infection_threshold(immune, K),
            threshold_oracle(immune, K),
            f"random Part 4A case {case}",
        )

        N = rng.randint(1, 3)
        _expect_equal(
            impl.time_to_end_with_death_countdown(immune, D, K, N),
            death_countdown_oracle(immune, D, K, N),
            f"random Part 4B countdown case {case}",
        )
        _expect_equal(
            impl.time_to_end_with_death_on_recovery(immune, D, K),
            death_on_recovery_oracle(immune, D, K),
            f"random Part 4B strict case {case}",
        )

        infection_K = rng.randint(1, 4)
        death_K = rng.randint(1, 4)
        _expect_equal(
            impl.time_to_end_composite(immune, D, infection_K, death_K, N),
            composite_oracle(immune, D, infection_K, death_K, N),
            f"random Part 4C case {case}",
        )


def _run_all_tests(impl) -> None:
    _fixed_tests(impl)
    print("PASS fixed tests")
    _random_tests(impl)
    print("PASS 1,400 deterministic random comparisons")


def run_all_tests() -> None:
    _run_all_tests(sys.modules[__name__])


if __name__ == "__main__":
    run_all_tests()
