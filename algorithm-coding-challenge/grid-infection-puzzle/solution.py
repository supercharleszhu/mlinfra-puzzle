"""Reference solutions for the revised grid infection mock."""

from __future__ import annotations

import sys
from collections import deque

import grid_infection_mock as mock


Grid = mock.Grid


def _full_infection_bfs(grid: Grid, allowed: set[int]) -> int:
    state = mock._parse_grid(grid, allowed)
    if not state:
        return 0
    rows, cols = len(state), len(state[0])
    queue = deque()
    healthy = 0
    for r in range(rows):
        for c in range(cols):
            if state[r][c] == 1:
                queue.append((r, c, 0))
            elif state[r][c] == 0:
                healthy += 1
    if healthy == 0:
        return 0
    if not queue:
        return -1

    last_day = 0
    while queue:
        r, c, day = queue.popleft()
        for nr, nc in mock._neighbors(r, c, rows, cols):
            if state[nr][nc] != 0:
                continue
            state[nr][nc] = 1
            healthy -= 1
            last_day = day + 1
            queue.append((nr, nc, day + 1))
    return last_day if healthy == 0 else -1


def time_to_full_infection(grid: list[list[int]]) -> int:
    return _full_infection_bfs(grid, {0, 1})


def time_to_full_infection_with_immunity(
    grid: list[list[int]],
) -> int:
    return _full_infection_bfs(grid, {0, 1, 2})


def time_to_stable_state(grid: list[list[int]], D: int) -> int:
    if D < 1:
        raise ValueError("D must be positive")
    state = mock._parse_grid(grid, {0, 1})
    if not state:
        return 0
    rows, cols = len(state), len(state[0])
    queue = deque()
    seen = set()
    for r in range(rows):
        for c in range(cols):
            if state[r][c] == 1:
                queue.append((r, c, 0))
                seen.add((r, c))
    if not queue:
        return 0
    if D == 1:
        return 1

    last_infection_day = 0
    while queue:
        r, c, day = queue.popleft()
        last_infection_day = max(last_infection_day, day)
        for nr, nc in mock._neighbors(r, c, rows, cols):
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc, day + 1))
    return last_infection_day + D


def time_to_full_infection_threshold(
    grid: list[list[int]], K: int
) -> int:
    if not 1 <= K <= 4:
        raise ValueError("K must be between 1 and 4")
    state = mock._parse_grid(grid, {0, 1, 2})
    if not state:
        return 0
    rows, cols = len(state), len(state[0])
    healthy = sum(cell == 0 for row in state for cell in row)
    if healthy == 0:
        return 0

    counts = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if state[r][c] != 1:
                continue
            for nr, nc in mock._neighbors(r, c, rows, cols):
                if state[nr][nc] == 0:
                    counts[nr][nc] += 1

    ready = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if state[r][c] == 0 and counts[r][c] >= K
    }
    days = 0
    while ready:
        today = ready
        ready = set()
        for r, c in today:
            state[r][c] = 1
            healthy -= 1
        for r, c in today:
            for nr, nc in mock._neighbors(r, c, rows, cols):
                if state[nr][nc] != 0:
                    continue
                counts[nr][nc] += 1
                if counts[nr][nc] >= K:
                    ready.add((nr, nc))
        days += 1
    return days if healthy == 0 else -1


def _simulate_composite(
    grid: Grid,
    D: int,
    infection_threshold: int,
    death_threshold: int,
    death_countdown: int,
) -> tuple[int, int]:
    if D < 1 or death_countdown < 1:
        raise ValueError("day counts must be positive")
    if not 1 <= infection_threshold <= 4:
        raise ValueError("infection_threshold must be between 1 and 4")
    if not 1 <= death_threshold <= 4:
        raise ValueError("death_threshold must be between 1 and 4")
    state = mock._parse_grid(grid, {0, 1, 2})
    if not state:
        return 0, 0
    rows, cols = len(state), len(state[0])
    infected_on = {
        (r, c): 0
        for r in range(rows)
        for c in range(cols)
        if state[r][c] == 1
    }
    scheduled_death: dict[tuple[int, int], int] = {}
    current_day = 0
    deaths = 0

    while infected_on:
        current_day += 1
        survivors = {}
        for position, start_day in infected_on.items():
            r, c = position
            if scheduled_death.get(position) == current_day:
                state[r][c] = 3
                deaths += 1
                scheduled_death.pop(position, None)
            elif current_day - start_day >= D:
                state[r][c] = 2
                scheduled_death.pop(position, None)
            else:
                survivors[position] = start_day
        infected_on = survivors

        counts = [[0] * cols for _ in range(rows)]
        for r, c in infected_on:
            for nr, nc in mock._neighbors(r, c, rows, cols):
                counts[nr][nc] += 1

        for position in infected_on:
            r, c = position
            if (
                position not in scheduled_death
                and counts[r][c] >= death_threshold
            ):
                scheduled_death[position] = current_day + death_countdown

        newly_infected = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if state[r][c] == 0 and counts[r][c] >= infection_threshold
        ]
        for r, c in newly_infected:
            state[r][c] = 1
            infected_on[(r, c)] = current_day

    return current_day, deaths


def time_to_end_with_death_countdown(
    grid: list[list[int]], D: int, K: int, N: int
) -> tuple[int, int]:
    return _simulate_composite(grid, D, 1, K, N)


def time_to_end_with_death_on_recovery(
    grid: list[list[int]], D: int, K: int
) -> tuple[int, int]:
    if D < 1:
        raise ValueError("D must be positive")
    if not 1 <= K <= 4:
        raise ValueError("K must be between 1 and 4")
    state = mock._parse_grid(grid, {0, 1, 2})
    if not state:
        return 0, 0
    rows, cols = len(state), len(state[0])
    infected_on = {
        (r, c): 0
        for r in range(rows)
        for c in range(cols)
        if state[r][c] == 1
    }
    initial = set(infected_on)
    doomed = {
        (r, c)
        for r, c in initial
        if sum(
            (nr, nc) in initial
            for nr, nc in mock._neighbors(r, c, rows, cols)
        )
        >= K
    }
    current_day = 0
    deaths = 0

    while infected_on:
        current_day += 1
        survivors = {}
        for position, start_day in infected_on.items():
            r, c = position
            if current_day - start_day >= D:
                if position in doomed:
                    state[r][c] = 3
                    deaths += 1
                else:
                    state[r][c] = 2
            else:
                survivors[position] = start_day
        infected_on = survivors

        counts = [[0] * cols for _ in range(rows)]
        for r, c in infected_on:
            for nr, nc in mock._neighbors(r, c, rows, cols):
                counts[nr][nc] += 1
        newly_infected = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if state[r][c] == 0 and counts[r][c] >= 1
        ]
        for r, c in newly_infected:
            state[r][c] = 1
            infected_on[(r, c)] = current_day
            if counts[r][c] >= K:
                doomed.add((r, c))

    return current_day, deaths


def time_to_end_composite(
    grid: list[list[int]],
    D: int,
    infection_threshold: int,
    death_threshold: int,
    death_countdown: int,
) -> tuple[int, int]:
    return _simulate_composite(
        grid,
        D,
        infection_threshold,
        death_threshold,
        death_countdown,
    )


def run_all_tests() -> None:
    mock._run_all_tests(sys.modules[__name__])


if __name__ == "__main__":
    run_all_tests()
