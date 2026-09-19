# Grid Infection Simulation Mock Interview

This is a multi-stage interview exercise that can be copied into CoderPad or
run directly in Colab/Jupyter. It focuses on 4-neighbor multi-source BFS,
synchronous updates, per-cell state, off-by-one errors, unit tests, and
randomized differential testing.

> This version uses the four orthogonal neighbors: up, down, left, and right.
> Some variants include diagonal neighbors. Clarify this before coding because
> it changes the propagation time in every test case.

## Running the exercise

```bash
cd algorithm-coding-challenge/grid-infection-puzzle

# Single-file candidate mock; initially fails at the first TODO.
python3 grid_infection_mock.py

# Complete reference implementation.
python3 solution.py

# Colab/Jupyter version.
python3 scripts/generate_notebook.py
python3 scripts/check_notebook.py
```

`grid_infection_mock.py` contains the function signatures, fixed tests, slow
oracles, deterministic random tests, and `run_all_tests()`. The reference
implementation executes 1,400 randomized differential comparisons.

`grid_infection_mock.ipynb` is a self-contained mock interview that runs in
this order:

1. Load common helpers.
2. **Have the candidate write unit tests first**, covering synchronous updates,
   boundary cases, and off-by-one behavior.
3. Implement the seven candidate functions.
4. Load the read-only slow-oracle and randomized-test harness.
5. Run the candidate tests, followed by 1,400 differential comparisons.

The notebook has no third-party dependencies. After completing the TODOs, it
can be run from top to bottom.

## Part 1: Basic Spread

```python
def time_to_full_infection(grid: list[list[int]]) -> int:
    ...
```

- `0` means healthy and `1` means infected.
- At the start of each day, all infected cells simultaneously infect their
  healthy orthogonal neighbors.
- Cells infected today do not propagate until tomorrow.
- Multiple initial infection sources are allowed.
- Return the number of days needed to infect every cell.
- Return `-1` if there is no infection source or any healthy cell is
  unreachable.
- Return `0` for an empty or fully infected grid.

Examples:

- One center source in a 3x3 grid: `2`.
- Two sources in opposite corners of a 3x3 grid: `2`.
- `[[1, 0, 0, 0, 0]]`: `4`.
- `[[1]]`: `0`; `[[0]]`: `-1`.

**Reference approach:** enqueue every initial `1` as a level-0 source and run
multi-source BFS. Each cell is enqueued at most once, so time and space are
both `O(R*C)`.

## Part 2: Immune Cells

```python
def time_to_full_infection_with_immunity(
    grid: list[list[int]],
) -> int:
    ...
```

Add `2 = immune`. An immune cell is permanent, never propagates infection, and
is skipped when counting neighbors, so it behaves like a wall. Return the days
required to infect every non-immune healthy cell, or `-1` if any healthy cell
is permanently isolated. Return `0` if the grid contains only infected and
immune cells.

Some variants encode immune cells as `-1` or use `. / X / I`. Clarify the
encoding before coding. This harness always uses `0 / 1 / 2`.

## Part 3: D-day Recovery to Immunity

```python
def time_to_stable_state(grid: list[list[int]], D: int) -> int:
    ...
```

The input contains only `0` and `1`. After being infected for `D` days, a cell
recovers and becomes immune (`2`). Return the first day with no active infected
cells. Some healthy cells may remain uninfected forever.

### Exact time semantics

`infection_day[r][c]` records the day a cell became infected. Initial sources
have infection day `0`. On each tick:

1. Increment `current_day`.
2. Convert every cell satisfying
   `current_day - infection_day >= D` to immune.
3. A cell that recovered on this tick does **not** spread on the same day.
4. The remaining active infected cells spread from the start-of-day snapshot.
5. A newly infected cell gets `infection_day = current_day` and first spreads
   on the next day.

Therefore, when `D=1`, every initial source recovers before spreading on day 1.
`recovery_oracle()` implements a direct day-by-day simulation. With no initial
walls:

- For `D=1`, the answer is `1` when at least one source exists.
- For `D>=2`, spread time is the 4-neighbor BFS distance, and the answer is
  `last infection day + D`.

## Part 4A: Threshold Infection

```python
def time_to_full_infection_threshold(
    grid: list[list[int]], K: int
) -> int:
    ...
```

Require `1 <= K <= 4`. A healthy cell becomes infected on the next day only if
it has at least `K` infected orthogonal neighbors at the start of the day.
Infected cells never recover, and `2` is an immune wall. Return the number of
days needed to infect every non-immune cell, or `-1` if that cannot happen.

This is no longer ordinary BFS. You can scan the full grid every day, or
maintain an infected-neighbor count and update candidates incrementally from
the current frontier. `threshold_oracle()` provides the slow synchronous
version.

## Part 4B: Death Countdown

Different variants assign different meanings to `K` and death. This mock keeps
two common interpretations in separate functions.

### Countdown version

```python
def time_to_end_with_death_countdown(
    grid: list[list[int]], D: int, K: int, N: int
) -> tuple[int, int]:
    ...
```

- The healthy-cell infection threshold remains `1`.
- Each day processes expired death and recovery deadlines before survivors
  spread.
- An active infected cell starts a countdown when it has at least `K` active
  infected neighbors.
- Its death deadline is `current_day + N` and is not reset if its neighbor
  count later decreases.
- Death wins when death and recovery deadlines occur on the same day.
- If recovery occurs first, the countdown is canceled and the cell becomes
  immune.
- Return `(days until no active infected cells remain, total deaths)`.

### Strict recovery version

```python
def time_to_end_with_death_on_recovery(
    grid: list[list[int]], D: int, K: int
) -> tuple[int, int]:
    ...
```

If a cell has at least `K` active infected neighbors **when it becomes
infected**, mark it as doomed. At its recovery deadline it becomes dead instead
of immune. Initial sources determine their doomed status from the day-0
neighbor snapshot.

## Part 4C: Composite

```python
def time_to_end_composite(
    grid: list[list[int]],
    D: int,
    infection_threshold: int,
    death_threshold: int,
    death_countdown: int,
) -> tuple[int, int]:
    ...
```

Combine immune walls, threshold infection, recovery, and countdown death. This
version must track the active set, infection day, and scheduled death on every
day. Because infected neighbors can recover or die, neighbor counts are not
monotonic and the answer is not simply `last infection time + D`.

## Part 5: Row or Column Burn Optimization

Some interviews continue with: "Each day, choose an entire row or column to
burn and minimize total deaths." This statement is not specific enough to
define a unique problem. Clarify:

- Does burning occur before or after infection spreads?
- How many days can you act, and when does the process stop?
- Do infected and healthy cells have the same death cost?
- Is the goal to prevent all spread, protect specific cells, or eliminate the
  infection?
- Is a cell counted more than once if multiple burns cover it?

Once specified, the problem may become dynamic programming, graph
cut/covering, or time-expanded search. This harness leaves Part 5 as an
interviewer follow-up rather than inventing a false canonical answer.

## BFS versus Simulation

Parts 1-2 are textbook multi-source BFS: each cell is processed once for
`O(R*C)` time. Rescanning the entire grid every day can take
`O((R*C)^2)`. Parts 3-4 add recovery, thresholds, and death states, so they are
usually clearer as synchronous `snapshot -> compute next state -> swap`
simulations. Updating the grid in place while scanning is the most common bug.

## Additional practice variants

| Variant | Rule change | Main skill |
| --- | --- | --- |
| 4-neighbor / 8-neighbor | Parameterize the neighborhood | Contract clarification and parameterized tests |
| Infection time matrix | Return each cell's infection day | BFS distance and unreachable sentinels |
| Final stable grid | Return final state instead of only time | Mutation ownership |
| Multiple infection types | Competing colors and tie-breaking | Event ordering |
| Probabilistic spread | Each edge succeeds randomly | RNG injection and statistical tests |
| Weighted directions | Direction-dependent propagation cost | Dijkstra or 0-1 BFS |
| Terrain effects | Local spread and recovery rules | Separating state from graph structure |
| In-place only | Disallow double buffering | State encoding and day stamps |
| Many queries | Ask for state on arbitrary days | Timeline precomputation |
| Huge grid | Use a sparse frontier | Event-driven simulation and memory layout |
