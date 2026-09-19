# Memory Allocator Mock Interview

This allocator exercise can be copied into CoderPad or run directly in
Colab/Jupyter. It focuses on first-fit allocation, free-block coalescing,
fragmentation, and improving an `O(n)` linked-list baseline to expected
`O(log m)` operations.

## Running the exercise

```bash
cd algorithm-coding-challenge/memory-allocator-puzzle

# Single-file candidate mock; initially fails at the first TODO.
python3 memory_allocator_mock.py

# Augmented-treap reference implementation.
python3 solution.py

# Generate and validate the Colab/Jupyter version.
python3 scripts/generate_notebook.py
python3 scripts/check_notebook.py
```

`memory_allocator_mock.py` contains the function signatures, fixed tests, a
bitmap slow oracle, 80,000 deterministic random operations, a
fragmentation/scaling test, and `run_all_tests()`. It uses only the Python
standard library.

## Part 1: Pointer-only API

```python
class Allocator:
    def __init__(self, size: int = 1000) -> None: ...
    def malloc(self, size: int) -> int: ...
    def free(self, pointer: int) -> bool: ...
```

- The address space is the half-open interval `[0, size)` and starts entirely
  free.
- `malloc(size)` uses **first-fit by address**: select the lowest-address free
  gap large enough for the request and allocate from its left edge.
- Return `-1` when no contiguous gap can satisfy the request.
- Raise `ValueError` when `size <= 0`.
- `free(pointer)` releases one complete allocation and automatically merges
  adjacent free gaps.
- Return `False` for an unknown pointer or double-free.

Introspection helpers:

```python
def get_free_memory(self) -> int: ...
def get_largest_free_block(self) -> int: ...
```

These expose external fragmentation: sufficient total free memory does not
guarantee that a sufficiently large contiguous block exists.

## Part 2: Explicit-size API

```python
class MemoryAllocator:
    def __init__(self, total_capacity: int) -> None: ...
    def allocate(self, size: int) -> int: ...
    def free(self, address: int, size: int) -> None: ...
```

The allocation strategy matches Part 1, with these contract changes:

- `allocate` raises `MemoryError` when no contiguous block is large enough.
- The caller must pass the exact `(address, size)` pair back to `free`.
- A non-positive size, out-of-range interval, unknown address, incorrect size,
  or double-free raises `ValueError`.
- A `{address: size}` live-allocation map validates every free request.

Example:

```python
alloc = MemoryAllocator(100)
a = alloc.allocate(20)  # 0
b = alloc.allocate(30)  # 20
c = alloc.allocate(40)  # 50
alloc.free(b, 30)
d = alloc.allocate(25)  # 20; leaves free [45, 50)
alloc.free(a, 20)
alloc.free(d, 25)       # merges into free [0, 50)
```

## The four free/merge cases

When freeing `[start, end)`, locate its predecessor and successor by address:

| Left adjacent | Right adjacent | Operation |
| --- | --- | --- |
| No | No | Insert a new gap |
| Yes | No | Extend the left gap |
| No | Yes | Merge the freed block with the right gap |
| Yes | Yes | Collapse all three intervals into one gap |

Only adjacent intervals may be merged. A live allocation can never be skipped.

## Complexity progression

### Baseline: sorted doubly linked list

Store free gaps in start-address order using nodes containing
`{start, size, prev, next}`:

- `allocate` scans linearly for first fit: `O(m)`.
- `free` scans linearly for the insertion point and then merges neighbors in
  `O(1)`, for `O(m)` overall.
- Space is `O(m)`.

This is a reasonable first implementation but does not satisfy the performance
follow-up.

### Reference: augmented treap

`solution.py` uses a randomized treap keyed by gap start address. Every
subtree stores:

- `max_size`: its largest gap, allowing the search to skip any subtree that
  cannot satisfy the request.
- `total_size`: its total number of free bytes.

`_leftmost_fit` checks the left subtree, then the current node, then the right
subtree. This preserves **leftmost first-fit**, not best-fit. Allocation,
predecessor/successor lookup, insertion, erasure, and coalescing take expected
`O(log m)`. The two statistics are available in `O(1)`, and space is `O(m)`.

For deterministic worst-case `O(log m)`, replace the treap with an AVL or
red-black tree while retaining the same subtree augmentation.

## Follow-ups

| Topic | Discussion direction |
| --- | --- |
| Alignment | Round requests up and account for padding before an aligned gap start |
| Best-fit | Maintain a second `(size, start)` index and choose the smallest sufficient gap |
| Realloc | Grow into the right gap when possible; otherwise allocate, copy, and free |
| Thread safety | Begin with a global lock, then discuss size-class or arena sharding |
| Fragmentation | Segregated free lists, buddy allocation, and compaction |
| Metadata overhead | Implicit free lists, in-memory headers, and boundary tags |
| Fixed-size blocks | Bitmap or slab allocation |
| Invalid access | Generation IDs, guard regions, and poisoning; an allocator alone cannot fully prevent use-after-free |

A buddy system uses power-of-two blocks to make splitting and coalescing fast,
at the cost of internal fragmentation. Boundary tags place metadata at block
boundaries so an allocator can locate the previous block when metadata lives
inside the managed memory.

## Test coverage

- Exact fit, full memory, and capacity boundaries.
- Leftmost gap reuse and external fragmentation.
- No-, left-, right-, and both-neighbor merge cases.
- Invalid size, unknown pointer, incorrect-size free, and double-free.
- Random differential testing against the bitmap slow oracle.
- Search and coalescing after creating many fragmented one-byte blocks.
