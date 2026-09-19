"""Expected O(log m) augmented-treap solution."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass

import memory_allocator_mock as mock


@dataclass
class _Node:
    start: int
    size: int
    priority: float
    left: _Node | None = None
    right: _Node | None = None
    max_size: int = 0
    total_size: int = 0

    def __post_init__(self) -> None:
        self.max_size = self.size
        self.total_size = self.size


def _max_size(node: _Node | None) -> int:
    return node.max_size if node else 0


def _total_size(node: _Node | None) -> int:
    return node.total_size if node else 0


def _update(node: _Node | None) -> None:
    if node:
        node.max_size = max(node.size, _max_size(node.left), _max_size(node.right))
        node.total_size = (
            node.size + _total_size(node.left) + _total_size(node.right)
        )


def _split(
    root: _Node | None, start: int
) -> tuple[_Node | None, _Node | None]:
    if root is None:
        return None, None
    if root.start < start:
        root.right, right = _split(root.right, start)
        _update(root)
        return root, right
    left, root.left = _split(root.left, start)
    _update(root)
    return left, root


def _insert(root: _Node | None, node: _Node) -> _Node:
    if root is None:
        return node
    if node.priority < root.priority:
        node.left, node.right = _split(root, node.start)
        _update(node)
        return node
    if node.start < root.start:
        root.left = _insert(root.left, node)
    else:
        root.right = _insert(root.right, node)
    _update(root)
    return root


def _merge(left: _Node | None, right: _Node | None) -> _Node | None:
    if left is None:
        return right
    if right is None:
        return left
    if left.priority < right.priority:
        left.right = _merge(left.right, right)
        _update(left)
        return left
    right.left = _merge(left, right.left)
    _update(right)
    return right


def _erase(root: _Node | None, start: int) -> _Node | None:
    if root is None:
        raise AssertionError(f"free block {start} is missing")
    if start == root.start:
        return _merge(root.left, root.right)
    if start < root.start:
        root.left = _erase(root.left, start)
    else:
        root.right = _erase(root.right, start)
    _update(root)
    return root


def _leftmost_fit(root: _Node | None, size: int) -> _Node | None:
    if root is None or root.max_size < size:
        return None
    if root.left and root.left.max_size >= size:
        return _leftmost_fit(root.left, size)
    if root.size >= size:
        return root
    return _leftmost_fit(root.right, size)


def _predecessor(root: _Node | None, start: int) -> _Node | None:
    result = None
    while root:
        if root.start < start:
            result = root
            root = root.right
        else:
            root = root.left
    return result


def _successor(root: _Node | None, start: int) -> _Node | None:
    result = None
    while root:
        if root.start > start:
            result = root
            root = root.left
        else:
            root = root.right
    return result


class _TreapAllocator:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.allocations: dict[int, int] = {}
        self._rng = random.Random(0xA110CA7E)
        self._root: _Node | None = None
        self._add_gap(0, capacity)

    def _add_gap(self, start: int, size: int) -> None:
        self._root = _insert(
            self._root, _Node(start, size, self._rng.random())
        )

    def _allocate_or_none(self, size: int) -> int | None:
        if size <= 0:
            raise ValueError("size must be positive")
        gap = _leftmost_fit(self._root, size)
        if gap is None:
            return None
        start, gap_size = gap.start, gap.size
        self._root = _erase(self._root, start)
        if gap_size > size:
            self._add_gap(start + size, gap_size - size)
        self.allocations[start] = size
        return start

    def _release(self, start: int, size: int) -> None:
        left = _predecessor(self._root, start)
        right = _successor(self._root, start)
        merged_start = start
        merged_size = size

        if left and left.start + left.size == start:
            merged_start = left.start
            merged_size += left.size
            self._root = _erase(self._root, left.start)
        if right and start + size == right.start:
            merged_size += right.size
            self._root = _erase(self._root, right.start)
        self._add_gap(merged_start, merged_size)

    def get_free_memory(self) -> int:
        return _total_size(self._root)

    def get_largest_free_block(self) -> int:
        return _max_size(self._root)


class Allocator(_TreapAllocator):
    def __init__(self, size: int = 1000) -> None:
        super().__init__(size)

    def malloc(self, size: int) -> int:
        pointer = self._allocate_or_none(size)
        return -1 if pointer is None else pointer

    def free(self, pointer: int) -> bool:
        size = self.allocations.pop(pointer, None)
        if size is None:
            return False
        self._release(pointer, size)
        return True


class MemoryAllocator(_TreapAllocator):
    def __init__(self, total_capacity: int) -> None:
        super().__init__(total_capacity)

    def allocate(self, size: int) -> int:
        address = self._allocate_or_none(size)
        if address is None:
            raise MemoryError("no contiguous free block is large enough")
        return address

    def free(self, address: int, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if address < 0 or address >= self.capacity:
            raise ValueError("address is outside the allocator")
        if address + size > self.capacity:
            raise ValueError("range exceeds allocator capacity")
        recorded = self.allocations.get(address)
        if recorded is None:
            raise ValueError("address is not a live allocation")
        if recorded != size:
            raise ValueError("size does not match the live allocation")
        del self.allocations[address]
        self._release(address, size)


def run_all_tests() -> None:
    mock._run_all_tests(sys.modules[__name__])


if __name__ == "__main__":
    run_all_tests()
