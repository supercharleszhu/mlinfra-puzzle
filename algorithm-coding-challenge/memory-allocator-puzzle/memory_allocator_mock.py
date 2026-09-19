"""CoderPad-style memory allocator interview mock.

Implement both allocator APIs, then run this file. The fixed and deterministic
random tests compare behavior with a deliberately slow bitmap oracle.
"""

from __future__ import annotations

import random
import sys


class Allocator:
    """Pointer-only API. Target: expected O(log m) per operation."""

    def __init__(self, size: int = 1000) -> None:
        raise NotImplementedError

    def malloc(self, size: int) -> int:
        """Allocate from the leftmost sufficient gap; return -1 on exhaustion."""

        raise NotImplementedError

    def free(self, pointer: int) -> bool:
        """Free a live allocation; return False for invalid or double free."""

        raise NotImplementedError

    def get_free_memory(self) -> int:
        raise NotImplementedError

    def get_largest_free_block(self) -> int:
        raise NotImplementedError


class MemoryAllocator:
    """Explicit-size API. Target: expected O(log m) per operation."""

    def __init__(self, total_capacity: int) -> None:
        raise NotImplementedError

    def allocate(self, size: int) -> int:
        """Allocate first-fit; raise MemoryError when no contiguous gap fits."""

        raise NotImplementedError

    def free(self, address: int, size: int) -> None:
        """Free exactly one recorded allocation, validating address and size."""

        raise NotImplementedError

    def get_free_memory(self) -> int:
        raise NotImplementedError

    def get_largest_free_block(self) -> int:
        raise NotImplementedError


class _BitmapAllocatorOracle:
    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.free_bytes = [True] * capacity
        self.allocations: dict[int, int] = {}

    def malloc(self, size: int) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        run = 0
        for index, is_free in enumerate(self.free_bytes):
            run = run + 1 if is_free else 0
            if run == size:
                start = index - size + 1
                self.free_bytes[start : index + 1] = [False] * size
                self.allocations[start] = size
                return start
        return -1

    def free(self, pointer: int) -> bool:
        size = self.allocations.pop(pointer, None)
        if size is None:
            return False
        self.free_bytes[pointer : pointer + size] = [True] * size
        return True

    def get_free_memory(self) -> int:
        return sum(self.free_bytes)

    def get_largest_free_block(self) -> int:
        best = run = 0
        for is_free in self.free_bytes:
            run = run + 1 if is_free else 0
            best = max(best, run)
        return best


class _ExplicitBitmapOracle(_BitmapAllocatorOracle):
    def allocate(self, size: int) -> int:
        address = self.malloc(size)
        if address == -1:
            raise MemoryError("no contiguous free block is large enough")
        return address

    def free_explicit(self, address: int, size: int) -> None:
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
        super().free(address)


def _expect_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def _expect_raises(error_type, fn, *args) -> None:
    try:
        fn(*args)
    except error_type:
        return
    except Exception as error:
        raise AssertionError(
            f"{fn.__name__} raised {type(error).__name__}, "
            f"expected {error_type.__name__}"
        ) from error
    raise AssertionError(f"{fn.__name__}{args!r} should raise {error_type.__name__}")


def _assert_stats(allocator, free_memory: int, largest: int, label: str) -> None:
    _expect_equal(allocator.get_free_memory(), free_memory, f"{label} free")
    _expect_equal(
        allocator.get_largest_free_block(), largest, f"{label} largest"
    )


def _fixed_pointer_tests(impl) -> None:
    default = impl.Allocator()
    _assert_stats(default, 1000, 1000, "default capacity")

    allocator = impl.Allocator(100)
    a = allocator.malloc(20)
    b = allocator.malloc(30)
    c = allocator.malloc(40)
    _expect_equal((a, b, c), (0, 20, 50), "sequential allocation")
    _assert_stats(allocator, 10, 10, "after three allocations")

    _expect_equal(allocator.free(b), True, "free middle")
    d = allocator.malloc(25)
    _expect_equal(d, 20, "reuse leftmost middle gap")
    _assert_stats(allocator, 15, 10, "fragmented state")

    _expect_equal(allocator.free(a), True, "free left")
    _expect_equal(allocator.free(d), True, "merge both sides")
    _assert_stats(allocator, 60, 50, "both-neighbor merge")
    _expect_equal(allocator.free(d), False, "double free")
    _expect_equal(allocator.free(999), False, "unknown pointer")

    exact = impl.Allocator(8)
    _expect_equal(exact.malloc(8), 0, "exact fit")
    _expect_equal(exact.malloc(1), -1, "full allocator")
    _assert_stats(exact, 0, 0, "full state")
    _expect_equal(exact.free(0), True, "free exact fit")
    _assert_stats(exact, 8, 8, "fully coalesced")

    fragmented = impl.Allocator(12)
    pointers = [fragmented.malloc(3) for _ in range(4)]
    fragmented.free(pointers[0])
    fragmented.free(pointers[2])
    _expect_equal(fragmented.malloc(4), -1, "external fragmentation")
    _expect_equal(fragmented.malloc(3), 0, "leftmost first fit")

    merges = impl.Allocator(25)
    blocks = [merges.malloc(5) for _ in range(5)]
    _expect_equal(merges.free(blocks[1]), True, "no-neighbor merge case")
    _expect_equal(merges.free(blocks[3]), True, "second isolated gap")
    _expect_equal(merges.free(blocks[2]), True, "both-neighbor merge case")
    _assert_stats(merges, 15, 15, "after both-neighbor merge")
    _expect_equal(merges.free(blocks[0]), True, "right-neighbor merge case")
    _expect_equal(merges.free(blocks[4]), True, "left-neighbor merge case")
    _assert_stats(merges, 25, 25, "all four merge cases")

    _expect_raises(ValueError, impl.Allocator, 0)
    _expect_raises(ValueError, impl.Allocator, -1)
    _expect_raises(ValueError, allocator.malloc, 0)
    _expect_raises(ValueError, allocator.malloc, -3)


def _fixed_explicit_tests(impl) -> None:
    allocator = impl.MemoryAllocator(100)
    a = allocator.allocate(20)
    b = allocator.allocate(30)
    c = allocator.allocate(40)
    _expect_equal((a, b, c), (0, 20, 50), "explicit sequence")
    allocator.free(20, 30)
    d = allocator.allocate(25)
    _expect_equal(d, 20, "explicit gap reuse")
    allocator.free(0, 20)
    allocator.free(20, 25)
    _assert_stats(allocator, 60, 50, "explicit both-side merge")

    _expect_raises(ValueError, impl.MemoryAllocator, 0)
    _expect_raises(ValueError, allocator.allocate, 0)
    _expect_raises(MemoryError, allocator.allocate, 51)
    _expect_raises(ValueError, allocator.free, -1, 1)
    _expect_raises(ValueError, allocator.free, 100, 1)
    _expect_raises(ValueError, allocator.free, 90, 11)
    _expect_raises(ValueError, allocator.free, 50, 39)
    allocator.free(50, 40)
    _expect_raises(ValueError, allocator.free, 50, 40)
    _assert_stats(allocator, 100, 100, "explicit full merge")


def _compare_stats(actual, expected, label: str) -> None:
    _expect_equal(
        actual.get_free_memory(),
        expected.get_free_memory(),
        f"{label} free memory",
    )
    _expect_equal(
        actual.get_largest_free_block(),
        expected.get_largest_free_block(),
        f"{label} largest gap",
    )


def _random_pointer_tests(impl, rng: random.Random) -> None:
    for case in range(200):
        capacity = rng.randint(1, 80)
        actual = impl.Allocator(capacity)
        expected = _BitmapAllocatorOracle(capacity)
        known_pointers: list[int] = []
        for step in range(200):
            if rng.random() < 0.62:
                size = rng.randint(1, capacity + 10)
                actual_pointer = actual.malloc(size)
                expected_pointer = expected.malloc(size)
                _expect_equal(
                    actual_pointer,
                    expected_pointer,
                    f"pointer random case {case}, step {step}",
                )
                if actual_pointer != -1:
                    known_pointers.append(actual_pointer)
            else:
                if known_pointers and rng.random() < 0.8:
                    pointer = rng.choice(known_pointers)
                else:
                    pointer = rng.randint(-2, capacity + 2)
                _expect_equal(
                    actual.free(pointer),
                    expected.free(pointer),
                    f"pointer free case {case}, step {step}",
                )
            _compare_stats(actual, expected, f"pointer case {case}, step {step}")


def _random_explicit_tests(impl, rng: random.Random) -> None:
    for case in range(200):
        capacity = rng.randint(1, 80)
        actual = impl.MemoryAllocator(capacity)
        expected = _ExplicitBitmapOracle(capacity)
        live: dict[int, int] = {}
        for step in range(200):
            if rng.random() < 0.62:
                size = rng.randint(1, capacity + 10)
                try:
                    expected_address = expected.allocate(size)
                except MemoryError:
                    _expect_raises(MemoryError, actual.allocate, size)
                else:
                    actual_address = actual.allocate(size)
                    _expect_equal(
                        actual_address,
                        expected_address,
                        f"explicit allocate case {case}, step {step}",
                    )
                    live[actual_address] = size
            elif live and rng.random() < 0.85:
                address = rng.choice(list(live))
                size = live.pop(address)
                actual.free(address, size)
                expected.free_explicit(address, size)
            else:
                if live and rng.random() < 0.5:
                    address = rng.choice(list(live))
                    size = live[address] + 1
                else:
                    address = capacity
                    size = 1
                _expect_raises(ValueError, actual.free, address, size)
                _expect_raises(ValueError, expected.free_explicit, address, size)
            _compare_stats(actual, expected, f"explicit case {case}, step {step}")


def _scaling_smoke_test(impl) -> None:
    allocator = impl.Allocator(25_000)
    pointers = [allocator.malloc(1) for _ in range(25_000)]
    for pointer in pointers[::2]:
        if not allocator.free(pointer):
            raise AssertionError("scaling test free failed")
    for expected in range(0, 25_000, 2):
        _expect_equal(
            allocator.malloc(1), expected, "scaling test leftmost fit"
        )
    _assert_stats(allocator, 0, 0, "scaling test final")


def _run_all_tests(impl) -> None:
    _fixed_pointer_tests(impl)
    _fixed_explicit_tests(impl)
    print("PASS fixed tests")
    rng = random.Random(20260915)
    _random_pointer_tests(impl, rng)
    _random_explicit_tests(impl, rng)
    print("PASS 80,000 deterministic random operations")
    _scaling_smoke_test(impl)
    print("PASS 50,000-operation fragmentation smoke test")


def run_all_tests() -> None:
    _run_all_tests(sys.modules[__name__])


if __name__ == "__main__":
    run_all_tests()
