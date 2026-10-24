"""Deterministic local checker for the tensor broadcasting puzzles."""

from __future__ import annotations

import ast
import inspect
import operator
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch


tensor = torch.tensor


@dataclass(frozen=True)
class TensorSpec:
    dims: tuple[int | str, ...]
    dtype: torch.dtype


class TensorType:
    """Small runtime replacement for the upstream torchtyping annotations."""

    @classmethod
    def __class_getitem__(cls, item):
        values = item if isinstance(item, tuple) else (item,)
        dtype = torch.int64
        if values and values[-1] in (bool, float, int):
            dtype_marker = values[-1]
            values = values[:-1]
            dtype = {
                bool: torch.bool,
                float: torch.float64,
                int: torch.int64,
            }[dtype_marker]
        return TensorSpec(tuple(values), dtype)


TT = TensorType

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}


def _evaluate_dimension(value: int | str, sizes: dict[str, int]) -> int:
    if isinstance(value, int):
        return value

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Name):
            return sizes[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        raise ValueError(f"unsupported dimension expression: {value!r}")

    return evaluate(ast.parse(value, mode="eval").body)


def _dimension_names(spec: TensorSpec) -> set[str]:
    names: set[str] = set()
    for dim in spec.dims:
        if isinstance(dim, str):
            names.update(
                node.id
                for node in ast.walk(ast.parse(dim, mode="eval"))
                if isinstance(node, ast.Name)
            )
    return names


def _shape(spec: TensorSpec, sizes: dict[str, int]) -> tuple[int, ...]:
    return tuple(_evaluate_dimension(dim, sizes) for dim in spec.dims)


def _numpy_dtype(dtype: torch.dtype):
    return {
        torch.bool: np.bool_,
        torch.float64: np.float64,
        torch.int64: np.int64,
    }[dtype]


def _random_array(
    rng: np.random.Generator,
    spec: TensorSpec,
    sizes: dict[str, int],
) -> np.ndarray:
    shape = _shape(spec, sizes)
    if spec.dtype == torch.bool:
        return rng.integers(0, 2, size=shape, dtype=np.int64).astype(np.bool_)
    if spec.dtype == torch.float64:
        return rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    return rng.integers(-5, 6, size=shape, dtype=np.int64)


@dataclass(frozen=True)
class PuzzleTest:
    name: str
    problem: Callable
    problem_spec: Callable
    add_sizes: tuple[str, ...]
    constraint: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
    cases: int = 30

    def run(self) -> None:
        signature = inspect.signature(self.problem)
        tensor_parameters = {
            name: parameter.annotation
            for name, parameter in signature.parameters.items()
            if isinstance(parameter.annotation, TensorSpec)
        }
        output_spec = signature.return_annotation
        if not isinstance(output_spec, TensorSpec):
            raise TypeError(f"{self.name}: missing tensor return annotation")

        dimension_names = set()
        for spec in (*tensor_parameters.values(), output_spec):
            dimension_names.update(_dimension_names(spec))
        ordered_names = sorted(dimension_names)

        for case in range(self.cases):
            rng = np.random.default_rng(case)
            candidates = np.arange(1, 6)
            rng.shuffle(candidates)
            sizes = {
                name: int(candidates[index % len(candidates)])
                for index, name in enumerate(ordered_names)
            }

            data: dict[str, np.ndarray] = {
                name: _random_array(rng, spec, sizes)
                for name, spec in tensor_parameters.items()
            }
            data["return"] = np.zeros(
                _shape(output_spec, sizes),
                dtype=_numpy_dtype(output_spec.dtype),
            )
            data = self.constraint(data)

            expected_list = np.asarray(data["return"]).copy().tolist()
            self.problem_spec(
                *(data[name] for name in tensor_parameters),
                expected_list,
            )
            expected = torch.as_tensor(expected_list, dtype=output_spec.dtype)

            arguments = []
            for name, parameter in signature.parameters.items():
                annotation = parameter.annotation
                if isinstance(annotation, TensorSpec):
                    arguments.append(
                        torch.as_tensor(data[name], dtype=annotation.dtype)
                    )
                elif name in self.add_sizes:
                    arguments.append(sizes[name])
                else:
                    raise TypeError(
                        f"{self.name}: unsupported scalar argument {name!r}"
                    )

            try:
                actual = torch.as_tensor(self.problem(*arguments))
            except NotImplementedError as error:
                raise AssertionError(f"{self.name}: implementation is missing") from error

            try:
                actual = torch.broadcast_to(actual, expected.shape)
            except RuntimeError as error:
                raise AssertionError(
                    f"{self.name}: got shape {tuple(actual.shape)}, "
                    f"expected broadcastable to {tuple(expected.shape)}"
                ) from error

            actual = actual.to(expected.dtype)
            try:
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-6 if expected.is_floating_point() else 0,
                    atol=1e-6 if expected.is_floating_point() else 0,
                )
            except AssertionError as error:
                raise AssertionError(
                    f"{self.name}: failed deterministic case {case} "
                    f"with sizes {sizes}\n{error}"
                ) from error


def make_test(
    name: str,
    problem: Callable,
    problem_spec: Callable,
    add_sizes: Sequence[str] | None = None,
    constraint: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
    | None = None,
) -> PuzzleTest:
    """Build a deterministic test compatible with the upstream notebook API."""

    return PuzzleTest(
        name=name,
        problem=problem,
        problem_spec=problem_spec,
        add_sizes=tuple(add_sizes or ()),
        constraint=constraint or (lambda data: data),
    )


def run_test(test: PuzzleTest) -> None:
    test.run()
    print(f"Correct! {test.name}: {test.cases} deterministic cases")
