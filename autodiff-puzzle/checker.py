"""Minimal Jacobian checker for the autodiff puzzles."""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp


Array = jax.Array


class Shaped:
    """Runtime-neutral replacement for the notebook's shape annotations."""

    def __class_getitem__(cls, item):
        return Array


_RNG = onp.random.default_rng(0)


def _input(shape):
    return np.asarray(_RNG.uniform(-0.5, 0.5, size=shape))


def _assert_jacobian(name, actual, expected):
    actual_array = onp.asarray(actual)
    expected_array = onp.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{name} shape mismatch: got {actual_array.shape}, "
            f"expected {expected_array.shape}"
        )
    onp.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"{name} is incorrect",
    )


def check(f, jac, in_shape, in_shape2=None):
    """Compare a hand-written Jacobian with JAX automatic differentiation."""

    x = _input(in_shape)
    if in_shape2 is None:
        _assert_jacobian("dx", jac(x), jax.jacrev(f)(x))
    else:
        y = _input(in_shape2)
        actual_x, actual_y = jac(x, y)
        _assert_jacobian("dx", actual_x, jax.jacrev(f, argnums=0)(x, y))
        _assert_jacobian("dy", actual_y, jax.jacrev(f, argnums=1)(x, y))
    print("Passed")
