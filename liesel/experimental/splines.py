from functools import partial

import jax.numpy as jnp
from jax import jit

from liesel.model import Array


class EquidistantKnots:
    def __init__(
        self, x: Array, n_param: int, order: int = 3, eps: float = 0.01
    ) -> None:
        ...

        n_internal_knots = n_param - order + 1

        a = jnp.min(x)
        b = jnp.max(x)

        range_ = b - a

        min_k = a - range_ * (eps / 2)
        max_k = b + range_ * (eps / 2)

        internal_knots = jnp.linspace(min_k, max_k, n_internal_knots)

        step = internal_knots[1] - internal_knots[0]

        left_knots = jnp.linspace(min_k - (step * order), min_k - step, order)
        right_knots = jnp.linspace(max_k + step, max_k + (step * order), order)

        self.knots = jnp.concatenate((left_knots, internal_knots, right_knots))

        self.internal_knots = internal_knots
        self.order = order
        self.n_param = n_param
        self.n_knots = n_internal_knots + 2 * order


def _check_equidistant_knots(knots: Array) -> Array:
    """
    Checks if knots are equidistants.
    """
    diff = jnp.diff(knots)

    return jnp.allclose(diff, diff[0], 1e-3, 1e-3)


def _check_data_range(x: Array, knots: Array, order: int) -> bool:
    """
    Check that values in x are in the range
    [knots[order], knots[dim(knots) - order - 1]].
    """

    return (
        jnp.min(x) >= knots[order] and jnp.max(x) <= knots[knots.shape[0] - order - 1]
    )


def _check_b_spline_inputs(x: Array, knots: Array, order: int) -> None:
    if not order >= 0:
        raise ValueError("Order must non-negative")
    if not _check_equidistant_knots(knots):
        raise ValueError("Sorted knots are not equidistant")
    if not _check_data_range(x, knots, order):
        raise ValueError(
            f"Data values are not in the range                 [{knots[order]},"
            f" {knots[knots.shape[0] - order - 1]}]"
        )


# @jit
@partial(jit, static_argnums=(1, 2))
def create_equidistant_knots(x: Array, order: int = 3, n_params: int = 20) -> Array:
    """
    Create equidistant knots for B-Spline of the specified order.

    Some additional info:

    - ``dim(knots) = n_params + order + 1``
    - ``n_params = dim(knots) - order - 1``

    Parameters
    ----------
    x
        The data for which the knots are created.
    order
        A positive integer giving the order of the spline function.
        A cubic spline has order of 3.
    n_params
        Number of parameters of the B-spline.

    """
    epsilon = 0.01

    internal_k = n_params - order + 1

    a = jnp.min(x)
    b = jnp.max(x)

    min_k = a - jnp.abs(a * epsilon)
    max_k = b + jnp.abs(b * epsilon)

    internal_knots = jnp.linspace(min_k, max_k, internal_k)

    step = internal_knots[1] - internal_knots[0]

    left_knots = jnp.linspace(min_k - (step * order), min_k - step, order)
    right_knots = jnp.linspace(max_k + step, max_k + (step * order), order)

    return jnp.concatenate((left_knots, internal_knots, right_knots))
