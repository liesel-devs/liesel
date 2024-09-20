"""
Basic functionality for using B-splines in Liesel.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap

from liesel.model import Array


def equidistant_knots(
    x: Array, n_param: int, order: int = 3, eps: float = 0.01
) -> Array:
    """
    Create equidistant knots for a B-spline of the specified order.

    Parameters
    ----------
    x
        A 1d array of input data.
    order
        A positive integer giving the order of the spline function.
        A cubic spline has an order of 3.
    n_param
        Number of parameters of the B-spline.
    eps
        A factor by which the range of the interior knots is stretched. The range of
        interior knots will thus be ``eps * (jnp.max(x) - jnp.min(x))``.

    Returns
    -------
    A 1d array

    Notes
    -----

    Some additional info:

    - ``dim(knots) = n_params + order + 1``
    - ``n_param = dim(knots) - order - 1``
    - ``n_interior_knots = n_param - order + 1``

    """
    if order < 0:
        raise ValueError(f"Invalid {order=}.")

    if n_param < order:
        raise ValueError(f"{n_param=} must not be smaller than {order=}.")

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

    knots = jnp.concatenate((left_knots, internal_knots, right_knots))
    return knots


@partial(jit, static_argnames="order")
def _build_basis_vector(x: Array, knots: Array, order: int) -> Array:
    """
    Builds a vector of length ``dim(knots) - order - 1``. Each entry ``i`` is
    iterativaly updated. At time m, the entry i is the evaluation of the basis function
    at the observed value for the m-th order and for the i-th knot. The creation of the
    matrix needs a row-wise (order) loop (f1) and a column-wise (knot index) loop (f2).
    """
    k = knots.shape[0] - order - 1
    bv = jnp.full(knots.shape[0] - 1, jnp.nan)

    def basis_per_order(m, bv):
        def basis_per_knot(i, bv):
            def base_case(bv):
                return bv.at[i].set(
                    jnp.where(x >= knots[i], 1.0, 0.0)
                    * jnp.where(x < knots[i + 1], 1.0, 0.0)
                )

            def recursive_case(bv):
                b1 = (x - knots[i]) / (knots[i + m] - knots[i]) * bv[i]
                b2 = (
                    (knots[i + m + 1] - x)
                    / (knots[i + m + 1] - knots[i + 1])
                    * bv[i + 1]
                )

                return bv.at[i].set(b1 + b2)

            return lax.cond(m == 0, base_case, recursive_case, bv)

        return lax.fori_loop(0, k + order, basis_per_knot, bv)

    return lax.fori_loop(0, order + 1, basis_per_order, bv)[:k]


def basis_matrix(
    x: Array, knots: Array, order: int = 3, outer_ok: bool = False
) -> Array:
    """
    Builds a B-spline basis matrix.

    Parameters
    ----------
    x
        A 1d array of input data.
    knots
        A 1d array of knots. The knots will be sorted.
    order
        A positive integer giving the order of the spline function. \
        A cubic spline has an order of 3.
    outer_ok
        If ``False`` (default), values of x outside the range of interior knots \
        cause an error. If ``True``, they are allowed.

    Returns
    -------
    A 2d array, the B-spline basis matrix.


    Notes
    -----
    Under the hood, instead of applying the recursive
    definition of B-splines, a matrix of (order + 1) rows and (dim(knots) - order - 1)
    columns for each value in x is created. This matrix store the evaluation of
    the basis function at the observed value for the m-th order and for the i-th knot.

    .. rubric:: Jit-compilation

    The ``basis_matrix`` function internally uses a jit-compiled function to do the
    heavy lifting. However, you may want to make ``basis_matrix`` itself jit-compilable.
    In this case, you need to define the arguments ``order`` and ``outer_ok`` as
    static arguments. Further, ``outer_ok`` needs to be fixed to ``True``.

    If you just want to set up a basis matrix once, it is usually not necessary to go
    through this process.

    Example:

    .. code-block:: python

        from liesel.contrib.splines import equidistant_knots, basis_matrix

        x = jnp.linspace(-2.0, 2.0, 30)
        knots = equidistant_knots(x, n_param=10, order=3)

        basis_matrix_jit = jax.jit(basis_matrix, static_argnames=("order", "outer_ok"))

        B = basis_matrix_jit(x, knots, order, outer_ok=True)

    Another suitable way to go is to use ``functools.partial``::

        from functools import partial
        from liesel.contrib.splines import equidistant_knots, basis_matrix

        x = jnp.linspace(-2.0, 2.0, 30)
        knots = equidistant_knots(x, n_param=10, order=3)

        basis_matrix_fix = partial(basis_matrix, order=3, outer_ok=True)
        basis_matrix_jit = jax.jit(basis_matrix_fix)

        B = basis_matrix_jit(x, knots)

    """
    if order < 0:
        raise ValueError(f"Invalid {order=}.")

    # if x is a scalar, this ensures that the function still works
    x = jnp.atleast_1d(x)

    knots = jnp.sort(knots)

    if not outer_ok:
        min_ = knots[order]
        max_ = knots[knots.shape[0] - order - 1]
        geq_min = jnp.min(x) >= min_
        leq_max = jnp.max(x) <= max_
        if not geq_min and leq_max:
            raise ValueError(
                f"Values of x are not inside the range of interior knots, [{min_},"
                f" {max_}]"
            )

    design_matrix = vmap(lambda x: _build_basis_vector(x, knots, order))(x)

    return design_matrix


def pspline_penalty(d: int, diff: int = 2):
    """
    Builds an (n_param x n_param) P-spline penalty matrix.

    Parameters
    ----------
    d
        Integer, dimension of the matrix. Corresponds to the number of parameters \
        in a P-spline.
    diff
        Order of the differences used in constructing the penalty matrix. The default \
        of ``diff=2`` corresponds to the common P-spline default of penalizing second \
        differences.

    Returns
    -------
    A 2d array, the penalty matrix.
    """
    D = jnp.diff(jnp.identity(d), diff, axis=0)
    return D.T @ D
