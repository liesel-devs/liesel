from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline

from liesel.splines import basis_matrix, equidistant_knots, pspline_penalty


def test_knots_creation():
    x = jnp.arange(-1000, 1000)

    nparams = [10, 100, 2000]
    orders = [0, 1, 2, 3, 12]

    for nparam, order in product(nparams, orders):
        if nparam < order:
            with pytest.raises(ValueError):
                equidistant_knots(x, nparam, order)
            continue
        else:
            knots = equidistant_knots(x, nparam, order)
            assert len(knots) == nparam + order + 1
            diff = jnp.diff(knots)
            assert jnp.allclose(diff, diff[0], atol=1e-3)

    with pytest.raises(ValueError):
        equidistant_knots(x, nparam, -1)


class TestBasisMatrix:
    def test_shape(self):
        n = 200
        x = jnp.arange(0, n)
        n_params = 40
        order = 3

        knots = equidistant_knots(x, n_params, order)

        X = basis_matrix(x, knots, order)

        assert X.shape[0] == n
        assert X.shape[1] == n_params

    def test_vs_scipy(self):
        x = jnp.arange(0, 200)
        n_params = 40
        order = 3

        knots = equidistant_knots(x, n_params, order)

        X = basis_matrix(x, knots, order)
        beta = np.random.randn(X.shape[1])
        scipy_spl = BSpline(knots, beta, order)

        assert np.allclose(X @ beta, scipy_spl(x), 1e-3, 1e-3)

    def test_x_outside_range(self):
        n = 200
        x = jnp.arange(0, n)
        n_params = 40
        order = 3

        knots = equidistant_knots(x, n_params, order)

        with pytest.raises(ValueError):
            basis_matrix(-1, knots, order)

        bmat = basis_matrix(-1, knots, order, outer_ok=True)
        assert bmat.shape == (1, n_params)

    def test_jit(self):
        n = 200
        x = jnp.arange(0, n)
        n_params = 40
        order = 3

        knots = equidistant_knots(x, n_params, order)

        X = jax.jit(basis_matrix, static_argnames=("order", "outer_ok"))(
            x, knots, order, outer_ok=True
        )

        assert X.shape[0] == n
        assert X.shape[1] == n_params

        basis_matrix_fix = partial(basis_matrix, order=3, outer_ok=True)

        X = jax.jit(basis_matrix_fix)(x, knots)

        assert X.shape[0] == n
        assert X.shape[1] == n_params

    def test_vectorize(self):
        n = 200
        x = jnp.arange(0, n)
        x = jnp.c_[x, x].T
        n_params = 40
        order = 3

        knots = equidistant_knots(x, n_params, order)
        basis_matrix_vec = jnp.vectorize(
            basis_matrix, excluded=(1, 2, 3), signature="(n)->(n,p)"
        )

        B = basis_matrix_vec(x, knots, order, True)

        assert B.shape == (2, n, n_params)


class TestPsplinePenalty:
    def test_shape(self):
        d = 10

        P = pspline_penalty(d)
        assert P.shape == (d, d)

    def test_rank(self):
        d = 10
        for diff in [1, 2, 3]:
            P = pspline_penalty(d, diff=diff)
            assert jnp.linalg.matrix_rank(P) == d - diff
