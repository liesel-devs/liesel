from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from functools import partial
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf
from tensorflow_probability.substrates.jax.distributions import mvn_linear_operator

Array = Any


@partial(jnp.vectorize, excluded=(1, 2), signature='(l)->(n,n)')
def fill_lower_diag(vec_L: Array, n: int, k: int = 0) -> Array:
    """
    Given a vector which parametrize a lower-traingular matrix, 
    reconstructs the matrix where the entries of the input vector
    are arranged in a row-column order.
    """
    mask = np.tri(n, k=k, dtype=bool)
    out = jnp.zeros((n, n), dtype=vec_L.dtype)

    return out.at[mask].set(vec_L)


@partial(jnp.vectorize, excluded=(1,), signature='(l)->(n,n)')
def inverse_log_cholesky_parametrization(log_cholesky: Array, n: int) -> Array:
    """
    Given a vector representing the log-cholesky parametrization,
    reconstructs the lower triangular matrix of the the cholesky
    decomposition.
    """
    log_cholesky_tril = fill_lower_diag(log_cholesky, n)

    log_cholesky_tril = jax.lax.fori_loop(
        0,
        n,
        lambda i, lcct: lcct.at[(i, i)].set(jnp.exp(lcct[i][i])),
        log_cholesky_tril,
    )

    return log_cholesky_tril


@partial(jnp.vectorize, signature='(k,k)->()')
def _log_det_tril(a):
    return jnp.sum(jnp.log(jnp.diag(a)))


@partial(jnp.vectorize, signature='(k)->()')
def _log_prob_standard_normal(z):
    return jnp.sum(tfd.Normal(loc=0., scale=1.).log_prob(z))


@partial(jnp.vectorize, signature='(n),(n),(n,n)->()')
def mvn_precision_chol_log_prob(x: Array, loc: Array, precision_matrix_chol: Array) -> Array:
    """
    Returns the log-density of a multivariate normal distribution parametrized by
    the Cholesky decomposition of the precision matrix.
    """
    z = (x - loc) @ precision_matrix_chol
    log_prob = _log_prob_standard_normal(z)
    log_det = _log_det_tril(precision_matrix_chol)

    return log_prob + log_det


@partial(jnp.vectorize, excluded=(2,), signature='(k,k),(k)->(k)')
def _triangular_solve(a, b, lower):
    return jax.lax.linalg.triangular_solve(a, b, lower=lower)


class MultivariateNormalLogCholeskyParametrization(tfd.Distribution):
    def __init__(
        self,
        loc: Array,
        log_cholesky_parametrization: Array,
        d: int,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "MultivariateNormalLogCholeskyParametrization",
    ):
        parameters = dict(locals())

        loc_ = jnp.array(loc)

        self._d = d
        self._loc = jnp.repeat(loc_, self._d) if loc_.ndim == 0 else loc
        self._log_cholesky_parametrization = log_cholesky_parametrization
        self._cholesky_precision = inverse_log_cholesky_parametrization(
            self._log_cholesky_parametrization, self._d
        )

        cholesky_precision_shape = jnp.shape(log_cholesky_parametrization)[:-1]
        loc_batches = jnp.shape(loc)[:-1]
        self._broadcast_batch_shape = jnp.broadcast_shapes(
            cholesky_precision_shape,
            loc_batches
        )

        super().__init__(
            dtype=self._cholesky_precision.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name
        )


    @property
    def cholesky_precision(self) -> Array:
        return self._cholesky_precision
    

    @property
    def log_cholesky_parametrization(self) -> Array:
        return self._log_cholesky_parametrization


    @property
    def loc(self) -> Array:
        return self._loc


    def _sample_n(self, n, seed) -> Array:
        shape = [n] + self.batch_shape + self.event_shape
        z = jax.random.normal(seed, shape)

        return jnp.add(self._loc, _triangular_solve(self._cholesky_precision, z, True))


    def _log_prob(self, x: Array) -> Array | float:
        return mvn_precision_chol_log_prob(
            x,
            self._loc,
            self._cholesky_precision
        )


    def _event_shape(self):
        return tf.TensorShape((jnp.shape(self._loc)[-1],))


    def _event_shape_tensor(self):
        return jnp.array((jnp.shape(self._loc)[-1],), dtype=jnp.int32)


    def _batch_shape(self):
        return tf.TensorShape(self._broadcast_batch_shape)


    def _batch_shape_tensor(self):
        return jnp.array(self._broadcast_batch_shape, dtype=jnp.int32)
