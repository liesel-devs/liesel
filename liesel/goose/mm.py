"""
# Inverse mass matrix tuner

This module uses the error codes 70-79.
"""

import jax
import jax.numpy as jnp

from .types import Array, Position

_vravel = jax.vmap(jnp.ravel, in_axes=0, out_axes=0)


def _history_to_matrix(history: Position) -> Array:
    return jnp.column_stack([_vravel(x) for x in history.values()])


def tune_inv_mm_diag(history: Position) -> Array:
    """
    Tunes an inverse mass vector with the sample variances of the history.
    """

    matrix = _history_to_matrix(history)
    var = jnp.var(matrix, axis=0, ddof=1)
    var = jnp.atleast_1d(var)
    var = var + 0.001

    return var


def tune_inv_mm_full(history: Position) -> Array:
    """
    Tunes an inverse mass matrix with the sample variance-covariance matrix
    of the history.
    """

    matrix = _history_to_matrix(history)
    cov = jnp.cov(matrix, rowvar=False)
    cov = jnp.atleast_2d(cov)

    # stan regularization, see:
    # https://github.com/stan-dev/stan/blob/v2.28.2/src/stan/mcmc/covar_adaptation.hpp#L27-L29

    # n = matrix.shape[0]
    # cov = (n / (n + 5.0)) * cov
    # cov = cov.at[jnp.diag_indices_from(cov)].add(0.001 * (5.0 / (n + 5.0)))
    cov = cov.at[jnp.diag_indices_from(cov)].add(0.001)

    return cov
