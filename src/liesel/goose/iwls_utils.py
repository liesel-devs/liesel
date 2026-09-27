"""
Utilities for the IWLS sampler.
"""

import jax
import jax.numpy as jnp
import jax.scipy
from jax.typing import ArrayLike

from .types import KeyArray

triangular_solve = jax.lax.linalg.triangular_solve


def solve(chol_lhs: ArrayLike, rhs: ArrayLike) -> jax.Array:
    """
    Solves a system of linear equations `chol_lhs @ x = rhs` for x by applying
    forward and backward substitution. Returns x.

    Parameters
    ----------
    chol_lhs
        The lower triangular matrix of the Cholesky decomposition.
    rhs
        The right-hand side of the system.
    """

    tmp = triangular_solve(chol_lhs, rhs, left_side=True, lower=True)
    return triangular_solve(chol_lhs, tmp, lower=True)


def mvn_log_prob(x: ArrayLike, mean: ArrayLike, chol_inv_cov: ArrayLike) -> jax.Array:
    """
    Returns the log-density of a multivariate normal distribution.

    Parameters
    ----------
    x
        The vector of observations.
    mean
        The mean vector.
    chol_inv_cov
        The lower triangular matrix of the Cholesky decomposition of the inverse
        variance.
    """

    standardized = (jnp.asarray(x) - jnp.asarray(mean)) @ jnp.asarray(chol_inv_cov)
    log_prob = jnp.sum(jax.scipy.stats.norm.logpdf(standardized))
    adjustment = jnp.sum(jnp.log(jnp.diag(chol_inv_cov)))
    return log_prob + adjustment


def mvn_sample(
    prng_key: KeyArray, mean: ArrayLike, chol_inv_cov: ArrayLike
) -> jax.Array:
    """
    Samples from the normal distribution based on the Cholesky decomposition of the
    inverse covariance matrix.

    Parameters
    ----------
    prng_key
        The key for JAX' pseudo-random number generator.
    mean
        The mean vector.
    chol_inv_cov
        The lower triangular matrix of the Cholesky decomposition of the inverse
        variance.
    """

    mean = jnp.asarray(mean)
    standardized = jax.random.normal(prng_key, mean.shape)
    centered = triangular_solve(chol_inv_cov, standardized, lower=True)
    return centered + mean
