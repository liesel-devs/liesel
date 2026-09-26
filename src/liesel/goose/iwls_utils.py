"""
Utilities for the IWLS sampler.
"""

import jax
import jax.numpy as jnp
import jax.scipy

from .types import Array, KeyArray

triangular_solve = jax.lax.linalg.triangular_solve


def solve(chol_lhs: Array, rhs: Array) -> Array:
    """
    Solves ``(chol_lhs @ chol_lhs.T) @ x = rhs`` for x using forward and backward
    substitution. Returns x.

    Parameters
    ----------
    chol_lhs
        The lower triangular Cholesky factor, or a scalar for a multiple of I.
    rhs
        The right-hand side of the system.
    """

    if jnp.ndim(chol_lhs) == 0:
        return rhs / chol_lhs**2
    tmp = triangular_solve(chol_lhs, rhs, left_side=True, lower=True)
    return triangular_solve(chol_lhs, tmp, lower=True)


def mvn_log_prob(x: Array, mean: Array, chol_inv_cov: Array) -> Array:
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
        variance, or a scalar factor for isotropic precision.
    """

    if jnp.ndim(chol_inv_cov) == 0:
        return jnp.sum(jax.scipy.stats.norm.logpdf((x - mean) * chol_inv_cov)) + (
            x.size * jnp.log(chol_inv_cov)
        )
    standardized = (x - mean) @ chol_inv_cov
    log_prob = jnp.sum(jax.scipy.stats.norm.logpdf(standardized))
    adjustment = jnp.sum(jnp.log(jnp.diag(chol_inv_cov)))
    return log_prob + adjustment


def mvn_sample(prng_key: KeyArray, mean: Array, chol_inv_cov: Array) -> Array:
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
        variance, or a scalar factor for isotropic precision.
    """

    standardized = jax.random.normal(prng_key, mean.shape)
    if jnp.ndim(chol_inv_cov) == 0:
        return standardized / chol_inv_cov + mean
    centered = triangular_solve(chol_inv_cov, standardized, lower=True)
    return centered + mean
