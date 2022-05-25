import jax
import numpy as np
import scipy
from pytest import approx

from liesel.goose.iwls_utils import mvn_log_prob, mvn_sample, solve

rng = np.random.default_rng(1337)


def test_solve():
    n = 5

    lhs = rng.uniform(size=(n, n))
    lhs = 0.5 * (lhs + lhs.T) + n * np.identity(n)
    rhs = rng.uniform(size=n)

    goose_result = solve(np.linalg.cholesky(lhs), rhs)
    np_result = np.linalg.solve(lhs, rhs)

    assert goose_result == approx(np_result)


def test_mvn_log_prob():
    n = 5

    cov = rng.uniform(size=(n, n))
    cov = 0.5 * (cov + cov.T) + n * np.identity(n)
    mean = rng.uniform(size=n)

    x = rng.uniform(size=n)
    goose_result = mvn_log_prob(x, mean, np.linalg.cholesky(np.linalg.inv(cov)))
    scipy_result = scipy.stats.multivariate_normal.logpdf(x, mean, cov)

    assert goose_result == approx(scipy_result)


def test_mvn_random():
    n = 5
    m = 100_000
    seed = 42

    cov = rng.uniform(size=(n, n))
    cov = 0.5 * (cov + cov.T) + n * np.identity(n)
    mean = rng.uniform(size=n)

    prng_key = jax.random.split(jax.random.PRNGKey(seed), m)
    chol_inv_cov = np.linalg.cholesky(np.linalg.inv(cov))

    sample_fn = jax.vmap(lambda prng_key: mvn_sample(prng_key, mean, chol_inv_cov))

    sample = sample_fn(prng_key)

    assert np.cov(sample, rowvar=False) == approx(cov, abs=0.05)
