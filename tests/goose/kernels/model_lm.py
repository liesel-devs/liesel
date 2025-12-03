"""
# Simple linear regression test case
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from pytest import approx

import liesel.goose as gs
from liesel.goose.engine import SamplingResults
from liesel.goose.types import Kernel

rng = np.random.default_rng(1337)

n = 30
p = 2

beta = np.ones(p)
sigma = 0.1

X = np.column_stack([np.ones(n), rng.uniform(size=[n, p - 1])])
y = rng.normal(X @ beta, sigma, size=n)

beta_ols, rss_ols, _, _ = scipy.linalg.lstsq(X, y)
sigma_ols = np.sqrt(rss_ols / (n - p))

model_state = {"y": y, "X": X, "beta": beta, "log_sigma": np.log(sigma)}


def log_prob(model_state):
    y = model_state["y"]
    mu = model_state["X"] @ model_state["beta"]
    sigma = jnp.exp(model_state["log_sigma"])

    log_probs = jax.scipy.stats.norm.logpdf(y, mu, sigma)

    return jnp.sum(log_probs)


def run_kernel_test(
    mcmc_seed: int, kernels: Sequence[Kernel], test_da_target_accept: bool = True
) -> SamplingResults:
    builder = gs.EngineBuilder(mcmc_seed, num_chains=1)

    for kernel in kernels:
        builder.add_kernel(kernel)

    model = gs.DictInterface(log_prob)
    builder.set_model(model)

    builder.set_initial_values(model_state)

    builder.set_duration(
        warmup_duration=5000,
        posterior_duration=5000,
        term_duration=4000,
    )

    engine = builder.build()
    engine.sample_all_epochs()

    results = engine.get_results()
    infos = results.get_posterior_transition_infos()
    samples = results.get_posterior_samples()

    avg_beta = np.mean(samples["beta"], axis=(0, 1))
    assert avg_beta == approx(beta_ols, rel=0.05)

    avg_log_sigma = np.mean(samples["log_sigma"])
    assert avg_log_sigma == approx(np.log(sigma_ols), rel=0.05)

    if test_da_target_accept:
        for kernel in kernels:
            if hasattr(kernel, "da_target_accept"):
                avg_acceptance_prob = np.mean(infos[kernel.identifier].acceptance_prob)

                # the next line is safe since this if block checks for the existance
                # of da_target_accept
                target_accept = kernel.da_target_accept  # type: ignore
                assert avg_acceptance_prob == approx(target_accept, abs=0.05)

    return results
