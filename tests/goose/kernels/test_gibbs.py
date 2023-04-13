import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
from model_lm import beta_ols, model_state, run_kernel_test

import liesel.goose as gs
from liesel.goose.gibbs import GibbsKernelState, GibbsTransitionInfo, GibbsTuningInfo
from liesel.goose.types import Kernel

XX_inv = np.linalg.inv(model_state["X"].T @ model_state["X"])


def transition_fn(prng_key, model_state):
    cov = (jnp.exp(model_state["log_sigma"]) ** 2) * XX_inv
    position = {"beta": jax.random.multivariate_normal(prng_key, beta_ols, cov)}
    return position


def type_check() -> None:
    kernel = gs.GibbsKernel(["beta"], transition_fn)
    _: Kernel[GibbsKernelState, GibbsTransitionInfo, GibbsTuningInfo] = kernel


@pytest.mark.mcmc
def test_gibbs(mcmc_seed):
    kernel0 = gs.GibbsKernel(["beta"], transition_fn)
    kernel1 = gs.NUTSKernel(["log_sigma"])

    run_kernel_test(mcmc_seed, [kernel0, kernel1])
