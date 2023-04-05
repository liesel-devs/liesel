import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from model_lm import beta_ols, model_state, run_kernel_test

import liesel.goose as gs
import liesel.model as lsl
from liesel.goose.gibbs import (
    GibbsKernelState,
    GibbsTransitionInfo,
    GibbsTuningInfo,
    create_categorical_gibbs_kernel,
)
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


class TestGeneralCategoricalGibbsKernel:
    def test_transition(self):
        values = [0, 1, 2]
        prior_probs = [0.1, 0.2, 0.7]
        value_grid = lsl.Var(values, name="value_grid")

        prior = lsl.Dist(tfd.FiniteDiscrete, outcomes=value_grid, probs=prior_probs)
        categorical_var = lsl.Var(
            value=values[0],
            distribution=prior,
            name="categorical_var",
        )

        model = lsl.GraphBuilder().add(categorical_var).build_model()
        kernel = create_categorical_gibbs_kernel("categorical_var", values, model)

        draw = kernel._transition_fn(jax.random.PRNGKey(0), model.state)
        assert draw["categorical_var"] == pytest.approx(0)

        draw = kernel._transition_fn(jax.random.PRNGKey(1), model.state)
        assert draw["categorical_var"] == pytest.approx(2)

        draw = kernel._transition_fn(jax.random.PRNGKey(2), model.state)
        assert draw["categorical_var"] == pytest.approx(1)

    def test_transition_jit(self):
        values = [0, 1, 2]
        prior_probs = [0.1, 0.2, 0.7]
        value_grid = lsl.Var(values, name="value_grid")

        prior = lsl.Dist(tfd.FiniteDiscrete, outcomes=value_grid, probs=prior_probs)
        categorical_var = lsl.Var(
            value=values[0],
            distribution=prior,
            name="categorical_var",
        )

        model = lsl.GraphBuilder().add(categorical_var).build_model()
        kernel = create_categorical_gibbs_kernel("categorical_var", values, model)

        draw = jax.jit(kernel._transition_fn)(jax.random.PRNGKey(1), model.state)
        assert draw["categorical_var"] == pytest.approx(2)

    @pytest.mark.mcmc
    def test_sample(self):
        values = [0, 1, 2]
        prior_probs = [0.1, 0.2, 0.7]
        value_grid = lsl.Var(values, name="value_grid")

        prior = lsl.Dist(tfd.FiniteDiscrete, outcomes=value_grid, probs=prior_probs)
        categorical_var = lsl.Var(
            value=values[0],
            distribution=prior,
            name="categorical_var",
        )

        model = lsl.GraphBuilder().add(categorical_var).build_model()
        kernel = create_categorical_gibbs_kernel("categorical_var", values, model)

        eb = gs.EngineBuilder(1, num_chains=1)
        eb.add_kernel(kernel)
        eb.set_model(lsl.GooseModel(model))
        eb.set_initial_values(model.state)
        eb.set_duration(warmup_duration=500, posterior_duration=2000)

        engine = eb.build()
        engine.sample_all_epochs()

        results = engine.get_results()
        samples = results.get_posterior_samples()

        values, counts = np.unique(samples["categorical_var"], return_counts=True)
        relative_freq = counts / np.sum(counts)

        assert np.allclose(relative_freq, prior_probs, atol=0.1)
