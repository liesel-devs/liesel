"""
This test module still uses the old API. It will be removed in the future, when
lsl.GooseModel is removed.
"""

import jax
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
from liesel.model.goose import finite_discrete_gibbs_kernel


class TestFiniteDiscreteGibbsKernel:
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

        model = lsl.Model([categorical_var])
        kernel = finite_discrete_gibbs_kernel("categorical_var", model)

        draw = kernel._transition_fn(jax.random.PRNGKey(0), model.state)
        assert draw["categorical_var"] == pytest.approx(1)

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

        model = lsl.Model([categorical_var])
        kernel = finite_discrete_gibbs_kernel("categorical_var", model)

        draw = jax.jit(kernel._transition_fn)(jax.random.PRNGKey(1), model.state)
        assert draw["categorical_var"] == pytest.approx(2)

    @pytest.mark.mcmc
    def test_sample_categorical(self):
        values = [0.0, 1.0, 2.0]
        prior_probs = [0.1, 0.2, 0.7]
        value_grid = lsl.Var(values, name="value_grid")

        prior = lsl.Dist(tfd.FiniteDiscrete, outcomes=value_grid, probs=prior_probs)
        categorical_var = lsl.Var(
            value=values[0],
            distribution=prior,
            name="categorical_var",
        )

        model = lsl.Model([categorical_var])
        kernel = finite_discrete_gibbs_kernel("categorical_var", model)

        eb = gs.EngineBuilder(1, num_chains=1)
        eb.add_kernel(kernel)
        eb.set_model(gs.LieselInterface(model))
        eb.set_initial_values(model.state)
        eb.set_duration(warmup_duration=500, posterior_duration=2000)

        engine = eb.build()
        engine.sample_all_epochs()

        results = engine.get_results()
        samples = results.get_posterior_samples()

        _, counts = np.unique(samples["categorical_var"], return_counts=True)
        relative_freq = counts / np.sum(counts)

        assert np.allclose(relative_freq, prior_probs, atol=0.1)

    @pytest.mark.mcmc
    def test_sample_bernoulli(self):
        prior_prob = 0.7
        prior = lsl.Dist(tfd.Bernoulli, probs=lsl.Value(prior_prob))
        dummy_var = lsl.Var(
            value=1,
            distribution=prior,
            name="dummy_var",
        )

        model = lsl.Model([dummy_var])
        kernel = finite_discrete_gibbs_kernel("dummy_var", model, outcomes=[0, 1])

        eb = gs.EngineBuilder(1, num_chains=1)
        eb.add_kernel(kernel)
        eb.set_model(gs.LieselInterface(model))
        eb.set_initial_values(model.state)
        eb.set_duration(warmup_duration=500, posterior_duration=2000)

        engine = eb.build()
        engine.sample_all_epochs()

        results = engine.get_results()
        samples = results.get_posterior_samples()

        _, counts = np.unique(samples["dummy_var"], return_counts=True)
        relative_freq = counts / np.sum(counts)

        assert np.allclose(relative_freq, [1 - prior_prob, prior_prob], atol=0.1)

    @pytest.mark.mcmc
    def test_bernoulli_no_outcomes(self):
        prior_prob = 0.7
        prior = lsl.Dist(tfd.Bernoulli, probs=lsl.Value(prior_prob))
        dummy_var = lsl.Var(
            value=1,
            distribution=prior,
            name="dummy_var",
        )

        model = lsl.Model([dummy_var])
        kernel = finite_discrete_gibbs_kernel("dummy_var", model)

        eb = gs.EngineBuilder(1, num_chains=1)
        eb.add_kernel(kernel)
        eb.set_model(gs.LieselInterface(model))
        eb.set_initial_values(model.state)
        eb.set_duration(warmup_duration=500, posterior_duration=2000)

        engine = eb.build()
        engine.sample_all_epochs()

        results = engine.get_results()
        samples = results.get_posterior_samples()

        _, counts = np.unique(samples["dummy_var"], return_counts=True)
        relative_freq = counts / np.sum(counts)

        assert np.allclose(relative_freq, [1 - prior_prob, prior_prob], atol=0.1)
