"""
This test module still uses the old API. It will be removed in the future, when
lsl.GooseModel is removed.
"""

import jax
import jax.random as rd
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
from liesel.goose.types import Position
from liesel.model.goose import GooseModel, finite_discrete_gibbs_kernel
from liesel.model.model import GraphBuilder, Model
from liesel.model.nodes import Dist, Var


@pytest.fixture
def model():
    key = rd.PRNGKey(1337)
    mu = Var(0.0, name="mu")
    sigma = Var(1.0, name="sigma")

    x = Var(rd.normal(key, shape=(500,)), name="x")
    y = Var(x, Dist(tfd.Normal, loc=mu, scale=sigma), "y")
    yield Model([y])


class TestGooseModel:
    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_get_position(self, model) -> None:
        gm = GooseModel(model)
        pos = gm.extract_position(["mu_value"], model.state)
        assert pos["mu_value"] == 0.0

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_update_state(self, model) -> None:
        gm = GooseModel(model)
        pos = Position({"mu_value": 10.0})
        new_state = gm.update_state(pos, model.state)
        assert new_state["mu_value"].value == 10.0

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_get_log_prob(self, model) -> None:
        gm = GooseModel(model)
        lp_before = gm.log_prob(model.state)

        pos = Position({"mu_value": 10.0})
        new_state = gm.update_state(pos, model.state)
        lp_after = gm.log_prob(new_state)

        assert lp_before == pytest.approx(-719.46875)
        assert lp_after == pytest.approx(-25616.779296875)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.mcmc
def test_sample_model(model) -> None:
    mcmc_seed = 1337
    builder = gs.EngineBuilder(mcmc_seed, num_chains=1)

    builder.add_kernel(gs.NUTSKernel(["mu_value"]))

    goose_model = GooseModel(model)
    builder.set_model(goose_model)

    builder.set_initial_values(model.state)

    builder.set_duration(
        warmup_duration=1000,
        posterior_duration=1000,
    )

    engine = builder.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()
    avg_mu = np.mean(samples["mu_value"], axis=(0, 1))

    assert avg_mu == pytest.approx(0.0, abs=0.05)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.mcmc
def test_sample_transformed_model(model: Model):
    _, vars = model.copy_nodes_and_vars()
    vsigma = vars["sigma"]
    vsigma.dist_node = Dist(tfd.InverseGamma, 0.1, 0.1)
    gb = GraphBuilder().add(*vars.values())
    gb.transform(vsigma)
    model = gb.build_model()

    mcmc_seed = 1337
    builder = gs.EngineBuilder(mcmc_seed, num_chains=1)

    builder.add_kernel(gs.NUTSKernel(["mu_value", "sigma_transformed_value"]))
    builder.positions_included = ["sigma"]

    goose_model = GooseModel(model)
    builder.set_model(goose_model)

    builder.set_initial_values(model.state)

    builder.set_duration(
        warmup_duration=1000,
        posterior_duration=1000,
    )

    engine = builder.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()
    avg_mu = np.mean(samples["mu_value"], axis=(0, 1))
    avg_sigma = np.mean(samples["sigma"], axis=(0, 1))

    assert avg_mu == pytest.approx(0.0, abs=0.05)
    assert avg_sigma == pytest.approx(1.0, abs=0.05)


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
