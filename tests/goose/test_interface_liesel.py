import jax.numpy as jnp
import jax.random as rd
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
from liesel.goose.types import Position
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


class TestLieselInterface:
    def test_get_position(self, model) -> None:
        gm = gs.LieselInterface(model)
        pos = gm.extract_position(["mu_value"], model.state)
        assert pos["mu_value"] == 0.0

    def test_update_state(self, model) -> None:
        gm = gs.LieselInterface(model)
        pos = Position({"mu_value": 10.0})
        new_state = gm.update_state(pos, model.state)
        assert new_state["mu_value"].value == 10.0

    def test_get_log_prob(self, model) -> None:
        gm = gs.LieselInterface(model)
        lp_before = gm.log_prob(model.state)

        pos = Position({"mu_value": 10.0})
        new_state = gm.update_state(pos, model.state)
        lp_after = gm.log_prob(new_state)

        assert lp_before == pytest.approx(-719.46875)
        assert lp_after == pytest.approx(-25616.779296875)

    def test_log_prob_vars(self):
        # Create simple model with two parameters
        param1 = Var.new_param(1.0, Dist(tfd.Normal, loc=0.0, scale=1.0), name="param1")
        param2 = Var.new_param(2.0, Dist(tfd.Normal, loc=0.0, scale=1.0), name="param2")

        # Likelihood: y ~ Normal(param1 + param2, 0.1)
        y_mean = Var.new_calc(lambda p1, p2: p1 + p2, param1, param2, name="y_mean")

        # Observed data
        y_data = jnp.array([1.5, 2.5, 3.5])
        y = Var.new_obs(y_data, Dist(tfd.Normal, loc=y_mean, scale=0.1), name="y")

        # Build model
        model = GraphBuilder().add(param1, param2, y).build_model()
        interface = gs.LieselInterface(model)

        model_state = model.state

        # Test individual parameter log-probs
        param1_log_prob = interface.log_prob_vars(model_state, ["param1"])
        param2_log_prob = interface.log_prob_vars(model_state, ["param2"])

        # Test multiple parameters
        both_params_log_prob = interface.log_prob_vars(
            model_state, ["param1", "param2"]
        )
        assert jnp.allclose(both_params_log_prob, param1_log_prob + param2_log_prob)

        # Test likelihood
        likelihood_log_prob = interface.log_prob_vars(model_state, ["y"])

        # Verify that sum of parts equals total
        total_log_prob = interface.log_prob(model_state)
        expected_total = param1_log_prob + param2_log_prob + likelihood_log_prob

        assert jnp.allclose(total_log_prob, expected_total)


@pytest.mark.mcmc
def test_sample_model(model) -> None:
    mcmc_seed = 1337
    builder = gs.EngineBuilder(mcmc_seed, num_chains=1)

    builder.add_kernel(gs.NUTSKernel(["mu_value"]))

    goose_model = gs.LieselInterface(model)
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


@pytest.mark.mcmc
def test_sample_transformed_model(model: Model):
    _, vars = model.copy_nodes_and_vars()
    vsigma = vars["sigma"]
    vsigma.dist_node = Dist(tfd.InverseGamma, 0.1, 0.1)
    gb = GraphBuilder().add(*vars.values())
    vsigma.transform()
    model = gb.build_model()

    mcmc_seed = 1337
    builder = gs.EngineBuilder(mcmc_seed, num_chains=1)

    builder.add_kernel(gs.NUTSKernel(["mu_value", "sigma_transformed_value"]))
    builder.positions_included = ["sigma"]

    goose_model = gs.LieselInterface(model)
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
