"""
some tests for the engine builder
"""

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
from liesel.goose.builder import EngineBuilder
from liesel.goose.interface import DictInterface


def test_seed_input():
    int_seed = 0
    key_seed = jax.random.PRNGKey(int_seed)
    builder = EngineBuilder(seed=int_seed, num_chains=2)
    builder2 = EngineBuilder(seed=key_seed, num_chains=2)

    assert jnp.all(builder._prng_key == builder2._prng_key)
    assert jnp.all(builder._engine_key == builder2._engine_key)
    assert jnp.all(builder._jitter_key == builder2._jitter_key)


def test_jitter_fns():
    con = DictInterface(lambda ms: -0.5 * ms["x"] ** 2 - 0.5 * ms["y"])
    ms = {"x": jnp.array(1), "y": jnp.array(-1)}

    num_chains = 2

    builder = EngineBuilder(seed=1, num_chains=num_chains)
    builder.set_model(con)
    builder.set_initial_values(ms, multiple_chains=False)
    builder.set_jitter_fns(
        {
            "x": (
                lambda key, cv: cv
                + tfd.Uniform(-1.0, 1.0).sample(sample_shape=cv.shape, seed=key)
            ),
            "y": (
                lambda key, cv: cv
                + tfd.Uniform(-1.0, 1.0).sample(sample_shape=cv.shape, seed=key)
            ),
        }
    )
    builder.add_kernel(gs.IWLSKernel(["x", "y"]))
    builder.set_duration(warmup_duration=200, posterior_duration=10, term_duration=10)
    engine = builder.build()

    assert not jnp.allclose(ms["x"], engine._model_states["x"][0])
    assert not jnp.allclose(ms["y"], engine._model_states["y"][0])
    assert not jnp.allclose(ms["x"], engine._model_states["x"][1])
    assert not jnp.allclose(ms["y"], engine._model_states["y"][1])

    assert not jnp.allclose(engine._model_states["x"][0], engine._model_states["x"][1])


@pytest.mark.parametrize(("burnin", "posterior"), ((10, 10), (1, 1)))
def test_simple_duration(burnin, posterior):
    mu = lsl.Var.new_param(0.0, name="mu", inference=gs.MCMCSpec(gs.IWLSKernel))
    y = lsl.Var.new_obs(
        jnp.array([10.0, 5.0]), lsl.Dist(tfd.Normal, loc=mu, scale=1.0), name="y"
    )
    model = lsl.Model([y])

    eb = gs.LieselMCMC(model).get_engine_builder(0, 4)
    eb.set_duration_simple(burnin, posterior)
    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()
    assert "mu" in samples
    assert samples["mu"].shape == (4, posterior)


class TestAddEpochs:
    def test_add_adaptation(self):
        mu = lsl.Var.new_param(0.0, name="mu", inference=gs.MCMCSpec(gs.IWLSKernel))
        y = lsl.Var.new_obs(
            jnp.array([10.0, 5.0]), lsl.Dist(tfd.Normal, loc=mu, scale=1.0), name="y"
        )
        model = lsl.Model([y])

        eb = gs.LieselMCMC(model).get_engine_builder(0, 4)

        eb.add_adaptation(50)
        eb.add_posterior(10)

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()

        samples = results.get_adaptation_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 50)

        samples = results.get_warmup_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 50)

        samples = results.get_posterior_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 10)

    def test_add_adaptation_with_integers(self):
        mu = lsl.Var.new_param(0.0, name="mu", inference=gs.MCMCSpec(gs.IWLSKernel))
        y = lsl.Var.new_obs(
            jnp.array([10.0, 5.0]), lsl.Dist(tfd.Normal, loc=mu, scale=1.0), name="y"
        )
        model = lsl.Model([y])

        eb = gs.LieselMCMC(model).get_engine_builder(0, 4)

        with pytest.raises(ValueError):
            eb.add_adaptation(50, init=20, term=50)

        eb.add_adaptation(500, init=20, term=50)
        eb.add_posterior(10)

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()

        samples = results.get_adaptation_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 500)

        samples = results.get_warmup_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 500)

        samples = results.get_posterior_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 10)

    def test_add_burnin(self):
        mu = lsl.Var.new_param(0.0, name="mu", inference=gs.MCMCSpec(gs.IWLSKernel))
        y = lsl.Var.new_obs(
            jnp.array([10.0, 5.0]), lsl.Dist(tfd.Normal, loc=mu, scale=1.0), name="y"
        )
        model = lsl.Model([y])

        eb = gs.LieselMCMC(model).get_engine_builder(0, 4)

        eb.add_burnin(50)
        eb.add_posterior(10)

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()

        with pytest.raises(RuntimeError):
            results.get_adaptation_samples()

        samples = results.get_warmup_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 50)

        samples = results.get_posterior_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 10)

    def test_add_posterior(self):
        mu = lsl.Var.new_param(0.0, name="mu", inference=gs.MCMCSpec(gs.IWLSKernel))
        y = lsl.Var.new_obs(
            jnp.array([10.0, 5.0]), lsl.Dist(tfd.Normal, loc=mu, scale=1.0), name="y"
        )
        model = lsl.Model([y])

        eb = gs.LieselMCMC(model).get_engine_builder(0, 4)

        eb.add_posterior(100, thinning=2)

        engine = eb.build()
        engine.sample_all_epochs()
        results = engine.get_results()

        samples = results.get_posterior_samples()
        assert "mu" in samples
        assert samples["mu"].shape == (4, 50)
