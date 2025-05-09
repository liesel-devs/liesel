"""
some tests for the engine builder
"""

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
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


class TestLieselMCMC:
    def test_engine(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(gs.NUTSKernel),
            name="mu",
        )

        model = lsl.Model([mu])

        mcmc = gs.LieselMCMC(model)
        eb = mcmc.get_engine_builder(seed=1, num_chains=4)
        eb.set_duration(warmup_duration=200, posterior_duration=100)
        engine = eb.build()

        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        assert "mu" in samples

    def test_multiple_specs(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference={
                "a": gs.MCMCSpec(gs.NUTSKernel),
                "b": gs.MCMCSpec(gs.IWLSKernel),
            },
            name="mu",
        )

        model = lsl.Model([mu])

        with pytest.raises(ValueError):
            gs.LieselMCMC(model).get_kernel_list()

        kernels = gs.LieselMCMC(model, which="a").get_kernel_list()
        assert isinstance(kernels[0], gs.NUTSKernel)

        kernels = gs.LieselMCMC(model, which="b").get_kernel_list()
        assert isinstance(kernels[0], gs.IWLSKernel)

    def test_multiple_and_single_specs(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference={
                "a": gs.MCMCSpec(gs.NUTSKernel),
                "b": gs.MCMCSpec(gs.IWLSKernel),
            },
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=2.0, scale=1.0),
            inference=gs.MCMCSpec(gs.IWLSKernel),
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        with pytest.raises(ValueError):
            gs.LieselMCMC(model).get_kernel_list()

        kernels = gs.LieselMCMC(model, which="a").get_kernel_list()
        assert isinstance(kernels[0], gs.IWLSKernel)
        assert isinstance(kernels[1], gs.NUTSKernel)

        kernels = gs.LieselMCMC(model, which="b").get_kernel_list()
        assert isinstance(kernels[0], gs.IWLSKernel)
        assert isinstance(kernels[1], gs.IWLSKernel)

    def test_kernel_group(self):
        spec = gs.MCMCSpec(
            gs.NUTSKernel,
            kernel_group="a",
            kernel_kwargs={"mm_diag": True, "da_target_accept": 0.8},
        )

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=spec,
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=spec,
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        kernels = mcmc.get_kernel_list()

        assert len(kernels) == 1
        assert kernels[0].position_keys == ("sigma", "mu")

    def test_incoherent_kernel_group(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(gs.NUTSKernel, kernel_group="a"),
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=gs.MCMCSpec(gs.IWLSKernel, kernel_group="a"),
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        with pytest.raises(ValueError):
            mcmc.get_kernel_list()

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(
                gs.NUTSKernel, {"da_target_accept": 0.6}, kernel_group="a"
            ),
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=gs.MCMCSpec(
                gs.NUTSKernel, {"da_target_accept": 0.7}, kernel_group="a"
            ),
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        with pytest.raises(ValueError):
            mcmc.get_kernel_list()

    def test_jitter_functions(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(
                gs.NUTSKernel,
                kernel_group="a",
                jitter_dist=tfd.Uniform(low=-1.0, high=1.0),
            ),
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=gs.MCMCSpec(
                gs.NUTSKernel,
                kernel_group="a",
                jitter_dist=tfd.Uniform(low=0.0, high=1.0),
            ),
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        jitter_fns = mcmc.get_jitter_functions()

        assert len(jitter_fns) == 2

        eb = mcmc.get_engine_builder(1, 4)
        assert len(eb.jitter_fns.expect("")) == 2

    def test_jitter_draw_shape(self):
        # 1d array
        mu = lsl.Var.new_param(
            jnp.zeros(3),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(
                gs.NUTSKernel,
                kernel_group="a",
                jitter_dist=tfd.Uniform(low=-1.0, high=1.0),
            ),
            name="mu",
        )

        model = lsl.Model([mu])

        mcmc = gs.LieselMCMC(model)

        jitter_funs = mcmc.get_jitter_functions()
        jitter_draw = jitter_funs["mu"](jax.random.key(0), mu.value)
        assert not jnp.all(jitter_draw == jitter_draw[0])  # not all equal
        # no two are equal
        assert len(jnp.unique(jitter_draw)) == len(jitter_draw.flatten())

        # 2d array
        mu = lsl.Var.new_param(
            jnp.zeros((3, 3)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(
                gs.NUTSKernel,
                kernel_group="a",
                jitter_dist=tfd.Uniform(low=-1.0, high=1.0),
            ),
            name="mu",
        )

        model = lsl.Model([mu])

        mcmc = gs.LieselMCMC(model)

        jitter_funs = mcmc.get_jitter_functions()
        jitter_draw = jitter_funs["mu"](jax.random.key(0), mu.value)
        assert not jnp.all(jitter_draw == jitter_draw[0])  # not all equal
        # no two are equal
        assert len(jnp.unique(jitter_draw)) == len(jitter_draw.flatten())

    def test_transform_var_with_inference_new(self):
        """
        It is allowed to pass a new inferece object during transformation.
        In this case, the inference object of the original variable is removed.
        """
        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=2.0, scale=1.0),
            inference=gs.MCMCSpec(gs.IWLSKernel),
            name="sigma",
        )
        inference = sigma.inference

        log_sigma = sigma.transform(tfb.Exp(), inference=gs.MCMCSpec(gs.NUTSKernel))
        assert log_sigma.inference is not inference
        assert sigma.inference is None
        assert log_sigma.inference.kernel is gs.NUTSKernel

    def test_transform_var_with_inference_none(self):
        """
        Default behavior when trying to transform a variable *with* inference
        information: Error. You need to declare explicitly, what you want to do.
        In this case, ``"drop"`` means the inference information is deleted.
        """
        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=2.0, scale=1.0),
            inference=gs.MCMCSpec(gs.IWLSKernel),
            name="sigma",
        )
        inference = sigma.inference

        with pytest.raises(ValueError):
            sigma.transform(tfb.Exp())

        log_sigma = sigma.transform(tfb.Exp(), inference="drop")
        assert log_sigma.inference is not inference
        assert sigma.inference is None
        assert log_sigma.inference is None

    def test_transform_var_without_inference(self):
        """
        Default when the original variable has no inference information:
        Everything works smoothly.
        """
        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=2.0, scale=1.0),
            name="sigma",
        )

        log_sigma = sigma.transform(tfb.Exp())

        assert log_sigma.inference is None
        assert sigma.inference is None
        assert log_sigma.inference is None
