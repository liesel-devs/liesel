import jax
import jax.numpy as jnp
import jax.random as random
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
from liesel.goose.mcmc_spec import MCMCSpec

type Array = jax.Array


class FixedDistribution(tfd.Distribution):
    def __init__(self, fixed_value: Array):
        super().__init__(
            dtype=jnp.float32,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False,
        )
        self.fixed_value = fixed_value

    def _sample_n(self, n, seed=None):
        return jnp.tile(self.fixed_value, (n, 1))

    def _batch_shape_tensor(self):
        return jnp.array([])

    def _batch_shape(self):
        return jnp.array([])

    def _event_shape_tensor(self):
        return self.fixed_value.shape

    def _event_shape(self):
        return self.fixed_value.shape


class DummyKernel:
    def __init__(self, position_keys, **kwargs):
        self.position_keys = position_keys
        self.kwargs = kwargs


# Create a dummy kernel factory for testing
def dummy_kernel_factory(position_keys, **kwargs):
    return DummyKernel(position_keys, **kwargs)


class TestMCMCSpec:
    def setup_method(self):
        # Setup a basic MCMCSpec for testing
        self.kernel_factory = dummy_kernel_factory
        self.key = random.PRNGKey(42)
        self.value = jnp.ones((3,))

    def test_no_jitter(self):
        """Test that no jitter is applied when jitter_dist is None."""
        spec = MCMCSpec(kernel=self.kernel_factory)
        result = spec.apply_jitter(self.key, self.value)
        assert jnp.array_equal(result, self.value)

    def test_jitter_type_none(self):
        """Test that no jitter is applied when jitter_type is NONE."""
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.Normal(loc=0.0, scale=1.0),
            jitter_method="none",
        )
        result = spec.apply_jitter(self.key, self.value)
        assert jnp.array_equal(result, self.value)

    def test_additive_jitter(self):
        """Test additive jitter application."""
        # Using a fixed distribution for deterministic testing
        fixed_jitter = jnp.array([0.1, 0.2, 0.3])
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=FixedDistribution(jnp.array([0.1, 0.2, 0.3])),
            jitter_method="additive",
        )

        result = spec.apply_jitter(self.key, self.value)
        expected = self.value + fixed_jitter
        assert jnp.allclose(result, expected)

    def test_multiplicative_jitter(self):
        """Test multiplicative jitter application."""
        # Using a fixed distribution for deterministic testing
        fixed_jitter = jnp.array([2.0, 3.0, 4.0])

        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=FixedDistribution(fixed_jitter),
            jitter_method="multiplicative",
        )

        result = spec.apply_jitter(self.key, self.value)
        expected = self.value * fixed_jitter
        assert jnp.allclose(result, expected)

    def test_replacement_jitter(self):
        """Test replacement jitter application."""
        # Using a fixed distribution for deterministic testing
        fixed_jitter = jnp.array([5.0, 6.0, 5.0])
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=FixedDistribution(fixed_jitter),
            jitter_method="replacement",
        )

        result = spec.apply_jitter(self.key, self.value)
        expected = fixed_jitter
        assert jnp.allclose(result, expected)

    def test_with_normal_distribution(self):
        """Test with a standard TFP distribution (Normal)."""
        # Using a seeded jitter for reproducible testing
        seed = 42
        value = jnp.ones(3)

        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.Normal(loc=0.0, scale=1.0),
            jitter_method="additive",
        )

        # Generate jitter separately to compare
        key = random.PRNGKey(seed)
        jitter = tfd.Normal(loc=0.0, scale=1.0).sample(sample_shape=(3,), seed=key)
        expected = value + jitter

        # Apply jitter via the MCMCSpec
        result = spec.apply_jitter(key, value)

        assert jnp.allclose(result, expected)

    def test_shape_compatibility(self):
        """Test jitter with different shape configurations."""
        # Test with scalar value and distribution
        scalar_value = jnp.array(1.0)
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.Normal(loc=0.0, scale=1.0),
            jitter_method="replacement",
        )
        result = spec.apply_jitter(self.key, scalar_value)
        assert result.shape == scalar_value.shape

        # Test with vector value and scalar distribution
        vector_value = jnp.ones(3)
        result = spec.apply_jitter(self.key, vector_value)
        assert result.shape == vector_value.shape
        assert result[0] != result[1]  # Check that we have multiple draws

        # Test with multivariate distribution
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.MultivariateNormalDiag(
                loc=jnp.zeros(3), scale_diag=jnp.ones(3)
            ),
        )
        result = spec.apply_jitter(self.key, vector_value)
        assert result.shape == vector_value.shape

    def test_incompatible_shapes(self):
        """Test error handling for incompatible shapes."""
        # Distribution shape (2,) for value shape (3,)
        value = jnp.ones(3)
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.MultivariateNormalDiag(
                loc=jnp.zeros(2), scale_diag=jnp.ones(2)
            ),
            jitter_method="additive",
        )

        with pytest.raises(ValueError, match="do not match variable shape"):
            spec.apply_jitter(self.key, value)

    def test_batch_shape(self):
        """Test jitter application with batched distributions."""
        value = jnp.ones((2, 3))
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            # Create a batch of 2 normal distributions
            jitter_dist=tfd.Independent(
                tfd.Normal(loc=jnp.zeros((2, 3)), scale=jnp.ones((2, 3))),
                reinterpreted_batch_ndims=1,
            ),
            jitter_method="additive",
        )

        result = spec.apply_jitter(self.key, value)
        assert result.shape == value.shape


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
        spec = gs.MCMCSpec(gs.NUTSKernel, kernel_group="a")

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
