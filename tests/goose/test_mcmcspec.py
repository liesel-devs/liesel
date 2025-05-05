import jax.numpy as jnp
import jax.random as random
import jax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.goose.mcmc_spec import JitterType, MCMCSpec
Array = jax.Array

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
            jitter_type=JitterType.NONE,
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
            jitter_type=JitterType.ADDITIVE,
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
            jitter_type=JitterType.MULTIPLICATIVE,
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
            jitter_type=JitterType.REPLACEMENT,
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
            jitter_type=JitterType.ADDITIVE,
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
            jitter_type=JitterType.REPLACEMENT,
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
            jitter_type=JitterType.ADDITIVE,
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
            jitter_type=JitterType.ADDITIVE,
        )

        result = spec.apply_jitter(self.key, value)
        assert result.shape == value.shape
