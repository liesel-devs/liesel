"""
Tests for the ESSKernel.
"""

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from model_lm import run_kernel_test

import liesel.goose as gs
import liesel.model as lsl


def test_ess_kernel_init():
    """Test ESSKernel initialization."""
    # Should work with single position key
    kernel = gs.ESSKernel(
        position_keys=["f"],
    )
    assert kernel.position_keys == ("f",)

    # Should fail with multiple position keys
    with pytest.raises(ValueError, match="exactly one position key"):
        gs.ESSKernel(
            position_keys=["f1", "f2"],
        )


def test_ess_kernel_gaussian_validation():
    """Test that ESSKernel validates Gaussian priors."""
    # Create model with Gaussian prior (should work)
    n = 10
    mu = jnp.zeros(n)
    K = jnp.eye(n)

    f = lsl.Var.new_param(
        mu,
        lsl.Dist(tfd.MultivariateNormalFullCovariance, loc=mu, covariance_matrix=K),
        name="f",
    )

    # Likelihood: y ~ Normal(f, 0.1)
    sigma = lsl.Var.new_param(0.1, name="sigma")

    # Observed data
    y_data = jnp.ones(n) * 0.5
    y = lsl.Var.new_obs(y_data, lsl.Dist(tfd.Normal, loc=f, scale=sigma), name="y")

    model = lsl.GraphBuilder().add(f, sigma, y).build_model()
    interface = gs.LieselInterface(model)

    kernel = gs.ESSKernel(
        position_keys=["f"],
    )

    # Set model and initialize
    kernel.set_model(interface)
    kernel.init_state(jax.random.PRNGKey(0), model.state)


def test_ess_kernel_non_gaussian_error():
    """Test that ESSKernel rejects non-Gaussian priors."""
    # Create model with non-Gaussian prior (should fail)
    f = lsl.Var.new_param(
        1.0,
        lsl.Dist(tfd.Gamma, concentration=1.0, rate=1.0),  # Non-Gaussian!
        name="f",
    )

    y_data = jnp.array([1.5])
    y = lsl.Var.new_obs(y_data, lsl.Dist(tfd.Normal, loc=f, scale=0.1), name="y")

    model = lsl.GraphBuilder().add(f, y).build_model()
    interface = gs.LieselInterface(model)

    kernel = gs.ESSKernel(
        position_keys=["f"],
    )
    kernel.set_model(interface)

    with pytest.raises(TypeError):
        # Should fail when trying to initialize with non-Gaussian prior
        kernel.init_state(jax.random.PRNGKey(0), model.state)


# Integration test for ESSKernel with full MCMC sampling.


@pytest.mark.mcmc
def test_ess_kernel_mcmc_sampling():
    """Test ESSKernel with full MCMC sampling from the prior."""
    # Prior: f ~ MVN(mu=(0, 1), cov=diag(1, 0.5))
    mu = jnp.array([0.0, 1.0])
    cov = jnp.diag(jnp.array([1.0, 0.5]))

    # Create model that samples from prior only (no likelihood)
    f = lsl.Var.new_param(
        mu,
        lsl.Dist(tfd.MultivariateNormalFullCovariance, loc=mu, covariance_matrix=cov),
        name="f",
    )

    # Build model with just the prior
    model = lsl.GraphBuilder().add(f).build_model()
    interface = gs.LieselInterface(model)

    # Create ESSKernel
    kernel = gs.ESSKernel(
        position_keys=["f"],
    )

    # Build MCMC setup and run
    builder = gs.EngineBuilder(seed=42, num_chains=4)
    builder.set_model(interface)
    builder.set_initial_values(model.state)

    builder.set_duration(
        warmup_duration=200,
        posterior_duration=1000,
    )

    builder.add_kernel(kernel)
    engine = builder.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    # Flatten chains dimension for statistics
    f_samples = samples["f"]  # (num_chains, posterior_duration, 2)
    f_flat = f_samples.reshape(-1, 2)

    # Check that samples have approximately correct mean and covariance
    sample_mean = jnp.mean(f_flat, axis=0)
    sample_cov = jnp.cov(f_flat.T)

    # Allow for sampling variation
    assert jnp.allclose(sample_mean, mu, atol=0.01)
    assert jnp.allclose(sample_cov, cov, atol=0.1)


@pytest.mark.mcmc
def test_ess_kernel_with_likelihood():
    """Test ESSKernel with both prior and likelihood."""
    # Prior: f ~ MVN(mu = (0, 1), cov = diag(1, 0.5))
    mu = jnp.array([0.0, 1.0])
    cov = jnp.diag(jnp.array([1.0, 0.5]))

    f = lsl.Var.new_param(
        mu,
        lsl.Dist(tfd.MultivariateNormalFullCovariance, loc=mu, covariance_matrix=cov),
        name="f",
    )

    # Likelihood: y ~ Normal(f[0] + f[1], 0.1)
    y_mean = lsl.Var.new_calc(lambda f_vec: f_vec[0] + f_vec[1], f, name="y_mean")
    y_data = jnp.array([1.5])
    y = lsl.Var.new_obs(y_data, lsl.Dist(tfd.Normal, loc=y_mean, scale=0.1), name="y")

    # Build model
    model = lsl.GraphBuilder().add(f, y).build_model()
    interface = gs.LieselInterface(model)

    # Create ESSKernel
    kernel = gs.ESSKernel(
        position_keys=["f"],
    )

    # Build and run sampler
    builder = gs.EngineBuilder(seed=42, num_chains=1)
    builder.set_model(interface)
    builder.set_initial_values(model.state)
    builder.set_duration(
        warmup_duration=200,
        posterior_duration=300,
    )
    builder.add_kernel(kernel)
    engine = builder.build()
    engine.sample_all_epochs()

    # Get samples
    results = engine.get_results()
    samples = results.get_posterior_samples()
    f_samples = samples["f"]

    # Basic sanity checks
    assert f_samples.shape == (1, 300, 2)
    assert jnp.isfinite(f_samples).all()


@pytest.mark.skip("ESS only supports Liesel Models")
def test_ess_on_model(mcmc_seed):
    k1 = gs.ESSKernel(["beta"])
    k2 = gs.HMCKernel(["log_sigma"])
    run_kernel_test(mcmc_seed, [k1, k2])
