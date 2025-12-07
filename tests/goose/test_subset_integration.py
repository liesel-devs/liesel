"""Integration tests for subset log probability functionality."""

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl


def test_subset_with_nonexistent_calc_node_raises():
    # create simple model without subset calc
    param1 = lsl.Var.new_param(
        jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="param1"
    )

    # build model
    gb = lsl.GraphBuilder().add(param1)
    model = gb.build_model()

    # create HMC kernel with non-existent subset
    kernel = gs.HMCKernel(
        position_keys=["param1"],
        subset_logp_name="_nonexistent_subset",
    )

    # create interface
    interface = gs.LieselInterface(model)
    kernel.set_model(interface)

    # test that log_prob_fn raises error when called
    model_state = model.state
    log_prob_fn = kernel.log_prob_fn(model_state)
    current_position = kernel.position(model_state)

    with pytest.raises(
        KeyError,
    ):
        log_prob_fn(current_position)


@pytest.mark.mcmc
def test_hierarchical_model_subset_vs_full_mcmc():
    """
    Test subset vs full model sampling in hierarchical model.

    This test is demonstrating cutting feedback.
    """

    # create hierarchical model: param1 ~ N(0,1), param2 ~ N(param1,1), x ~ N(param2,1)
    n = 100
    param1 = lsl.Var.new_param(
        jnp.array(0.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="param1"
    )

    param2 = lsl.Var.new_param(
        jnp.array(0.0),
        lsl.Dist(tfd.Normal, loc=param1, scale=1.0),
        name="param2",
    )

    # generate synthetic data
    x = jax.random.normal(jax.random.key(0), (n,)) + 5.0
    obs = lsl.Var.new_obs(x, lsl.Dist(tfd.Normal, loc=param2, scale=1.0))

    # create subset calc for only param1 (cuts feedback from data)
    subset_calc = lsl.create_subset_log_prob_calc([param1], "lp_subset_param1")

    # build model
    gb = lsl.GraphBuilder().add(param1, param2, obs, subset_calc)
    model = gb.build_model()
    interface = gs.LieselInterface(model)

    # subset sampling setup
    builder_subset = gs.EngineBuilder(seed=42, num_chains=1)
    builder_subset.show_progress = False
    builder_subset.add_kernel(
        gs.HMCKernel(position_keys=["param1"], subset_logp_name="lp_subset_param1")
    )
    builder_subset.add_kernel(gs.HMCKernel(position_keys=["param2"]))
    builder_subset.set_model(interface)
    builder_subset.set_initial_values(model.state)
    builder_subset.set_duration(warmup_duration=1000, posterior_duration=1000)
    engine_subset = builder_subset.build()

    # full model sampling setup
    builder_full = gs.EngineBuilder(seed=42, num_chains=1)
    builder_full.show_progress = False
    builder_full.add_kernel(gs.HMCKernel(position_keys=["param1", "param2"]))
    builder_full.set_model(interface)
    builder_full.set_initial_values(model.state)
    builder_full.set_duration(warmup_duration=1000, posterior_duration=1000)
    engine_full = builder_full.build()

    # run sampling
    engine_subset.sample_all_epochs()
    results_subset = engine_subset.get_results()
    samples_subset = results_subset.get_posterior_samples()

    engine_full.sample_all_epochs()
    results_full = engine_full.get_results()
    samples_full = results_full.get_posterior_samples()

    # calculate sample statistics
    mean_subset_param1 = jnp.mean(samples_subset["param1"], axis=(0, 1))
    std_subset_param1 = jnp.std(samples_subset["param1"], axis=(0, 1))

    mean_full_param1 = jnp.mean(samples_full["param1"], axis=(0, 1))
    std_full_param1 = jnp.std(samples_full["param1"], axis=(0, 1))

    # analytical posteriors
    x_mean = jnp.mean(x)

    # full model: param1 | x ~ N(n*x_mean/(2n+1), sqrt((n+1)/(2n+1)))
    analytical_mean_full = (n * x_mean) / (2 * n + 1)
    analytical_std_full = jnp.sqrt((n + 1) / (2 * n + 1))

    # subset model: param1 ~ N(0, 1) (unchanged prior)
    analytical_mean_subset = 0.0
    analytical_std_subset = 1.0

    # verify subset sampling matches prior, cutting feedback works
    assert mean_subset_param1 == pytest.approx(analytical_mean_subset, abs=0.1)
    assert std_subset_param1 == pytest.approx(analytical_std_subset, abs=0.1)

    # verify full model incorporates data information
    assert mean_full_param1 == pytest.approx(analytical_mean_full, abs=0.1)
    assert std_full_param1 == pytest.approx(analytical_std_full, abs=0.1)

    # the subset model should have larger variance, less information
    assert std_subset_param1 > std_full_param1

    # verify the models produce meaningfully different results, cutting feedback effect
    mean_difference = jnp.abs(mean_subset_param1 - mean_full_param1)
    assert mean_difference > 1.0
