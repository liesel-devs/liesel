import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl


def rw_proposal(key, model_state, step_size):
    a = model_state["a_value"].value
    return gs.MHProposal({"a": a + step_size * jax.random.normal(key)}, 0.0)


def model_with_cut() -> lsl.Model:
    # without the cut, the posterior of a is close to 5
    a = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="a")
    y = lsl.Var.new_obs(jnp.full(20, 5.0), lsl.Dist(tfd.Normal, a, 0.5), name="y")
    return lsl.Model([y])


@pytest.mark.parametrize(
    "kernel",
    [
        gs.NUTSKernel,
        gs.HMCKernel,
        gs.IWLSKernel,
        gs.RWKernel,
        lambda keys, **kwargs: gs.MHKernel(keys, rw_proposal, **kwargs),
    ],
    ids=["nuts", "hmc", "iwls", "rw", "mh"],
)
def test_kernel_samples_target(kernel) -> None:
    model = model_with_cut()
    target = model.target(exclude=["y"])

    builder = gs.EngineBuilder(seed=1, num_chains=2)
    builder.set_model(gs.LieselInterface(model))
    builder.set_initial_values(model.state)
    builder.add_kernel(kernel(["a"], target=target))
    builder.set_duration(warmup_duration=300, posterior_duration=500)
    builder.show_progress = False
    engine = builder.build()
    engine.sample_all_epochs()

    samples = engine.get_results().get_posterior_samples()["a"]
    assert jnp.mean(samples) == pytest.approx(0.0, abs=0.5)
    assert jnp.std(samples) == pytest.approx(1.0, abs=0.3)


def test_mcmc_spec_target() -> None:
    cut = lsl.TargetSpec(exclude=["y"])
    a = lsl.Var.new_param(
        0.0,
        lsl.Dist(tfd.Normal, 0.0, 1.0),
        inference=gs.MCMCSpec(gs.NUTSKernel, target=cut),
        name="a",
    )
    b = lsl.Var.new_param(
        0.0,
        lsl.Dist(tfd.Normal, a, 1.0),
        inference=gs.MCMCSpec(gs.NUTSKernel),
        name="b",
    )
    y = lsl.Var.new_obs(jnp.zeros(3), lsl.Dist(tfd.Normal, b, 1.0), name="y")
    model = lsl.Model([y])

    kernel_a, kernel_b = sorted(
        gs.LieselMCMC(model).get_kernel_list(), key=lambda k: k.position_keys
    )
    assert isinstance(kernel_a, gs.NUTSKernel)
    assert isinstance(kernel_b, gs.NUTSKernel)
    assert isinstance(kernel_a.target, lsl.LogProbTarget)
    assert kernel_a.target.weights == {"a_log_prob": 1.0, "b_log_prob": 1.0}
    assert kernel_b.target is None

    b.inference = gs.MCMCSpec(gs.NUTSKernel, target=lsl.TargetSpec(include=["a"]))
    with pytest.raises(ValueError, match="'b' does not affect its target"):
        gs.LieselMCMC(model).get_kernel_list()
