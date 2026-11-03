"""Joint variational blocks that draw and score in one call."""

import jax
import jax.numpy as jnp
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
from liesel.optim.types import Position


def make_model():
    mu = lsl.Var.new_param(
        jnp.zeros(2), dist=lsl.Dist(tfd.Normal, 0.0, 10.0), name="mu"
    )
    tau = lsl.Var.new_param(
        jnp.array(0.0), dist=lsl.Dist(tfd.Normal, 0.0, 10.0), name="tau"
    )
    loc = lsl.Var.new_calc(lambda m, t: jnp.sum(m) + t, mu, tau, name="loc")
    y = lsl.Var.new_obs(
        jnp.array([0.5, 1.0, 1.5]), dist=lsl.Dist(tfd.Normal, loc, 1.0), name="y"
    )
    return lsl.Model(y)


def diag_normal(params, family_state, key, num_samples, given):
    loc, scale = params["mu_loc"], jnp.exp(params["mu_log_scale"])
    eps = jax.random.normal(key, (num_samples, *loc.shape), loc.dtype)
    draws = loc + scale * eps
    log_q = jnp.sum(tfd.Normal(loc, scale).log_prob(draws), axis=-1)
    return Position({"mu": draws}), log_q, family_state


def counting_normal(params, family_state, key, num_samples, given):
    draws, log_q, _ = diag_normal(params, (), key, num_samples, given)
    return draws, log_q, family_state + 1


INIT_PARAMS = {"mu_loc": jnp.zeros(2), "mu_log_scale": jnp.log(jnp.full(2, 0.1))}


def joint_block(model, fn=counting_normal, **kwargs):
    kwargs.setdefault("init_family_state", jnp.zeros((), dtype=jnp.int32))
    return opt.JointVDist(
        ["mu"], model, sample_and_log_prob=fn, init_params=INIT_PARAMS, **kwargs
    )


def make_loss(model=None, nsamples=8, **kwargs):
    model = make_model() if model is None else model
    tau_block = opt.VDist(["tau"], model).mvn_diag()
    q = opt.CompositeVDist(joint_block(model, **kwargs), tau_block).build()
    return opt.NegElboLoss.from_vdist(q, nsamples=nsamples)


def fit(loss, epochs=6, loss_monitor="train_full_data"):
    return opt.LieselVI(
        loss.p,
        loss=loss,
        optimizers=optax.adam(0.05),
        loss_monitor=loss_monitor,
        stopper=opt.Stopper(epochs=epochs, patience=epochs),
        show_progress=False,
    ).fit()


def test_joint_diag_normal_matches_mvn_diag():
    model = make_model()
    tau_block = opt.VDist(["tau"], model).mvn_diag()
    joint_q = opt.CompositeVDist(joint_block(model, fn=diag_normal), tau_block)
    joint_loss = opt.NegElboLoss.from_vdist(joint_q.build(), nsamples=20_000)
    mu_block = opt.VDist(["mu"], model).mvn_diag()
    vdist_q = opt.CompositeVDist(mu_block, opt.VDist(["tau"], model).mvn_diag())
    vdist_loss = opt.NegElboLoss.from_vdist(vdist_q.build(), nsamples=20_000)

    def elbo(loss, mu_loc_key):
        def f(mu_loc):
            params = loss.position(loss.parameters)
            params[mu_loc_key] = mu_loc
            return loss.estimate_elbo(params, jax.random.key(0))

        return jax.value_and_grad(f)(jnp.array([0.3, -0.2]))

    joint_value, joint_grad = elbo(joint_loss, "mu_loc")
    vdist_value, vdist_grad = elbo(vdist_loss, "(mu)_loc")

    assert jnp.allclose(joint_value, vdist_value, rtol=0.01)
    assert jnp.allclose(joint_grad, vdist_grad, rtol=0.02)


def test_joint_fn_is_called_once_for_all_samples():
    calls = []

    def recording(params, family_state, key, num_samples, given):
        calls.append(num_samples)
        return counting_normal(params, family_state, key, num_samples, given)

    loss = make_loss(fn=recording, nsamples=7)
    params = loss.position(loss.parameters)

    elbo, states = loss.estimate_elbo_and_state(params, jax.random.key(0))

    assert calls == [7]
    assert elbo.shape == ()
    assert int(states[0]) == 1
    assert loss.estimate_elbo(params, jax.random.key(0)).shape == ()


def test_family_state_counts_training_steps():
    loss = make_loss()
    result = fit(loss, epochs=6)

    assert int(result.loss_state_final[0]) == 6
    assert int(loss.init_loss_state()[0]) == 0


def test_posterior_sampling_uses_final_family_state_and_keeps_none():
    seen = []

    def recording(params, family_state, key, num_samples, given):
        seen.append(family_state)
        return counting_normal(params, family_state, key, num_samples, given)

    loss = make_loss(fn=recording)
    result = fit(loss, epochs=4)
    posterior = loss.approximate_joint_posterior(result)
    seen.clear()

    draws = posterior.sample((3, 5), seed=jax.random.key(1))

    assert [int(state) for state in seen] == [4]
    assert draws["mu"].shape == (3, 5, 2)
    assert draws["tau"].shape == (3, 5)
    assert int(result.loss_state_final[0]) == 4


def test_min_monitor_posterior_uses_matching_family_state():
    seen = []

    def recording(params, family_state, key, num_samples, given):
        seen.append(family_state)
        return counting_normal(params, family_state, key, num_samples, given)

    loss = make_loss(fn=recording)
    result = fit(loss, epochs=6, loss_monitor=opt.EmaTrainLossMonitor(1.0))
    posterior = loss.approximate_joint_posterior(result, at="min_monitor")
    seen.clear()

    posterior.sample(2, seed=jax.random.key(1))

    assert result.min_monitor_epoch is not None
    expected = int(result.loss_state_min_monitor[0])
    assert expected == result.min_monitor_epoch + 1
    assert [int(state) for state in seen] == [expected]


def test_joint_block_sees_vdist_draws_as_given():
    seen = []

    def reads_given(params, family_state, key, num_samples, given):
        seen.append(given)
        draws, log_q, state = counting_normal(
            params, family_state, key, num_samples, given
        )
        return Position({"mu": draws["mu"] + given["tau"][:, None]}), log_q, state

    loss = make_loss(fn=reads_given)
    loss.estimate_elbo(loss.position(loss.parameters), jax.random.key(0))
    assert list(seen[0]) == ["tau"]
    assert seen[0]["tau"].shape == (8,)

    result = fit(loss, epochs=2)
    seen.clear()
    draws = loss.approximate_joint_posterior(result).sample(4, seed=jax.random.key(2))

    assert jnp.array_equal(seen[0]["tau"], draws["tau"])


def test_penalty_enters_training_loss():
    model = make_model()
    plain = make_loss(model)
    penalized = make_loss(model, penalty=lambda params: 5.0 + 0.0 * params["mu_loc"][0])
    params = plain.position(plain.parameters)
    key = jax.random.key(0)

    difference = plain.estimate_elbo(params, key) - penalized.estimate_elbo(params, key)

    assert jnp.allclose(difference, 5.0)


def test_loss_parameters_include_joint_parameters_and_are_trained():
    loss = make_loss()
    result = fit(loss, epochs=5)

    assert {"mu_loc", "mu_log_scale"} <= set(loss.parameters)
    assert not jnp.array_equal(result.position_final["mu_loc"], INIT_PARAMS["mu_loc"])
    assert loss.vdist.parameters == loss.parameters


def test_composite_of_only_joint_blocks():
    model = make_model()

    def tau_block(params, family_state, key, num_samples, given):
        draws = params["tau_loc"] + jax.random.normal(key, (num_samples,))
        return (
            Position({"tau": draws}),
            tfd.Normal(params["tau_loc"], 1.0).log_prob(draws),
            family_state,
        )

    q = opt.CompositeVDist(
        joint_block(model),
        opt.JointVDist(
            ["tau"],
            model,
            sample_and_log_prob=tau_block,
            init_params={"tau_loc": jnp.array(0.0)},
        ),
    ).build()
    loss = opt.NegElboLoss.from_vdist(q, nsamples=4)
    result = fit(loss, epochs=3)

    draws = loss.approximate_joint_posterior(result).sample(5, seed=jax.random.key(0))

    assert int(result.loss_state_final[0]) == 3
    assert draws["tau"].shape == (5,)
    assert draws["mu"].shape == (5, 2)


def test_joint_vdist_validates_its_inputs():
    model = make_model()
    with pytest.raises(ValueError, match="position_key"):
        opt.JointVDist([], model, sample_and_log_prob=diag_normal, init_params={})
    with pytest.raises(ValueError, match="Duplicate"):
        opt.JointVDist(
            ["mu", "mu"], model, sample_and_log_prob=diag_normal, init_params={}
        )


def test_composite_rejects_clashing_parameter_names():
    model = make_model()
    other = opt.JointVDist(
        ["tau"], model, sample_and_log_prob=diag_normal, init_params=INIT_PARAMS
    )

    with pytest.raises(ValueError, match="mu_loc"):
        opt.CompositeVDist(joint_block(model), other).build()


def test_joint_fn_with_wrong_output_is_rejected():
    def wrong_keys(params, family_state, key, num_samples, given):
        draws, log_q, state = counting_normal(
            params, family_state, key, num_samples, given
        )
        return Position({"nu": draws["mu"]}), log_q, state

    loss = make_loss(fn=wrong_keys)

    with pytest.raises(ValueError, match="draws"):
        loss.estimate_elbo(loss.position(loss.parameters), jax.random.key(0))
