"""Behavior of the documented variational inference defaults."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt


def model():
    z = lsl.Var.new_param(jnp.zeros(2), name="z")
    y = lsl.Var.new_obs(jnp.zeros(2), lsl.Dist(tfd.Normal, z, 1.0), name="y")
    return lsl.Model(y)


@pytest.mark.parametrize("family", ["normal", "mvn_diag", "mvn_tril"])
def test_gaussian_builder_initial_standard_deviation(family):
    builder = getattr(opt.VDist(["z"], model()), family)()
    np.testing.assert_allclose(builder.var.dist_node.init_dist().stddev(), [0.1, 0.1])


@pytest.mark.parametrize("family", ["mvn_diag", "mvn_tril", "mvn_blocked"])
def test_gaussian_loss_initial_standard_deviation(family):
    loss = getattr(opt.NegElboLoss, family)(model())
    for var in loss.q.observed.values():
        np.testing.assert_allclose(var.dist_node.init_dist().stddev(), [0.1, 0.1])


def test_elbo_uses_current_target_state_unless_supplied():
    target = model()
    loss = opt.NegElboLoss.mvn_diag(target)
    params = loss.position(list(loss.q.parameters))
    key = jax.random.key(17)
    previous_state = target.state
    previous = loss.estimate_elbo(params, key, previous_state)
    target.vars["y"].value = jnp.full(2, 5.0)
    implicit = loss.estimate_elbo(params, key)
    current = loss.estimate_elbo(params, key, target.state)
    np.testing.assert_array_equal(implicit, current)
    assert not np.isclose(implicit, previous)
    np.testing.assert_array_equal(
        loss.estimate_elbo(params, key, previous_state), previous
    )


@pytest.mark.parametrize("composite", [False, True])
def test_builder_sampling_shape_seed_and_position(composite):
    block = opt.VDist(["z"], model()).mvn_diag()
    builder = opt.CompositeVDist(block).build() if composite else block.build()
    key = jax.random.key(81)
    loc_key = next(name for name in builder.parameters if name.endswith("_loc"))
    position = {loc_key: jnp.full(2, 3.0)}
    assert builder.sample(seed=key)["z"].shape == (2,)
    for shape, expected in [(4, (4, 2)), ((2, 4), (2, 4, 2))]:
        draws = builder.sample(shape, seed=key)["z"]
        shifted = jax.jit(
            lambda k, shape=shape: builder.sample(shape, seed=k, at_position=position)[
                "z"
            ]
        )(key)
        assert draws.shape == expected and shifted.shape == expected
        np.testing.assert_allclose(shifted - draws, 3.0, atol=1e-6)
    sample: Any = builder.sample
    with pytest.raises(TypeError):
        sample(key)
    with pytest.raises(TypeError):
        sample(key, (4,))


@pytest.mark.parametrize("constructor", ["direct", "from_vdist"])
@pytest.mark.parametrize("entropy", ["auto", "mc"])
def test_q_prior_penalty_is_opt_in_and_target_prior_remains(constructor, entropy):
    z = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="z")
    y = lsl.Var.new_obs(jnp.array([2.0]), lsl.Dist(tfd.Normal, z, 1.0), name="y")
    target = lsl.Model(y)
    loc = lsl.Var.new_param(2.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="q_loc")
    builder = opt.VDist(["z"], target).init(lsl.Dist(tfd.Normal, loc, 1.0)).build()
    assert builder.q is not None and builder.var is not None
    if constructor == "direct":
        loss = opt.NegElboLoss(
            target, builder.q, q_to_p=builder.q_to_p, nsamples=16, entropy=entropy
        )
        penalized = opt.NegElboLoss(
            target,
            builder.q,
            q_to_p=builder.q_to_p,
            regularize_q_prior=True,
            nsamples=16,
            entropy=entropy,
        )
    else:
        loss = opt.NegElboLoss.from_vdist(builder, nsamples=16, entropy=entropy)
        penalized = opt.NegElboLoss.from_vdist(
            builder, regularize_q_prior=True, nsamples=16, entropy=entropy
        )
    params = loss.position(builder.parameters)
    key = jax.random.key(15)
    draws = builder.q.sample((16,), seed=key, newdata=params, fixed=builder.parameters)[
        builder.var.name
    ]
    # Gaussian log joint plus entropy; the z**2 term is the target prior.
    entropy_term = 1.0 if entropy == "auto" else jnp.mean((draws - 2.0) ** 2)
    ordinary = -0.5 * jnp.log(2 * jnp.pi) + 0.5 * (
        entropy_term - jnp.mean(draws**2 + (2.0 - draws) ** 2)
    )
    value, gradient = jax.value_and_grad(loss.estimate_elbo)(params, key)
    value_penalized, gradient_penalized = jax.value_and_grad(penalized.estimate_elbo)(
        params, key
    )
    np.testing.assert_allclose(value, ordinary, rtol=1e-6)
    np.testing.assert_allclose(gradient["q_loc"], 2.0 - 2.0 * draws.mean(), rtol=1e-6)
    np.testing.assert_allclose(
        value_penalized - value, -0.5 * jnp.log(2 * jnp.pi) - 2.0, rtol=1e-6
    )
    np.testing.assert_allclose(gradient_penalized["q_loc"] - gradient["q_loc"], -2.0)


@pytest.mark.parametrize("family", ["mvn_diag", "mvn_tril", "mvn_blocked"])
def test_q_prior_penalty_default_propagates_through_factories_and_wrapper(family):
    assert getattr(opt.NegElboLoss, family)(model()).regularize_q_prior is False
    vi = opt.LieselVI(
        model(),
        loss=family,
        optimizers=optax.adam(0.01),
        loss_monitor="train_full_data",
    )
    assert vi.loss.regularize_q_prior is False
    explicit = getattr(opt.NegElboLoss, family)(model(), regularize_q_prior=True)
    assert explicit.regularize_q_prior is True
