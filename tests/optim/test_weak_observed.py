"""Copula factors follow their strong data inputs through splits and batches."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
from liesel.distributions import GaussianCopula


def copula_model(extra=False):
    loc = lsl.Var.new_param(0.0, name="loc")
    rho = lsl.Var.new_param(0.2, bijector=tfb.Tanh(), name="rho")
    x1 = lsl.Var.new_obs(
        jnp.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]) + 0.3,
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="x1",
    ).update()
    x2 = lsl.Var.new_obs(
        jnp.array([-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0]) + 0.3,
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="x2",
    ).update()
    copula = lsl.Var.new_calc(
        lambda a, b: jnp.stack((a, b), axis=-1),
        lsl.PIT(x1).update(),
        lsl.PIT(x2).update(),
        dist=lsl.Dist(GaussianCopula, dependence=rho),
        name="copula",
    )
    copula.observed = True
    roots = [copula]
    if extra:
        roots.append(
            lsl.Var.new_obs(
                jnp.arange(6.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="extra"
            )
        )
    return lsl.Model(roots)


def joint_log_prob(model, state):
    values = model.extract_position(["x1", "x2", "loc", "rho"], state)
    rho = values["rho"]
    dist = tfd.MultivariateNormalFullCovariance(
        loc=jnp.repeat(values["loc"], 2),
        covariance_matrix=jnp.array([[1.0, rho], [rho, 1.0]]),
    )
    return dist.log_prob(jnp.stack((values["x1"], values["x2"]), axis=-1))


FACTORIES = [
    opt.PositionSplit,
    opt.PositionSplitManager,
    opt.Split,
    opt.SplitManager,
    opt.Batches,
    opt.BatchManager,
]


@pytest.mark.parametrize("factory", FACTORIES)
def test_factories_select_strong_inputs_and_reject_explicit_weak_keys(factory):
    model = copula_model()
    kwargs = {"batch_size": 2} if factory in (opt.Batches, opt.BatchManager) else {}
    result = factory.from_model(model, **kwargs)
    assert set(result.position_keys) == {"x1", "x2"}
    for key in ("copula", model.vars["copula"].value_node.name):
        with pytest.raises(ValueError, match="weak variable.*strong source"):
            factory.from_model(model, position_keys=[key], **kwargs)
    with pytest.raises(ValueError, match="one explicit group"):
        factory.from_model(model, position_keys=[["x1"], ["x2"]], **kwargs)


def test_full_data_copula_fit_and_parameter_gradients_match_joint_normal():
    model = copula_model()
    engine = opt.LieselOptim(
        model,
        optimizers="lbfgs",
        loss_monitor="train_full_data",
        scale_loss=False,
        show_progress=False,
        stopper=opt.Stopper(epochs=40, patience=8),
    ).build_engine()
    carry = engine._init_carry(40)
    original = model.vars["copula"].value
    for value in (-0.2, 0.4):
        position = model.extract_position(engine.position_keys)
        position["loc"] = jnp.asarray(value, dtype=position["loc"].dtype)
        actual, grad = jax.value_and_grad(engine.loss.loss_train)(position, carry)

        def expected(p):
            return -joint_log_prob(model, model.update_state(p, model.state)).sum()

        ref, ref_grad = jax.value_and_grad(expected)(position)
        np.testing.assert_allclose(actual, ref, rtol=2e-6)
        for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(ref_grad), strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
        state = model.update_state(position, model.state)
        assert not jnp.array_equal(
            model.extract_position(["copula"], state)["copula"], original
        )
    result = engine.fit()
    fitted = model.extract_position(
        ["loc", "rho"], model.update_state(result.position_final, model.state)
    )
    np.testing.assert_allclose([fitted["loc"], fitted["rho"]], [0.3, 0.5], atol=2e-4)


def test_heldout_copula_scores_include_weak_factor_and_exclude_unsplit_data():
    model = copula_model(extra=True)
    split = opt.PositionSplit.from_model(
        model,
        position_keys=["x1", "x2"],
        validate_axis_share=0.25,
        test_axis_share=0.25,
        shuffle=False,
    )
    for part, scale in (("train", 1.0), ("validate", 2.0), ("test", 2.0)):
        state = model.update_state(getattr(split, part), model.state)
        expected = scale * joint_log_prob(model, state).sum()
        if part == "train":
            expected += state["extra_log_prob"].value.sum()
        np.testing.assert_allclose(
            split.scaled_log_lik(model, state, part), expected, rtol=2e-6
        )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("manager", [False, True])
def test_copula_minibatches_scale_all_factors_and_preserve_other_groups(
    weighted, manager
):
    model = copula_model(extra=manager)
    kwargs = (
        {
            "sampling_weights": {"x1": jnp.arange(1.0, 9.0)}
            if manager
            else jnp.arange(1.0, 9.0)
        }
        if weighted
        else {}
    )
    factory = opt.BatchManager if manager else opt.Batches
    batches = factory.from_model(
        model, batch_size=2, sample_with_replacement=weighted, **kwargs
    ).start_epoch(jax.random.key(2))
    children = batches.batches if manager else (batches,)
    for i in range(batches.n_full_batches):
        data = batches.extract_batched_position(model, model.state, i)
        state = model.update_state(data, model.state)
        expected = 0.0
        for child in children:
            terms = (
                joint_log_prob(model, state)
                if "x1" in child.position_keys
                else state["extra_log_prob"].value
            )
            expected += child.batch_sample_scale * jnp.sum(
                terms * child.correction_factors(i)
            )
        np.testing.assert_allclose(
            batches.scaled_log_lik(model, state, batch_index=i), expected, rtol=2e-6
        )


def test_weak_copula_likelihood_can_use_explicit_axis_in_weighted_manager():
    model = copula_model(extra=True)
    batches = opt.BatchManager.from_model(
        model,
        batch_size=2,
        sample_with_replacement=True,
        sampling_weights={"x1": jnp.arange(1.0, 9.0)},
        likelihood_axes={"copula": 0},
    ).start_epoch(jax.random.key(1))
    state = model.update_state(
        batches.extract_batched_position(model, model.state, 0), model.state
    )
    assert jnp.isfinite(batches.scaled_log_lik(model, state, batch_index=0))


def test_manually_constructed_groups_cannot_split_one_copula_factor():
    model = copula_model()
    split = opt.PositionSplit.from_model(model)
    groups = opt.PositionSplitManager(
        [
            opt.Split([key], axis_size=8).split_position(model.extract_position([key]))
            for key in ("x1", "x2")
        ]
    )
    with pytest.raises(ValueError, match="one explicit group"):
        opt.NegLogProbLoss(model, groups)
    batches = opt.BatchManager(
        [opt.Batches([key], axis_size=8, batch_size=2) for key in ("x1", "x2")]
    )
    with pytest.raises(ValueError, match="one explicit group"):
        opt.LieselOptim(
            model,
            split=split,
            batches=batches,
            optimizers=optax.sgd(0.01),
            loss_monitor="train_full_data",
        ).build_engine()


def test_weighted_multidimensional_weak_factor_requires_likelihood_axis():
    x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
    y = lsl.Var.new_calc(
        lambda x: jnp.stack((x, x)),
        x,
        dist=lsl.Dist(tfd.Normal, 0.0, 1.0),
        name="y",
    )
    y.observed = True
    model = lsl.Model([y])
    batches = opt.Batches.from_model(
        model,
        batch_size=2,
        sample_with_replacement=True,
        sampling_weights=jnp.arange(1.0, 9.0),
    ).start_epoch(jax.random.key(1))
    state = model.update_state(
        batches.extract_batched_position(model, model.state, 0), model.state
    )
    with pytest.raises(ValueError, match="Set likelihood_axes for weak"):
        batches.scaled_log_lik(model, state, batch_index=0)
    batches.likelihood_axes = {"y": 1}
    expected = 4 * jnp.sum(state["y_log_prob"].value * batches.correction_factors(0))
    np.testing.assert_allclose(
        batches.scaled_log_lik(model, state, batch_index=0), expected
    )


@pytest.mark.parametrize("factory", FACTORIES)
def test_automatic_grouping_when_only_a_weak_variable_has_a_likelihood(factory):
    loc = lsl.Var.new_param(0.0, name="loc")
    response = lsl.Var.new_obs(jnp.arange(6.0), name="response")
    residual = lsl.Var.new_calc(
        lambda response, loc: response - loc,
        response,
        loc,
        dist=lsl.Dist(tfd.Normal, 0.0, 1.0),
        name="residual",
    )
    residual.observed = True
    model = lsl.Model([residual])
    kwargs = {"batch_size": 2} if factory in (opt.Batches, opt.BatchManager) else {}
    data = factory.from_model(model, **kwargs)
    assert list(data.position_keys) == ["response"]

    engine = opt.LieselOptim(
        model,
        optimizers=optax.sgd(0.1),
        loss_monitor="train_full_data",
        scale_loss=False,
        show_progress=False,
        stopper=opt.Stopper(epochs=2, patience=2),
    ).build_engine()
    assert list(engine.split.split_position_keys) == ["response"]
    carry = engine._init_carry(2)
    np.testing.assert_allclose(
        engine.loss.loss_train({"loc": jnp.array(0.0)}, carry), -model.log_prob
    )
    assert engine.fit().position_final["loc"] > 0
