from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
from liesel.optim import (
    NegElboLoss,
    NegLogProbLoss,
    PositionSplit,
    PositionSplitManager,
    Split,
)
from liesel.optim.types import Position


def _normal_obs_model():
    y = lsl.Var.new_obs(
        jnp.arange(6.0),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="y",
    )
    return lsl.Model([y])


def _two_branch_model(n1: int = 8, n2: int = 5):
    y1 = lsl.Var.new_obs(
        jnp.arange(float(n1)),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="y1",
    )
    y2 = lsl.Var.new_obs(
        jnp.arange(float(n2)),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="y2",
    )
    return lsl.Model([y1, y2])


def _matrix_obs_model():
    y = lsl.Var.new_obs(
        jnp.arange(32.0).reshape(4, 8),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="y",
    )
    return lsl.Model([y])


def _empty_carry(model):
    return SimpleNamespace(
        batch=Position({}),
        fixed_position=Position({}),
        model_state=model.state,
    )


def test_neg_log_prob_loss_train_uses_full_training_split_not_current_batch():
    model = _normal_obs_model()
    split = Split(
        ["y"], axis_size=6, validate_axis_size=2, shuffle=False
    ).split_position(model.extract_position(["y"]))
    loss = NegLogProbLoss(model, split)
    carry = SimpleNamespace(
        batch=Position({"y": jnp.array([1000.0, 2000.0])}),
        fixed_position=Position({}),
        model_state=model.state,
    )

    value = loss.loss_train(Position({}), carry)
    train_state = model.update_state(split.train, model.state)
    manual = -split.scaled_log_lik(model, train_state, part="train")
    manual -= train_state["_model_log_prior"].value

    assert jnp.allclose(value, manual)


def test_neg_log_prob_loss_scale_uses_scalar_training_size():
    model = _normal_obs_model()
    split = Split(
        ["y"], axis_size=6, validate_axis_size=2, shuffle=False
    ).split_position(model.extract_position(["y"]))
    carry = _empty_carry(model)

    unscaled = NegLogProbLoss(model, split).loss_train(Position({}), carry)
    scaled_loss = NegLogProbLoss(model, split, scale=True)
    scaled = scaled_loss.loss_train(Position({}), carry)

    assert scaled_loss.scalar == split.train_axis_size
    assert jnp.allclose(scaled, unscaled / split.train_axis_size)


def test_neg_log_prob_loss_scale_uses_inferred_training_sample_size():
    model = _matrix_obs_model()
    split = PositionSplit.from_model(
        model,
        position_keys=["y"],
        validate_axis_share=0.25,
        split_axes={"y": 1},
    )
    carry = _empty_carry(model)

    unscaled = NegLogProbLoss(model, split).loss_train(Position({}), carry)
    scaled_loss = NegLogProbLoss(model, split, scale=True)
    scaled = scaled_loss.loss_train(Position({}), carry)

    assert split.train_axis_size == 6
    assert split.train_sample_size == 24.0
    assert scaled_loss.scalar == 24.0
    assert jnp.allclose(scaled, unscaled / 24.0)


def test_neg_log_prob_loss_scale_uses_total_unequal_branch_training_size():
    model = _two_branch_model()
    split = PositionSplitManager.from_model(model, position_keys=["y1", "y2"])
    carry = _empty_carry(model)

    unscaled = NegLogProbLoss(model, split).loss_train(Position({}), carry)
    scaled_loss = NegLogProbLoss(model, split, scale=True)
    scaled = scaled_loss.loss_train(Position({}), carry)
    scalar = sum(split.train_axis_sizes)

    assert scaled_loss.scalar == scalar
    assert jnp.allclose(scaled, unscaled / scalar)


def test_neg_log_prob_loss_scale_uses_total_equal_branch_training_size():
    model = _two_branch_model(n1=4, n2=4)
    position = model.extract_position(["y1", "y2"])
    split = PositionSplitManager(
        [
            Split(["y1"], axis_size=4).split_position(position),
            Split(["y2"], axis_size=4).split_position(position),
        ]
    )
    carry = _empty_carry(model)

    unscaled = NegLogProbLoss(model, split).loss_train(Position({}), carry)
    scaled_loss = NegLogProbLoss(model, split, scale=True)
    scaled = scaled_loss.loss_train(Position({}), carry)
    scalar = sum(split.train_axis_sizes)

    assert split.train_axis_size == 4
    assert scaled_loss.scalar == scalar
    assert jnp.allclose(scaled, unscaled / scalar)


def test_neg_log_prob_loss_rejects_unknown_validation_strategy():
    model = _normal_obs_model()
    split = PositionSplit.from_model(model, position_keys=["y"])

    with pytest.raises(ValueError, match="validation_strategy"):
        NegLogProbLoss(model, split, validation_strategy="deviance")


def test_neg_log_prob_loss_rejects_non_bool_scale():
    model = _normal_obs_model()
    split = PositionSplit.from_model(model, position_keys=["y"])

    with pytest.raises(ValueError, match="scale"):
        NegLogProbLoss(model, split, scale="yes")  # type: ignore[arg-type]


def test_neg_elbo_loss_train_uses_full_training_split_not_current_batch():
    train = Position({"y": jnp.array([1.0, 2.0, 3.0])})
    elbo = object.__new__(NegElboLoss)
    elbo.split = SimpleNamespace(train=train)
    elbo.q = SimpleNamespace(state={})
    elbo.nsamples = 7
    elbo.scalar = 1.0
    seen = {}

    def estimate_elbo(
        params,
        key,
        p_state,
        q_state,
        obs=None,
        split=None,
        split_part=None,
        nsamples=None,
    ):
        del params, key, p_state, q_state
        seen["obs"] = obs
        seen["split"] = split
        seen["split_part"] = split_part
        seen["nsamples"] = nsamples
        return jnp.array(5.0)

    elbo.estimate_elbo = estimate_elbo
    carry = SimpleNamespace(
        key=None,
        batch=Position({"y": jnp.array([1000.0])}),
        fixed_position=Position({}),
        model_state={},
    )

    value = elbo.loss_train(Position({}), carry)

    assert value == -5.0
    assert seen["nsamples"] == 7
    assert seen["split"] is elbo.split
    assert seen["split_part"] == "train"
    assert jnp.array_equal(seen["obs"]["y"], train["y"])


def test_neg_elbo_loss_monitoring_uses_training_split_scaling():
    train = Position({"y": jnp.array([1.0, 2.0, 3.0])})
    elbo = object.__new__(NegElboLoss)
    elbo.split = SimpleNamespace(train=train, has_validation=False)
    elbo.q = SimpleNamespace(state={})
    elbo.nsamples = 5
    elbo.scalar = 1.0
    seen = {}

    def estimate_elbo(
        params,
        key,
        p_state,
        q_state,
        obs=None,
        split=None,
        split_part=None,
        nsamples=None,
    ):
        del params, key, p_state, q_state
        seen["obs"] = obs
        seen["split"] = split
        seen["split_part"] = split_part
        seen["nsamples"] = nsamples
        return jnp.array(9.0)

    elbo.estimate_elbo = estimate_elbo
    carry = SimpleNamespace(
        key=None,
        fixed_position=Position({}),
        model_state={},
    )

    value = elbo.loss_validate(Position({}), carry)

    assert value == -9.0
    assert seen["nsamples"] == 5
    assert seen["split"] is elbo.split
    assert seen["split_part"] == "train"
    assert jnp.array_equal(seen["obs"]["y"], train["y"])


def test_neg_elbo_scale_uses_total_branch_training_size():
    model = _two_branch_model()
    split = PositionSplitManager.from_model(model, position_keys=["y1", "y2"])

    elbo = NegElboLoss(model, model, split=split, scale=True)

    assert elbo.scalar == sum(split.train_axis_sizes)


def test_neg_elbo_scale_uses_inferred_training_sample_size():
    model = _matrix_obs_model()
    split = PositionSplit.from_model(
        model,
        position_keys=["y"],
        split_axes={"y": 1},
    )

    elbo = NegElboLoss(model, model, split=split, scale=True)

    assert split.train_axis_size == 8
    assert split.train_sample_size == 32.0
    assert elbo.scalar == 32.0


def test_neg_elbo_rejects_non_bool_scale():
    model = _normal_obs_model()
    split = PositionSplit.from_model(model, position_keys=["y"])

    with pytest.raises(ValueError, match="scale"):
        NegElboLoss(model, model, split=split, scale="yes")  # type: ignore[arg-type]
