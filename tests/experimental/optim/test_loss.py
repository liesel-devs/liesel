from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
from liesel.experimental.optim import Elbo, NegLogProbLoss, PositionSplit, Split
from liesel.experimental.optim.types import Position


def _normal_obs_model():
    y = lsl.Var.new_obs(
        jnp.arange(6.0),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="y",
    )
    return lsl.Model([y])


def test_neg_log_prob_loss_train_uses_full_training_split_not_current_batch():
    model = _normal_obs_model()
    split = Split(["y"], n=6, n_validate=2, shuffle=False).split_position(
        model.extract_position(["y"])
    )
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


def test_neg_log_prob_loss_rejects_unknown_validation_strategy():
    model = _normal_obs_model()
    split = PositionSplit.from_model(model, position_keys=["y"])

    with pytest.raises(ValueError, match="validation_strategy"):
        NegLogProbLoss(model, split, validation_strategy="deviance")


def test_elbo_loss_train_uses_full_training_split_not_current_batch():
    train = Position({"y": jnp.array([1.0, 2.0, 3.0])})
    elbo = object.__new__(Elbo)
    elbo.split = SimpleNamespace(train=train)
    elbo.q = SimpleNamespace(state={})
    elbo.nsamples = 7
    elbo.scalar = 1.0
    seen = {}

    def evaluate(params, key, p_state, q_state, obs=None, nsamples=None):
        del params, key, p_state, q_state
        seen["obs"] = obs
        seen["nsamples"] = nsamples
        return jnp.array(5.0)

    elbo.evaluate = evaluate
    carry = SimpleNamespace(
        key=None,
        batch=Position({"y": jnp.array([1000.0])}),
        fixed_position=Position({}),
        model_state={},
    )

    value = elbo.loss_train(Position({}), carry)

    assert value == -5.0
    assert seen["nsamples"] == 7
    assert jnp.array_equal(seen["obs"]["y"], train["y"])
