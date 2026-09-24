import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
from liesel.optim import (
    Batches,
    NegLogProbLoss,
    PositionSplit,
    PositionSplitManager,
    Split,
)
from liesel.optim.state import OptimCarry
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


def _empty_carry(model) -> OptimCarry:
    return OptimCarry.new(
        key=jax.random.key(0),
        epochs=1,
        position=Position({}),
        batches=Batches([], axis_size=1, batch_size=None),
        optimizers=[],
        model_state=model.state,
        save_position_history=False,
    )


def test_neg_log_prob_loss_train_uses_full_training_split_not_current_batch():
    model = _normal_obs_model()
    split = Split(
        ["y"], axis_size=6, validate_axis_size=2, shuffle=False
    ).split_position(model.extract_position(["y"]))
    loss = NegLogProbLoss(model, split)
    carry = _empty_carry(model)
    carry.batch = Position({"y": jnp.array([1000.0, 2000.0])})

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


def test_passthrough_likelihood_is_not_split_scaled_or_batched_by_default():
    y = lsl.Var.new_obs(
        jnp.arange(10.0),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="y",
    )
    z = lsl.Var.new_obs(
        jnp.arange(3.0),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="z",
    )
    model = lsl.Model([y, z])
    split = PositionSplit.from_model(
        model,
        position_keys=["y", "z"],
        validate_axis_share=0.2,
        split_axes={"y": 0, "z": None},
    )

    state = model.update_state(split.validate, model.state)
    value = split.scaled_log_lik(model, state)
    manual = split.validate_sample_scale * state["y_log_prob"].value.sum()
    batches = Batches.from_split(split, batch_size=2, shuffle=False)

    assert jnp.allclose(value, manual)
    assert batches.position_keys == ["y"]


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
        NegLogProbLoss(
            model,
            split,
            validation_strategy="deviance",  # ty: ignore[invalid-argument-type]
        )


def test_neg_log_prob_loss_rejects_non_bool_scale():
    model = _normal_obs_model()
    split = PositionSplit.from_model(model, position_keys=["y"])

    with pytest.raises(ValueError, match="scale"):
        NegLogProbLoss(model, split, scale="yes")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("strategy", ["log_lik", "log_prob"])
def test_held_out_scores_exclude_unsplit_likelihoods(managed, passthrough, strategy):
    held_out_scores = []
    training_scores = []
    for offset in (0.0, 100.0):
        loc = lsl.Var.new_param(
            jnp.array(0.5), lsl.Dist(tfd.Normal, loc=0.0, scale=2.0), name="loc"
        )
        ys = [
            lsl.Var.new_obs(
                jnp.arange(float(n)),
                lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
                name=f"y{i}",
            )
            for i, n in enumerate((10, 6) if managed else (10,))
        ]
        z = lsl.Var.new_obs(
            jnp.full(100, offset), lsl.Dist(tfd.Normal, loc=loc, scale=1.0), name="z"
        )
        model = lsl.Model([*ys, z])
        keys = [y.name for y in ys] + (["z"] if passthrough else [])
        split = PositionSplit.from_model(
            model,
            position_keys=keys,
            split_axes={"z": None},
            validate_axis_share=0.2,
            test_axis_share=0.2,
            multi_size="manager",
            shuffle=False,
        )
        loss = NegLogProbLoss(model, split, validation_strategy=strategy, scale=True)
        carry = _empty_carry(model)
        params = Position({"loc": loc.value})
        children = split.splits if isinstance(split, PositionSplitManager) else [split]
        scores = []
        for part in ("validate", "test"):
            state = model.update_state(getattr(split, part), model.state)
            manual = sum(
                child.sample_scale(part)
                * tfd.Normal(loc=loc.value, scale=1.0)
                .log_prob(getattr(child, part)[y.name])
                .sum()
                for child, y in zip(children, ys, strict=True)
            )
            actual = split.scaled_log_lik(model, state, part=part)
            assert jnp.allclose(actual, manual)
            scores.append(float(actual))
            if part == "validate":
                prior = tfd.Normal(loc=0.0, scale=2.0).log_prob(loc.value)
                expected = -(manual + (prior if strategy == "log_prob" else 0.0))
                assert jnp.allclose(
                    loss.loss_monitor(params, carry), expected / loss.scalar
                )
        held_out_scores.append(scores)
        training_scores.append(float(loss.loss_train(params, carry)))
        full_split = PositionSplit.from_model(
            model, position_keys=keys, split_axes={"z": None}, multi_size="manager"
        )
        fallback = NegLogProbLoss(model, full_split, validation_strategy="log_prob")
        assert jnp.allclose(
            fallback.loss_monitor(params, carry), fallback.loss_train(params, carry)
        )
    assert held_out_scores[0] == held_out_scores[1]
    assert training_scores[0] != training_scores[1]


def test_validation_best_position_ignores_unsplit_training_branch():
    import optax

    from liesel.optim import LieselOptim, Optimizer, Stopper

    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    y = lsl.Var.new_obs(
        jnp.zeros(4), lsl.Dist(tfd.Normal, loc=loc, scale=1.0), name="y"
    )
    z = lsl.Var.new_obs(
        jnp.full(100, 10.0), lsl.Dist(tfd.Normal, loc=loc, scale=1.0), name="z"
    )
    model = lsl.Model([y, z])
    split = PositionSplit.from_model(
        model, position_keys=["y"], validate_axis_share=0.5
    )
    result = LieselOptim(
        model,
        split=split,
        optimizers=[Optimizer(["loc"], optax.sgd(0.0001))],
        loss_monitor="validation",
        stopper=Stopper(epochs=4, patience=4),
        scale_loss=False,
        show_progress=False,
    ).fit()
    # Training moves toward z=10; validation is best at the first completed epoch.
    assert result.min_monitor_epoch == 0
    assert result.position_min_monitor["loc"] < result.position_final["loc"]
    assert result.history.position is not None
    expected = -2 * tfd.Normal(loc=result.history.position["loc"], scale=1.0).log_prob(
        0.0
    )
    assert jnp.allclose(result.history.loss_monitor, expected)
