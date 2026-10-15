from types import SimpleNamespace

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
from liesel.optim.types import Position


def pointwise_sum(state, node_name):
    return state[node_name].value.sum()


def empty_carry(model, **kwargs):
    values = {
        "model_state": model.state,
        "fixed_position": Position({}),
        "batch": Position({}),
        "batches": None,
        "i_batch": 0,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def normal_response(name, value, loc=0.0):
    return lsl.Var.new_obs(
        jnp.asarray(value), lsl.Dist(tfd.Normal, loc=loc, scale=1.0), name=name
    )


def simple_model(n=12):
    y = normal_response("y", jnp.linspace(-1.5, 1.5, n))
    return lsl.Model([y])


def split_array_model(n_rows=4, n_cols=10):
    X_value = jnp.linspace(-1.0, 1.0, 2 * n_cols).reshape(2, n_cols)
    y_value = jnp.linspace(-0.5, 1.5, n_rows * n_cols).reshape(n_rows, n_cols)
    X = lsl.Var.new_obs(X_value, name="X")
    loc = lsl.Var.new_calc(lambda x: x.mean(axis=0), X, name="loc")
    y = normal_response("y", y_value, loc=loc)
    return lsl.Model([y])


def batch_array_model(n_rows=3, n_cols=12):
    X_value = jnp.linspace(-2.0, 2.0, 2 * n_cols).reshape(2, n_cols)
    y_value = jnp.linspace(-1.0, 1.0, n_rows * n_cols).reshape(n_rows, n_cols)
    X = lsl.Var.new_obs(X_value, name="X")
    loc = lsl.Var.new_calc(lambda x: x.mean(axis=0), X, name="loc")
    y = normal_response("y", y_value, loc=loc)
    return lsl.Model([y])


def split_batch_array_model(n_rows=6, n_cols=8):
    X_split_value = jnp.linspace(-1.0, 1.0, n_rows * 2).reshape(n_rows, 2)
    Z_batch_value = jnp.linspace(-0.5, 0.5, 3 * n_cols).reshape(3, n_cols)
    y_value = jnp.linspace(-2.0, 2.0, n_rows * n_cols).reshape(n_rows, n_cols)
    X_split = lsl.Var.new_obs(X_split_value, name="X_split")
    Z_batch = lsl.Var.new_obs(Z_batch_value, name="Z_batch")
    loc = lsl.Var.new_calc(
        lambda x, z: x[:, :1] + z.mean(axis=0)[None, :], X_split, Z_batch, name="loc"
    )
    y = normal_response("y", y_value, loc=loc)
    return lsl.Model([y])


def test_split_scalar():
    model = simple_model(10)
    split = opt.PositionSplit.from_model(
        model, ["y"], validate_axis_share=0.2, test_axis_share=0.1, shuffle=False
    )
    loss = opt.NegLogProbLoss(model, split, scale=False)
    scaled_loss = opt.NegLogProbLoss(model, split, scale=True)
    carry = empty_carry(model)
    train_state = model.update_state(split.train, model.state)
    validate_state = model.update_state(split.validate, model.state)
    assert jnp.allclose(
        loss.loss_train(Position({}), carry), -pointwise_sum(train_state, "y_log_prob")
    ), "train loss"
    assert jnp.allclose(
        loss.loss_monitor(Position({}), carry),
        -split.validate_sample_scale * pointwise_sum(validate_state, "y_log_prob"),
    ), "validation loss"
    assert jnp.allclose(
        scaled_loss.loss_train(Position({}), carry),
        -pointwise_sum(train_state, "y_log_prob") / split.train_sample_size,
    ), "scaled train loss"


def test_split_array():
    model = split_array_model()
    split = opt.PositionSplit.from_model(
        model,
        ["X", "y"],
        validate_axis_share=0.2,
        test_axis_share=0.1,
        split_axes={"X": 1, "y": 1},
        shuffle=False,
    )
    loss = opt.NegLogProbLoss(model, split, scale=False)
    scaled_loss = opt.NegLogProbLoss(model, split, scale=True)
    carry = empty_carry(model)
    train_state = model.update_state(split.train, model.state)
    validate_state = model.update_state(split.validate, model.state)
    assert jnp.allclose(
        loss.loss_train(Position({}), carry), -pointwise_sum(train_state, "y_log_prob")
    ), "train loss"
    assert jnp.allclose(
        loss.loss_monitor(Position({}), carry),
        -split.validate_sample_scale * pointwise_sum(validate_state, "y_log_prob"),
    ), "validation loss"
    assert jnp.allclose(
        scaled_loss.loss_monitor(Position({}), carry),
        -split.validate_sample_scale
        * pointwise_sum(validate_state, "y_log_prob")
        / split.train_sample_size,
    ), "scaled validation loss"


def test_batch_scalar():
    model = simple_model(12)
    split = opt.PositionSplit.from_model(model, ["y"], shuffle=False)
    batches = opt.Batches.from_model(
        model, batch_size=4, position_keys=["y"], shuffle=False
    )
    batch = batches.get_batched_position(model.extract_position(["y"]), 0)
    loss = opt.NegLogProbLoss(model, split, scale=False)
    scaled_loss = opt.NegLogProbLoss(model, split, scale=True)
    carry = empty_carry(model, batch=batch, batches=batches)
    batch_state = model.update_state(batch, model.state)
    assert jnp.allclose(
        loss.loss_train_batched(Position({}), carry),
        -batches.batch_sample_scale * pointwise_sum(batch_state, "y_log_prob"),
    ), "batch loss"
    assert jnp.allclose(
        scaled_loss.loss_train_batched(Position({}), carry),
        -batches.batch_sample_scale
        * pointwise_sum(batch_state, "y_log_prob")
        / split.train_sample_size,
    ), "scaled batch loss"


def test_batch_array():
    model = batch_array_model()
    split = opt.PositionSplit.from_model(
        model, ["X", "y"], split_axes={"X": 1, "y": 1}, shuffle=False
    )
    batches = opt.Batches.from_model(
        model,
        batch_size=4,
        position_keys=["X", "y"],
        batch_axes={"X": 1, "y": 1},
        shuffle=False,
    )
    batch = batches.get_batched_position(model.extract_position(["X", "y"]), 0)
    loss = opt.NegLogProbLoss(model, split, scale=False)
    scaled_loss = opt.NegLogProbLoss(model, split, scale=True)
    carry = empty_carry(model, batch=batch, batches=batches)
    batch_state = model.update_state(batch, model.state)
    assert jnp.allclose(
        loss.loss_train_batched(Position({}), carry),
        -batches.batch_sample_scale * pointwise_sum(batch_state, "y_log_prob"),
    ), "batch loss"
    assert jnp.allclose(
        scaled_loss.loss_train_batched(Position({}), carry),
        -batches.batch_sample_scale
        * pointwise_sum(batch_state, "y_log_prob")
        / split.train_sample_size,
    ), "scaled batch loss"


def test_split_and_batch_scalar():
    model = simple_model(12)
    split = opt.PositionSplit.from_model(
        model, ["y"], validate_axis_share=0.25, shuffle=False
    )
    batches = opt.Batches(
        ["y"],
        axis_size=split.train_axis_size,
        batch_size=3,
        shuffle=False,
        sample_size=split.train_sample_size,
    )
    batch = batches.get_batched_position(split.train, 0)
    loss = opt.NegLogProbLoss(model, split, scale=False)
    scaled_loss = opt.NegLogProbLoss(model, split, scale=True)
    carry = empty_carry(model, batch=batch, batches=batches)
    batch_state = model.update_state(batch, model.state)
    validate_state = model.update_state(split.validate, model.state)
    assert jnp.allclose(
        loss.loss_train_batched(Position({}), carry),
        -batches.batch_sample_scale * pointwise_sum(batch_state, "y_log_prob"),
    ), "batch loss"
    assert jnp.allclose(
        loss.loss_monitor(Position({}), empty_carry(model)),
        -split.validate_sample_scale * pointwise_sum(validate_state, "y_log_prob"),
    ), "validation loss"
    assert jnp.allclose(
        scaled_loss.loss_train_batched(Position({}), carry),
        -batches.batch_sample_scale
        * pointwise_sum(batch_state, "y_log_prob")
        / split.train_sample_size,
    ), "scaled batch loss"


def test_split_and_batch_different_axes():
    model = split_batch_array_model()
    split = opt.PositionSplit.from_model(
        model, ["X_split", "y"], validate_axis_share=1 / 3, shuffle=False
    )
    Z_batch_full = model.extract_position(["Z_batch"])["Z_batch"]
    batches = opt.Batches(
        ["Z_batch", "y"],
        axis_size=8,
        batch_size=2,
        batch_axes={"Z_batch": 1, "y": 1},
        shuffle=False,
        sample_size=split.train_sample_size,
    )
    train_position_for_batching = Position(
        {
            "X_split": split.train["X_split"],
            "Z_batch": Z_batch_full,
            "y": split.train["y"],
        }
    )
    batch = batches.get_batched_position(train_position_for_batching, 0)
    fixed_train = Position({"X_split": split.train["X_split"]})
    loss = opt.NegLogProbLoss(model, split, scale=False)
    scaled_loss = opt.NegLogProbLoss(model, split, scale=True)
    carry = empty_carry(model, batch=batch, batches=batches, fixed_position=fixed_train)
    batch_state = model.update_state(Position(batch | fixed_train), model.state)
    validate_state = model.update_state(
        Position(split.validate | {"Z_batch": Z_batch_full}), model.state
    )
    assert jnp.allclose(
        loss.loss_train_batched(Position({}), carry),
        -batches.batch_sample_scale * pointwise_sum(batch_state, "y_log_prob"),
    ), "batch loss"
    assert jnp.allclose(
        loss.loss_monitor(
            Position({}),
            empty_carry(model, fixed_position=Position({"Z_batch": Z_batch_full})),
        ),
        -split.validate_sample_scale * pointwise_sum(validate_state, "y_log_prob"),
    ), "validation loss"
    assert jnp.allclose(
        scaled_loss.loss_train_batched(Position({}), carry),
        -batches.batch_sample_scale
        * pointwise_sum(batch_state, "y_log_prob")
        / split.train_sample_size,
    ), "scaled batch loss"
