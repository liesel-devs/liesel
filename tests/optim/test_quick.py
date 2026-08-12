from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
from liesel.optim import (
    Batches,
    BatchManager,
    LieselOptim,
    NegLogProbLoss,
    OptimEngine,
    PositionSplit,
    PositionSplitManager,
    Stopper,
)
from liesel.optim.state import OptimResult


def _normal_model(n: int = 6, *, to_float32: bool | None = None):
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    y = lsl.Var.new_obs(
        jnp.arange(float(n)),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y",
    )
    if to_float32 is None:
        return lsl.Model([y])

    return lsl.Model([y], to_float32=to_float32)


def _two_branch_model():
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    y1 = lsl.Var.new_obs(
        jnp.arange(8.0),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y1",
    )
    y2 = lsl.Var.new_obs(
        jnp.arange(5.0),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y2",
    )
    return lsl.Model([y1, y2])


def test_lieseloptim_imports():
    from liesel.optim.liesel_optim import (
        LieselOptim as LieselOptimFromQuick,
    )

    assert opt.LieselOptim is LieselOptim
    assert LieselOptimFromQuick is LieselOptim
    assert not hasattr(opt, "QuickOptim")


def test_default_build_engine_uses_opinionated_defaults():
    model = _normal_model()

    engine = LieselOptim(model, seed=1).build_engine()

    assert isinstance(engine, OptimEngine)
    assert isinstance(engine.loss, NegLogProbLoss)
    assert engine.loss.scale is True
    assert isinstance(engine.batches, Batches)
    assert engine.batches.is_full_data
    assert engine.batches.axis_size == engine.split.train_axis_size
    assert engine.optimizers[0].position_keys == tuple(model.parameters)
    assert engine.stopper == Stopper(epochs=1000, patience=10, rtol=1e-6)
    assert engine.train_monitor == "auto"
    assert engine.progress_update_every == 10
    assert engine.show_step_progress is False
    assert engine.step_progress_update_every == 10


def test_batch_size_shortcut_builds_training_batches():
    model = _normal_model()
    split = PositionSplit.from_model(model, validate_axis_share=0.25)

    engine = LieselOptim(model, split=split, batch_size=2, seed=1).build_engine()

    assert isinstance(engine.batches, Batches)
    assert engine.batches.axis_size == split.train_axis_size
    assert engine.batches.batch_size == 2


def test_old_batch_axis_size_shortcut_still_works():
    model = _normal_model()

    engine = LieselOptim(model, batch_axis_size=2, seed=1).build_engine()

    assert isinstance(engine.batches, Batches)
    assert engine.batches.batch_size == 2


def test_batches_and_batch_size_are_mutually_exclusive():
    model = _normal_model()
    batches = Batches(["y"], axis_size=6, batch_size=None)

    with pytest.raises(ValueError, match="batches or batch_size"):
        LieselOptim(model, batches=batches, batch_size=2)


def test_batch_size_and_old_keyword_are_mutually_exclusive():
    model = _normal_model()

    with pytest.raises(ValueError, match="batch_size or batch_axis_size"):
        LieselOptim(model, batch_size=2, batch_axis_size=2)


def test_user_provided_batches_are_not_mutated():
    model = _normal_model()
    batches = Batches(["y"], axis_size=2, batch_size=None)

    quick = LieselOptim(model, batches=batches, seed=1)
    engine = quick.build_engine()

    assert quick.batches is batches
    assert engine.batches is batches
    assert batches.axis_size == 2


def test_multi_size_default_split_builds_batch_manager():
    model = _two_branch_model()

    engine = LieselOptim(model, batch_size=None, seed=1).build_engine()

    assert isinstance(engine.split, PositionSplitManager)
    assert isinstance(engine.batches, BatchManager)
    assert engine.batches.axis_size == engine.split.train_axis_sizes
    assert isinstance(engine.loss, NegLogProbLoss)
    assert engine.loss.scale is True
    assert engine.loss.scalar == sum(engine.split.train_axis_sizes)


def test_scale_loss_false_builds_unscaled_default_loss():
    model = _normal_model()

    engine = LieselOptim(model, scale_loss=False, seed=1).build_engine()

    assert isinstance(engine.loss, NegLogProbLoss)
    assert engine.loss.scale is False
    assert engine.loss.scalar == 1.0


def test_scale_loss_is_ignored_for_custom_loss():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    loss = NegLogProbLoss(model, split, scale=False)

    engine = LieselOptim(model, loss=loss, scale_loss=True, seed=1).build_engine()

    assert engine.loss is loss
    assert loss.scale is False


def test_custom_loss_and_conflicting_split_raise():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    other_split = PositionSplit.from_model(model)
    loss = NegLogProbLoss(model, split)

    with pytest.raises(ValueError, match="loss.split"):
        LieselOptim(model, loss=loss, split=other_split)


def test_unknown_optimizer_string_raises():
    model = _normal_model()

    with pytest.raises(ValueError, match="optimizers"):
        LieselOptim(model, optimizers="sgd")


def test_progress_and_train_monitor_are_passed_to_engine():
    model = _normal_model()

    engine = LieselOptim(
        model,
        batch_size=1,
        train_monitor="weighted_epoch_average",
        show_progress=False,
        progress_n_updates=7,
        progress_update_every=3,
        show_step_progress=True,
        step_progress_update_every=4,
        step_progress_n_updates=4,
        seed=1,
    ).build_engine()

    assert engine.train_monitor == "weighted_epoch_average"
    assert engine.show_progress is False
    assert engine.progress_n_updates == 7
    assert engine.progress_update_every == 143
    assert engine.show_step_progress is True
    assert engine.step_progress_update_every == 2
    assert engine.step_progress_n_updates == 3


def test_fit_returns_optim_result():
    model = _normal_model()

    result = LieselOptim(
        model,
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
    ).fit()

    assert isinstance(result, OptimResult)


def test_fit_handles_float32_model_with_x64_enabled():
    with jax.enable_x64(True):
        model = _normal_model(to_float32=True)
        result = LieselOptim(
            model,
            stopper=Stopper(epochs=1, patience=1),
            seed=1,
        ).fit()

    assert isinstance(result, OptimResult)
    assert result.history.loss_train.dtype == jnp.float32


def test_batched_fit_handles_float32_model_with_x64_enabled():
    with jax.enable_x64(True):
        model = _normal_model(to_float32=True)
        result = LieselOptim(
            model,
            batch_size=2,
            stopper=Stopper(epochs=1, patience=1),
            seed=1,
        ).fit()

    assert isinstance(result, OptimResult)
    assert result.history.loss_train.dtype == jnp.float32


def test_fit_can_split_response_and_batch_shared_covariate_on_different_axes():
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    response = lsl.Var.new_obs(
        jnp.arange(24.0).reshape(4, 6),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="response",
    )
    land = lsl.Var.new_obs(jnp.arange(6.0).reshape(6, 1), name="land")
    model = lsl.Model([response, land])
    split = PositionSplit.from_model(
        model,
        position_keys=["response", "land"],
        axis_size=4,
        validate_axis_share=0.25,
        split_axes={"response": 0, "land": None},
    )
    batches = Batches.from_model(
        model,
        batch_size=3,
        position_keys=["response", "land"],
        axis_size=6,
        batch_axes={"response": 1, "land": 0},
        shuffle=False,
    )

    result = LieselOptim(
        model,
        split=split,
        batches=batches,
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        show_progress=False,
    ).fit()

    assert jnp.isfinite(result.history.loss_train[0])


def test_fit_handles_float64_model_with_x64_enabled():
    with jax.enable_x64(True):
        loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
        y = lsl.Var.new_obs(
            jnp.arange(6.0),
            lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
            name="y",
        )
        model = lsl.Model([y], to_float32=False)
        result = LieselOptim(
            model,
            stopper=Stopper(epochs=1, patience=1),
            seed=1,
        ).fit()

    assert isinstance(result, OptimResult)
    assert result.history.loss_train.dtype == jnp.float64
