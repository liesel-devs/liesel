from __future__ import annotations

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.experimental.optim as opt
import liesel.model as lsl
from liesel.experimental.optim import (
    Batches,
    BatchManager,
    LieselOptim,
    NegLogProbLoss,
    OptimEngine,
    PositionSplit,
    PositionSplitManager,
    Stopper,
)
from liesel.experimental.optim.state import OptimResult


def _normal_model(n: int = 6):
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    y = lsl.Var.new_obs(
        jnp.arange(float(n)),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y",
    )
    return lsl.Model([y])


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
    from liesel.experimental.optim.quick import LieselOptim as LieselOptimFromQuick

    assert opt.LieselOptim is LieselOptim
    assert LieselOptimFromQuick is LieselOptim
    assert not hasattr(opt, "QuickOptim")


def test_default_build_engine_uses_opinionated_defaults():
    model = _normal_model()

    engine = LieselOptim(model, seed=1).build_engine()

    assert isinstance(engine, OptimEngine)
    assert isinstance(engine.loss, NegLogProbLoss)
    assert isinstance(engine.batches, Batches)
    assert engine.batches.is_full_data
    assert engine.batches.n == engine.split.n_train
    assert engine.optimizers[0].position_keys == tuple(model.parameters)
    assert engine.stopper == Stopper(epochs=1000, patience=10, rtol=1e-6)
    assert engine.train_monitor == "auto"


def test_batch_size_shortcut_builds_training_batches():
    model = _normal_model()
    split = PositionSplit.from_model(model, share_validate=0.25)

    engine = LieselOptim(model, split=split, batch_size=2, seed=1).build_engine()

    assert isinstance(engine.batches, Batches)
    assert engine.batches.n == split.n_train
    assert engine.batches.batch_size == 2


def test_batches_and_batch_size_are_mutually_exclusive():
    model = _normal_model()
    batches = Batches(["y"], n=6, batch_size=None)

    with pytest.raises(ValueError, match="batches or batch_size"):
        LieselOptim(model, batches=batches, batch_size=2)


def test_user_provided_batches_are_not_mutated():
    model = _normal_model()
    batches = Batches(["y"], n=2, batch_size=None)

    quick = LieselOptim(model, batches=batches, seed=1)
    engine = quick.build_engine()

    assert quick.batches is batches
    assert engine.batches is batches
    assert batches.n == 2


def test_multi_size_default_split_builds_batch_manager():
    model = _two_branch_model()

    engine = LieselOptim(model, batch_size=None, seed=1).build_engine()

    assert isinstance(engine.split, PositionSplitManager)
    assert isinstance(engine.batches, BatchManager)
    assert engine.batches.n == engine.split.n_trains


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


def test_train_monitor_is_passed_to_engine():
    model = _normal_model()

    engine = LieselOptim(
        model, train_monitor="weighted_epoch_average", seed=1
    ).build_engine()

    assert engine.train_monitor == "weighted_epoch_average"


def test_fit_returns_optim_result():
    model = _normal_model()

    result = LieselOptim(
        model,
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
    ).fit()

    assert isinstance(result, OptimResult)
