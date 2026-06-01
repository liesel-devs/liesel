from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.experimental.optim as opt
import liesel.model as lsl
from liesel.experimental.optim import (
    LBFGS,
    Batches,
    BatchManager,
    LieselVI,
    NegElboLoss,
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


def test_lieselvi_imports():
    from liesel.experimental.optim.liesel_vi import LieselVI as LieselVIFromModule

    assert opt.LieselVI is LieselVI
    assert LieselVIFromModule is LieselVI


def test_default_build_engine_uses_opinionated_defaults():
    model = _normal_model()

    engine = LieselVI(model, seed=1).build_engine()

    assert isinstance(engine, OptimEngine)
    assert isinstance(engine.loss, NegElboLoss)
    assert engine.loss.scale is True
    assert engine.loss.scalar == engine.split.n_train
    assert engine.loss.nsamples == 10
    assert engine.loss.nsamples_validate == 50
    assert engine.loss.vdist.var.dist_node.distribution is tfd.MultivariateNormalDiag
    assert isinstance(engine.batches, Batches)
    assert engine.batches.is_full_data
    assert engine.batches.n == engine.split.n_train
    assert engine.optimizers[0].position_keys == tuple(engine.loss.q.parameters)
    assert engine.stopper == Stopper(epochs=1000, patience=10, rtol=1e-6)
    assert engine.train_monitor == "auto"


def test_batch_size_shortcut_builds_training_batches():
    model = _normal_model()
    split = PositionSplit.from_model(model, share_validate=0.25)

    engine = LieselVI(model, split=split, batch_size=2, seed=1).build_engine()

    assert isinstance(engine.batches, Batches)
    assert engine.batches.n == split.n_train
    assert engine.batches.batch_size == 2


def test_batches_and_batch_size_are_mutually_exclusive():
    model = _normal_model()
    batches = Batches(["y"], n=6, batch_size=None)

    with pytest.raises(ValueError, match="batches or batch_size"):
        LieselVI(model, batches=batches, batch_size=2)


def test_user_provided_batches_are_not_mutated():
    model = _normal_model()
    batches = Batches(["y"], n=2, batch_size=None)

    vi = LieselVI(model, batches=batches, seed=1)
    engine = vi.build_engine()

    assert vi.batches is batches
    assert engine.batches is batches
    assert batches.n == 2


def test_multi_size_default_split_builds_batch_manager():
    model = _two_branch_model()

    engine = LieselVI(model, batch_size=None, seed=1).build_engine()

    assert isinstance(engine.split, PositionSplitManager)
    assert isinstance(engine.batches, BatchManager)
    assert engine.batches.n == engine.split.n_trains
    assert isinstance(engine.loss, NegElboLoss)
    assert engine.loss.scale is True
    assert engine.loss.scalar == sum(engine.split.n_trains)


def test_scale_loss_false_builds_unscaled_default_loss():
    model = _normal_model()

    engine = LieselVI(model, scale_loss=False, seed=1).build_engine()

    assert isinstance(engine.loss, NegElboLoss)
    assert engine.loss.scale is False
    assert engine.loss.scalar == 1.0


def test_custom_loss_and_conflicting_split_raise():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    other_split = PositionSplit.from_model(model)
    loss = NegElboLoss.mvn_diag(model, split=split)

    with pytest.raises(ValueError, match="loss.split"):
        LieselVI(model, loss=loss, split=other_split)


def test_custom_loss_is_passed_through_unchanged():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    loss = NegElboLoss.mvn_diag(model, split=split, nsamples=3, scale=False)

    engine = LieselVI(
        model, loss=loss, nsamples=99, scale_loss=True, seed=1
    ).build_engine()

    assert engine.loss is loss
    assert loss.nsamples == 3
    assert loss.scale is False


def test_unknown_loss_string_raises():
    model = _normal_model()

    with pytest.raises(ValueError, match="loss"):
        LieselVI(model, loss="mean_field")  # type: ignore[arg-type]


def test_lbfgs_string_shortcut_raises_with_vi_specific_message():
    model = _normal_model()

    with pytest.raises(ValueError, match="ELBO.*stochastic"):
        LieselVI(model, optimizers="lbfgs")  # type: ignore[arg-type]


def test_explicit_lbfgs_optimizer_sequence_is_accepted():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    loss = NegElboLoss.mvn_diag(model, split=split)
    optimizer = LBFGS(list(loss.q.parameters))

    engine = LieselVI(model, loss=loss, optimizers=[optimizer], seed=1).build_engine()

    assert engine.optimizers[0] is optimizer


def test_train_monitor_is_passed_to_engine():
    model = _normal_model()

    engine = LieselVI(
        model, train_monitor="weighted_epoch_average", seed=1
    ).build_engine()

    assert engine.train_monitor == "weighted_epoch_average"


def test_fit_returns_optim_result():
    model = _normal_model()

    result = LieselVI(
        model,
        stopper=Stopper(epochs=1, patience=1),
        nsamples=1,
        nsamples_validate=1,
        seed=1,
    ).fit()

    assert isinstance(result, OptimResult)


def test_fit_handles_float64_model_with_x64_enabled():
    with jax.enable_x64(True):
        loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
        y = lsl.Var.new_obs(
            jnp.arange(6.0),
            lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
            name="y",
        )
        model = lsl.Model([y], to_float32=False)

        result = LieselVI(
            model,
            stopper=Stopper(epochs=1, patience=1),
            nsamples=1,
            nsamples_validate=1,
            seed=1,
        ).fit()

    assert isinstance(result, OptimResult)
    assert result.history.loss_train.dtype == jnp.float64
