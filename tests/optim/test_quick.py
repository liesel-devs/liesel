from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
import liesel.optim.split as split_module
from liesel.optim import (
    Batches,
    BatchManager,
    EmaTrainLossMonitor,
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


@pytest.mark.parametrize("make_model", [_normal_model, _two_branch_model])
def test_seeded_automatic_full_data_setup_preserves_rows_and_repeats_fit(
    make_model, monkeypatch
):
    def unexpected_clock_read():
        raise AssertionError("Automatic full-data setup must not generate a split seed")

    monkeypatch.setattr(
        split_module, "time", SimpleNamespace(time=unexpected_clock_read)
    )

    def run():
        model = make_model()
        quick = LieselOptim(
            model,
            seed=42,
            batches=Batches.from_model(model, batch_size=2, multi_size="manager"),
            loss_monitor=EmaTrainLossMonitor(1),
            stopper=Stopper(epochs=3, patience=3),
            show_progress=False,
        )
        original = model.extract_position(quick.split.position_keys)
        for key, value in original.items():
            assert jnp.array_equal(quick.split.train[key], value)
        return quick.fit()

    first, second = run(), run()
    assert jnp.array_equal(first.history.loss_train, second.history.loss_train)
    assert jnp.array_equal(first.position_final["loc"], second.position_final["loc"])


def test_lieseloptim_requires_explicit_loss_monitor():
    with pytest.raises(TypeError, match="loss_monitor"):
        LieselOptim(_normal_model())  # ty: ignore[missing-argument]


@pytest.mark.parametrize("optimizers", ["lbfgs", [opt.LBFGS(["loc"])]])
def test_lbfgs_rejects_minibatches_during_wrapper_construction(optimizers):
    with pytest.raises(ValueError, match="LBFGS.*full-data.*deterministic"):
        LieselOptim(
            _normal_model(),
            optimizers=optimizers,
            batch_size=2,
            loss_monitor="train_full_data",
        )


@pytest.mark.parametrize(
    "seed", [1, np.int64(1), jax.random.key(1), jax.random.PRNGKey(1)]
)
def test_engine_accepts_integer_and_jax_seeds(seed):
    engine = LieselOptim(
        _normal_model(),
        loss_monitor="train_full_data",
        stopper=Stopper(epochs=2, patience=2),
        show_progress=False,
    ).build_engine()
    # Exercise the low-level constructor as well as a real JAX execution.
    engine = OptimEngine(
        loss=engine.loss,
        batches=engine.batches,
        optimizers=engine.optimizers,
        stopper=engine.stopper,
        seed=seed,
        initial_state=engine.initial_state,
        loss_monitor="train_full_data",
        show_progress=False,
    )
    assert jnp.array_equal(
        jax.random.key_data(engine.seed), jax.random.key_data(jax.random.key(1))
    )
    assert engine.fit().status == "max_epochs"


@pytest.mark.parametrize("kind", ["per_obs", "custom_log_lik"])
def test_automatic_inference_error_explains_manual_split(kind):
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    dist = lsl.Dist(tfd.Normal, loc=loc, scale=1.0)
    dist.per_obs = kind != "per_obs"
    y = lsl.Var.new_obs(jnp.arange(4.0), dist, name="y")
    builder = lsl.GraphBuilder().add(y)
    if kind == "custom_log_lik":
        builder.log_lik_node = lsl.Calc(
            lambda value: value.sum(), dist, _name="custom_log_lik"
        )
    model = builder.build_model()
    with pytest.raises(ValueError, match=r"PositionSplit.from_model.*split=split"):
        LieselOptim(model, loss_monitor="train_full_data", scale_loss=False)
    for split in (
        PositionSplit.from_model(model, infer_sample_sizes=False),
        PositionSplit.from_model(model, sample_sizes={"train": 4}),
    ):
        result = LieselOptim(
            model,
            split=split,
            optimizers="lbfgs",
            loss_monitor="train_full_data",
            seed=1,
            show_progress=False,
        ).fit()
        assert float(result.position_final["loc"]) == pytest.approx(1.5, abs=1e-5)


@pytest.mark.parametrize(
    "keyword",
    [
        "batch_mode",
        "axis_size",
        "split_axes",
        "default_split_axis",
        "shuffle_batches",
        "epoch_size",
        "batch_axis_size",
        "progress_n_updates",
        "step_progress_n_updates",
    ],
)
def test_lieseloptim_removed_data_shortcuts_are_rejected(keyword):
    with pytest.raises(TypeError, match=keyword):
        LieselOptim(
            _normal_model(),
            loss_monitor="train_full_data",
            **{keyword: None},  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(("epoch_size", "expected"), [("max", 4), ("min", 2), (3, 3)])
def test_lieseloptim_preserves_explicit_multi_branch_batches(epoch_size, expected):
    model = _two_branch_model()
    split = PositionSplit.from_model(model, multi_size="manager")
    batches = Batches.from_split(split, batch_size=2, epoch_size=epoch_size)
    optimizer = LieselOptim(
        model,
        loss_monitor="train_full_data",
        split=split,
        batches=batches,
    )
    assert optimizer.batches is batches
    assert isinstance(optimizer.batches, BatchManager)
    assert optimizer.batches.n_full_batches == expected


def test_lieseloptim_validation_monitor_requires_validation_data():
    with pytest.raises(ValueError, match="validation"):
        LieselOptim(_normal_model(), loss_monitor="validation")


def test_lieseloptim_rejects_unknown_loss_monitor():
    with pytest.raises(ValueError, match="loss_monitor"):
        LieselOptim(
            _normal_model(),
            loss_monitor="sometimes",  # ty: ignore[invalid-argument-type]
        )


def test_default_build_engine_uses_opinionated_defaults():
    model = _normal_model()
    loss_monitor = EmaTrainLossMonitor(effective_window=1.0)

    engine = LieselOptim(model, loss_monitor=loss_monitor, seed=1).build_engine()

    assert isinstance(engine, OptimEngine)
    assert isinstance(engine.loss, NegLogProbLoss)
    assert engine.loss.scale is True
    assert isinstance(engine.batches, Batches)
    assert engine.batches.is_full_data
    assert engine.batches.axis_size == engine.split.train_axis_size
    assert engine.optimizers[0].position_keys == tuple(model.parameters)
    assert engine.stopper == Stopper(epochs=1000, patience=10, rtol=1e-6)
    assert engine.loss_monitor is loss_monitor
    assert engine.progress_update_every == 10
    assert engine.show_step_progress is False
    assert engine.step_progress_update_every == 10


@pytest.mark.parametrize("batch_size", [None, 20])
@pytest.mark.parametrize("spread", [0.0, 0.5])
def test_default_adam_fits_normal_mean_within_default_budget(batch_size, spread):
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    y = lsl.Var.new_obs(
        5.13 + jnp.linspace(-spread, spread, 200),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y",
    )
    result = LieselOptim(
        lsl.Model([y]),
        batch_size=batch_size,
        loss_monitor="train_full_data",
        seed=0,
        show_progress=False,
    ).fit()
    tolerance = 0.05 if spread and batch_size else 0.01
    assert float(result.position_final["loc"]) == pytest.approx(5.13, abs=tolerance)


def test_default_stopper_is_independent_between_instances():
    model = _normal_model()
    first = LieselOptim(model, loss_monitor="train_full_data", seed=1).build_engine()
    second = LieselOptim(model, loss_monitor="train_full_data", seed=1).build_engine()

    first.stopper.epochs = 50
    assert second.stopper.epochs == 1000


def test_explicit_batches_use_training_split():
    model = _normal_model()
    split = PositionSplit.from_model(model, validate_axis_share=0.25)

    engine = LieselOptim(
        model,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        split=split,
        batches=Batches.from_split(split, batch_size=2),
        seed=1,
    ).build_engine()

    assert isinstance(engine.batches, Batches)
    assert engine.batches.axis_size == split.train_axis_size
    assert engine.batches.batch_size == 2


@pytest.mark.parametrize("make_model", [_normal_model, _two_branch_model])
@pytest.mark.parametrize("batch_size", [None, 2])
def test_batch_size_shortcut_uses_training_split_defaults(make_model, batch_size):
    model = make_model()
    split = PositionSplit.from_model(
        model, validate_axis_share=0.25, seed=42, multi_size="manager"
    )
    expected = Batches.from_split(split, batch_size=batch_size)
    engine = LieselOptim(
        model,
        split=split,
        batch_size=batch_size,
        loss_monitor="validation",
        seed=1,
    ).build_engine()

    assert type(engine.batches) is type(expected)
    assert engine.batches.axis_size == expected.axis_size
    assert engine.batches.batch_size == expected.batch_size
    assert engine.batches.batch_sample_scales == expected.batch_sample_scales
    assert engine.batches.n_full_batches == expected.n_full_batches
    actual_children = (
        engine.batches.batches
        if isinstance(engine.batches, BatchManager)
        else (engine.batches,)
    )
    assert all(child.shuffle == (batch_size is not None) for child in actual_children)


def test_batch_size_cannot_be_combined_with_explicit_batches():
    with pytest.raises(ValueError, match="either batch_size or batches"):
        LieselOptim(
            _normal_model(),
            batch_size=2,
            batches=Batches(["y"], axis_size=6, batch_size=3),
            loss_monitor="train_full_data",
        )


def test_user_provided_batches_are_not_mutated():
    model = _normal_model()
    batches = Batches(["y"], axis_size=6, batch_size=None)

    quick = LieselOptim(
        model,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=batches,
        seed=1,
    )
    engine = quick.build_engine()

    assert quick.batches is batches
    assert engine.batches is batches
    assert batches.axis_size == 6


def test_multi_size_default_split_builds_batch_manager():
    model = _two_branch_model()

    engine = LieselOptim(
        model,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        seed=1,
    ).build_engine()

    assert isinstance(engine.split, PositionSplitManager)
    assert isinstance(engine.batches, BatchManager)
    assert engine.batches.axis_size == engine.split.train_axis_sizes
    assert isinstance(engine.loss, NegLogProbLoss)
    assert engine.loss.scale is True
    assert engine.loss.scalar == sum(engine.split.train_axis_sizes)


def test_scale_loss_false_builds_unscaled_default_loss():
    model = _normal_model()

    engine = LieselOptim(
        model,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        scale_loss=False,
        seed=1,
    ).build_engine()

    assert isinstance(engine.loss, NegLogProbLoss)
    assert engine.loss.scale is False
    assert engine.loss.scalar == 1.0


def test_scale_loss_is_ignored_for_custom_loss():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    loss = NegLogProbLoss(model, split, scale=False)

    engine = LieselOptim(
        model,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        loss=loss,
        scale_loss=True,
        seed=1,
    ).build_engine()

    assert engine.loss is loss
    assert loss.scale is False


def test_custom_loss_and_conflicting_split_raise():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    other_split = PositionSplit.from_model(model)
    loss = NegLogProbLoss(model, split)

    with pytest.raises(ValueError, match="loss.split"):
        LieselOptim(
            model,
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            loss=loss,
            split=other_split,
        )


def test_unknown_optimizer_string_raises():
    model = _normal_model()

    with pytest.raises(ValueError, match="optimizers"):
        LieselOptim(
            model,
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            optimizers="sgd",  # ty: ignore[invalid-argument-type]
        )


def test_progress_and_loss_monitor_are_passed_to_engine():
    model = _normal_model()
    loss_monitor = EmaTrainLossMonitor(0.5)

    engine = LieselOptim(
        model,
        batches=Batches.from_model(model, batch_size=1),
        loss_monitor=loss_monitor,
        show_progress=False,
        progress_update_every=3,
        show_step_progress=True,
        step_progress_update_every=4,
        seed=1,
    ).build_engine()

    assert engine.loss_monitor is loss_monitor
    assert engine.show_progress is False
    assert engine.progress_update_every == 3
    assert engine.show_step_progress is True
    assert engine.step_progress_update_every == 4


def test_fit_returns_optim_result():
    model = _normal_model()

    result = LieselOptim(
        model,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
    ).fit()

    assert isinstance(result, OptimResult)
    assert result.monitor_source == "train_ema"
    assert result.n_epochs == 1
    assert result.position_final.keys() == model.parameters.keys()
    assert result.position_min_monitor is not None
    assert result.position_min_monitor.keys() == result.position_final.keys()


def test_fit_handles_float32_model_with_x64_enabled():
    with jax.enable_x64(True):
        model = _normal_model(to_float32=True)
        result = LieselOptim(
            model,
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            batches=Batches.from_model(model, batch_size=2),
            stopper=Stopper(epochs=1, patience=1),
            seed=1,
        ).fit()

    assert isinstance(result, OptimResult)
    assert result.history.loss_train.dtype == jnp.float32


@pytest.mark.parametrize("make_model", [_normal_model, _two_branch_model])
@pytest.mark.parametrize("batch_size", [None, 2])
def test_batches_from_unsplit_model_are_rejected_before_fitting(make_model, batch_size):
    model = make_model()
    split = PositionSplit.from_model(
        model, validate_axis_share=0.25, seed=42, multi_size="manager"
    )
    batches = Batches.from_model(model, batch_size=batch_size, multi_size="manager")
    optim = LieselOptim(
        model, split=split, batches=batches, loss_monitor="validation", seed=1
    )

    with pytest.raises(ValueError, match=r"batch axis.*axis_size.*Batches\.from_split"):
        optim.build_engine()


@pytest.mark.parametrize("batch_axis", [1, -1])
@pytest.mark.parametrize(
    "factory", [Batches.from_model, Batches.from_split, BatchManager.from_split]
)
def test_fit_can_split_response_and_batch_shared_covariate_on_different_axes(
    batch_axis,
    factory,
):
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
    batches = factory(
        model if factory == Batches.from_model else split,
        batch_size=3,
        position_keys=["response", "land"],
        axis_size=6,
        batch_axes={"response": batch_axis, "land": 0},
        shuffle=False,
    )

    result = LieselOptim(
        model,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        split=split,
        batches=batches,
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        show_progress=False,
    ).fit()

    assert jnp.isfinite(result.history.loss_train[0])


def test_batch_manager_validates_later_groups_too():
    model = _two_branch_model()
    split = PositionSplitManager.from_model(
        model, position_keys=[["y1"], ["y2"]], validate_axis_share=0.25, seed=42
    )
    batches = Batches.from_split(split, batch_size=2)
    assert isinstance(batches, BatchManager)
    batches = BatchManager(
        [
            batches.batches[0],
            Batches.from_model(model, batch_size=2, position_keys=["y2"]),
        ],
        epoch_size="max",
    )

    with pytest.raises(ValueError, match="y2.*batch axis.*axis_size"):
        LieselOptim(
            model, split=split, batches=batches, loss_monitor="validation", seed=1
        ).build_engine()


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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            stopper=Stopper(epochs=1, patience=1),
            seed=1,
        ).fit()

    assert isinstance(result, OptimResult)
    assert result.history.loss_train.dtype == jnp.float64
