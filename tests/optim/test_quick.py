from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
import liesel.optim.liesel_optim as quick_module
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
@pytest.mark.parametrize("seed_kwargs", [{}, {"seed": 42}])
def test_seeded_automatic_full_data_setup_preserves_rows_and_repeats_fit(
    make_model, seed_kwargs, monkeypatch
):
    def unexpected_clock_read():
        raise AssertionError("Automatic full-data setup must not generate a split seed")

    monkeypatch.setattr(
        split_module, "time", SimpleNamespace(time=unexpected_clock_read)
    )
    monkeypatch.setattr(
        quick_module, "time", SimpleNamespace(time=unexpected_clock_read)
    )

    def run():
        model = make_model()
        quick = LieselOptim(
            model,
            optimizers=optax.adam(0.02),
            **seed_kwargs,
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


def test_explicit_none_seed_uses_clock(monkeypatch):
    monkeypatch.setattr(quick_module, "time", SimpleNamespace(time=lambda: 1234.5))
    optim = LieselOptim(
        _normal_model(),
        optimizers=optax.adam(0.02),
        loss_monitor="train_full_data",
        seed=None,
    )
    assert optim.seed == 1234


def test_lieseloptim_requires_explicit_loss_monitor():
    with pytest.raises(TypeError, match="loss_monitor"):
        LieselOptim(_normal_model(), optimizers=optax.adam(0.02))  # ty: ignore[missing-argument]


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
        optimizers=optax.adam(0.02),
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
        LieselOptim(
            model,
            optimizers=optax.adam(0.02),
            loss_monitor="train_full_data",
            scale_loss=False,
        )
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
            optimizers=optax.adam(0.02),
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
        optimizers=optax.adam(0.02),
        loss_monitor="train_full_data",
        split=split,
        batches=batches,
    )
    assert optimizer.batches is batches
    assert isinstance(optimizer.batches, BatchManager)
    assert optimizer.batches.n_full_batches == expected


def test_lieseloptim_validation_monitor_requires_validation_data():
    with pytest.raises(ValueError, match="validation"):
        LieselOptim(
            _normal_model(), optimizers=optax.adam(0.02), loss_monitor="validation"
        )


def test_lieseloptim_rejects_unknown_loss_monitor():
    with pytest.raises(ValueError, match="loss_monitor"):
        LieselOptim(
            _normal_model(),
            optimizers=optax.adam(0.02),
            loss_monitor="sometimes",  # ty: ignore[invalid-argument-type]
        )


def test_default_build_engine_uses_opinionated_defaults():
    model = _normal_model()
    loss_monitor = EmaTrainLossMonitor(effective_window=1.0)

    engine = LieselOptim(
        model, optimizers=optax.adam(0.02), loss_monitor=loss_monitor, seed=1
    ).build_engine()

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
def test_explicit_adam_fits_normal_mean_within_default_budget(batch_size, spread):
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    y = lsl.Var.new_obs(
        5.13 + jnp.linspace(-spread, spread, 200),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y",
    )
    result = LieselOptim(
        lsl.Model([y]),
        optimizers=optax.adam(0.02),
        batch_size=batch_size,
        loss_monitor="train_full_data",
        seed=0,
        show_progress=False,
    ).fit()
    tolerance = 0.05 if spread and batch_size else 0.01
    assert float(result.position_final["loc"]) == pytest.approx(5.13, abs=tolerance)


def test_default_stopper_is_independent_between_instances():
    model = _normal_model()
    first = LieselOptim(
        model, optimizers=optax.adam(0.02), loss_monitor="train_full_data", seed=1
    ).build_engine()
    second = LieselOptim(
        model, optimizers=optax.adam(0.02), loss_monitor="train_full_data", seed=1
    ).build_engine()

    first.stopper.epochs = 50
    assert second.stopper.epochs == 1000


def test_explicit_batches_use_training_split():
    model = _normal_model()
    split = PositionSplit.from_model(model, validate_axis_share=0.25)

    engine = LieselOptim(
        model,
        optimizers=optax.adam(0.02),
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
        optimizers=optax.adam(0.02),
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
            optimizers=optax.adam(0.02),
            batch_size=2,
            batches=Batches(["y"], axis_size=6, batch_size=3),
            loss_monitor="train_full_data",
        )


def test_user_provided_batches_are_not_mutated():
    model = _normal_model()
    batches = Batches(["y"], axis_size=6, batch_size=None)

    quick = LieselOptim(
        model,
        optimizers=optax.adam(0.02),
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
        optimizers=optax.adam(0.02),
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
        optimizers=optax.adam(0.02),
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
        optimizers=optax.adam(0.02),
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
            optimizers=optax.adam(0.02),
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
        optimizers=optax.adam(0.02),
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
        optimizers=optax.adam(0.02),
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
            optimizers=optax.adam(0.02),
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
            optimizers=optax.adam(0.02),
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
        model,
        optimizers=optax.adam(0.02),
        split=split,
        batches=batches,
        loss_monitor="validation",
        seed=1,
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
        optimizers=optax.adam(0.02),
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
            model,
            optimizers=optax.adam(0.02),
            split=split,
            batches=batches,
            loss_monitor="validation",
            seed=1,
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
            optimizers=optax.adam(0.02),
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            stopper=Stopper(epochs=1, patience=1),
            seed=1,
        ).fit()

    assert isinstance(result, OptimResult)
    assert result.history.loss_train.dtype == jnp.float64


@pytest.mark.parametrize("save_history", [False, True])
def test_wrapper_history_setting_preserves_positions_and_resume(save_history):
    quick = LieselOptim(
        _normal_model(),
        optimizers=optax.adam(0.02),
        loss_monitor="train_full_data",
        save_position_history=save_history,
        stopper=Stopper(epochs=4, patience=4),
        show_progress=False,
    )
    engine = quick.build_engine()
    whole = engine.fit()
    paused = engine.fit(pause_after=2)
    resumed = quick.build_engine().fit(checkpoint=paused.checkpoint)
    assert (whole.history.position is not None) == save_history
    assert (resumed.history.position is not None) == save_history
    assert resumed.position_min_monitor is not None
    assert jnp.array_equal(resumed.position_final["loc"], whole.position_final["loc"])
    assert jnp.array_equal(resumed.history.loss_monitor, whole.history.loss_monitor)
    assert jnp.array_equal(
        resumed.position_min_monitor["loc"], whole.position_min_monitor["loc"]
    )


def test_wrapper_requires_explicit_optimizer():
    with pytest.raises(TypeError, match="optimizers"):
        LieselOptim(_normal_model(), loss_monitor="train_full_data")  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    "transformation",
    [
        optax.adam(0.01),
        optax.sgd(0.1),
        optax.adam(optax.exponential_decay(0.02, transition_steps=2, decay_rate=0.5)),
        optax.GradientTransformation(*optax.sgd(0.1)),
        optax.chain(optax.clip(1.0), optax.adam(0.01)),
    ],
)
def test_direct_optax_transformation_matches_explicit_wrapper(transformation):
    model = _normal_model()

    def fit(optimizers):
        return LieselOptim(
            model,
            optimizers=optimizers,
            loss_monitor="train_full_data",
            stopper=Stopper(epochs=4, patience=4),
            show_progress=False,
        ).fit()

    direct = fit(transformation)
    wrapped = fit([opt.Optimizer(list(model.parameters), transformation)])
    assert jnp.array_equal(direct.position_final["loc"], wrapped.position_final["loc"])
    assert jnp.array_equal(direct.history.loss_monitor, wrapped.history.loss_monitor)


@pytest.mark.parametrize(
    "invalid, error", [("adam", ValueError), (None, TypeError), (optax.adam, TypeError)]
)
def test_wrapper_rejects_unconfigured_optimizer(invalid, error):
    with pytest.raises(error, match="configured Optax transformation"):
        LieselOptim(_normal_model(), optimizers=invalid, loss_monitor="train_full_data")


@pytest.mark.parametrize(
    "x64, to_float32", [(False, True), (True, True), (True, False)]
)
@pytest.mark.parametrize("debug", [False, True])
def test_lbfgs_preserves_model_dtype_during_line_search_and_resume(
    x64, to_float32, debug
):
    with jax.enable_x64(x64):
        model = _normal_model(8, to_float32=to_float32)
        engine = LieselOptim(
            model,
            optimizers="lbfgs",
            loss_monitor="train_full_data",
            stopper=Stopper(epochs=8, patience=4, min_epochs=8),
            show_progress=False,
        ).build_engine()
        engine.debug_nans = debug
        whole = engine.fit()
        paused = engine.fit(pause_after=1)
        resumed = engine.fit(checkpoint=paused.checkpoint)
        dtype = jnp.dtype("float64" if x64 and not to_float32 else "float32")
        for result in (whole, resumed):
            assert result.position_final["loc"].dtype == dtype
            assert result.history.position is not None
            assert result.history.position["loc"].dtype == dtype
            assert float(result.position_final["loc"]) == pytest.approx(3.5, abs=1e-5)
        assert jnp.allclose(resumed.history.loss_train, whole.history.loss_train)
        assert jnp.allclose(resumed.history.loss_monitor, whole.history.loss_monitor)


def _independent_parameter_model():
    observations = []
    for name, target in (("a", 3.0), ("b", 5.0)):
        parameter = lsl.Var.new_param(jnp.array(0.0), name=name)
        observations.append(
            lsl.Var.new_obs(
                jnp.full(4, target),
                lsl.Dist(tfd.Normal, loc=parameter, scale=1.0),
                name=f"y_{name}",
            )
        )
    return lsl.Model(observations)


@pytest.mark.parametrize(
    "kinds", [("lbfgs", "adam"), ("adam", "lbfgs"), ("lbfgs", "lbfgs")]
)
@pytest.mark.parametrize("delay", [0, 2])
@pytest.mark.parametrize("boundary", ["wrapper", "engine", "fit"])
def test_lbfgs_must_be_sole_optimizer(kinds, delay, boundary):
    model = _independent_parameter_model()
    optimizers = [
        opt.LBFGS([name], activate_after_epochs=delay)
        if kind == "lbfgs"
        else opt.Optimizer([name], optax.adam(0.01))
        for name, kind in zip(("a", "b"), kinds, strict=True)
    ]
    if boundary == "wrapper":
        with pytest.raises(ValueError, match="LBFGS must be the sole optimizer"):
            LieselOptim(model, optimizers=optimizers, loss_monitor="train_full_data")
    else:
        quick = LieselOptim(
            model,
            optimizers="lbfgs",
            loss_monitor="train_full_data",
            show_progress=False,
        )
        if boundary == "engine":
            quick.optimizers = optimizers
            with pytest.raises(ValueError, match="LBFGS must be the sole optimizer"):
                quick.build_engine()
        else:
            engine = quick.build_engine()
            engine.optimizers = optimizers
            with pytest.raises(ValueError, match="LBFGS must be the sole optimizer"):
                engine.fit()


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_single_lbfgs_can_select_a_subset_or_all_parameters(keys):
    result = LieselOptim(
        _independent_parameter_model(),
        optimizers=[opt.LBFGS(keys)],
        loss_monitor="train_full_data",
        stopper=Stopper(epochs=10, patience=3),
        show_progress=False,
    ).fit()
    assert float(result.position_final["a"]) == pytest.approx(3.0, abs=1e-5)
    if "b" in keys:
        assert float(result.position_final["b"]) == pytest.approx(5.0, abs=1e-5)
    else:
        assert "b" not in result.position_final


@pytest.mark.parametrize("index", [0, 1])
def test_wrapper_rejects_bare_transformations_in_sequences(index):
    optimizers = [opt.Optimizer(["loc"], optax.adam(0.1))] * index + [optax.adam(0.1)]
    with pytest.raises(
        TypeError, match=rf"optimizers\[{index}\].*single transformation.*Optimizer"
    ):
        LieselOptim(
            _normal_model(), optimizers=optimizers, loss_monitor="train_full_data"
        )


@pytest.mark.parametrize("wrapped", [False, True])
def test_unsupported_optax_lbfgs_has_actionable_update_error(wrapped):
    transformation = optax.lbfgs()
    optimizers = [opt.Optimizer(["loc"], transformation)] if wrapped else transformation
    quick = LieselOptim(
        _normal_model(),
        optimizers=optimizers,
        loss_monitor="train_full_data",
        show_progress=False,
    )
    with pytest.raises(
        TypeError, match="objective evaluations.*optimizers='lbfgs'"
    ) as caught:
        quick.fit()
    assert isinstance(caught.value.__cause__, TypeError)
    assert "value_fn" in str(caught.value.__cause__)
