from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
from liesel.optim import (
    LBFGS,
    Batches,
    BatchManager,
    EmaTrainLossMonitor,
    LieselVI,
    NegElboLoss,
    OptimEngine,
    PositionSplit,
    PositionSplitManager,
    Stopper,
)
from liesel.optim.state import OptimResult

LOSS_MONITOR = EmaTrainLossMonitor(effective_window=1.0)


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
    from liesel.optim.liesel_vi import LieselVI as LieselVIFromModule

    assert opt.LieselVI is LieselVI
    assert LieselVIFromModule is LieselVI


def test_default_build_engine_uses_opinionated_defaults():
    model = _normal_model()

    engine = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor=LOSS_MONITOR,
        seed=1,
    ).build_engine()

    assert isinstance(engine, OptimEngine)
    assert isinstance(engine.loss, NegElboLoss)
    assert engine.loss.scale is True
    assert engine.loss.scalar == engine.split.train_axis_size
    assert engine.loss.nsamples == 10
    assert engine.loss.entropy == "auto"
    assert isinstance(engine.loss.vdist, opt.VDist)
    assert engine.loss.vdist.var is not None
    assert engine.loss.vdist.var.dist_node is not None
    assert engine.loss.vdist.var.dist_node.distribution is tfd.MultivariateNormalDiag
    assert isinstance(engine.batches, Batches)
    assert engine.batches.is_full_data
    assert engine.batches.axis_size == engine.split.train_axis_size
    assert engine.optimizers[0].position_keys == tuple(engine.loss.q.parameters)
    assert engine.stopper == Stopper(epochs=1000, patience=10, rtol=1e-6)
    assert engine.loss_monitor is LOSS_MONITOR
    assert engine.progress_update_every == 10
    assert engine.show_step_progress is False
    assert engine.step_progress_update_every == 10


def test_batch_size_shortcut_builds_training_batches():
    model = _normal_model()
    split = PositionSplit.from_model(model, test_axis_share=0.25, seed=1)

    engine = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor=LOSS_MONITOR,
        split=split,
        batch_size=2,
        seed=1,
    ).build_engine()

    assert isinstance(engine.batches, Batches)
    assert engine.batches.axis_size == split.train_axis_size
    assert engine.batches.batch_size == 2


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("batch_axis_size", 2),
        ("elbo", "mvn_diag"),
        ("progress_n_updates", 7),
        ("step_progress_n_updates", 4),
    ],
)
def test_removed_alias_argument_raises(keyword, value):
    with pytest.raises(TypeError, match=keyword):
        LieselVI(
            _normal_model(),
            optimizers=optax.adam(1e-3),
            loss_monitor=LOSS_MONITOR,
            **{keyword: value},
        )


@pytest.mark.parametrize(
    "name", ["elbo", "progress_n_updates", "step_progress_n_updates"]
)
def test_removed_alias_attribute_unavailable(name):
    setup = LieselVI(
        _normal_model(), optimizers=optax.adam(1e-3), loss_monitor=LOSS_MONITOR
    )
    assert not hasattr(setup, name)


def test_validation_split_raises():
    model = _normal_model()
    split = PositionSplit.from_model(model, validate_axis_share=0.25, seed=1)

    with pytest.raises(ValueError, match="validation data"):
        LieselVI(
            model,
            optimizers=optax.adam(learning_rate=1e-3),
            loss_monitor=LOSS_MONITOR,
            split=split,
        )


def test_validation_monitor_raises():
    model = _normal_model()

    with pytest.raises(ValueError, match="does not support.*validation"):
        LieselVI(
            model, optimizers=optax.adam(learning_rate=1e-3), loss_monitor="validation"
        )


def test_invalid_monitor_raises():
    model = _normal_model()

    with pytest.raises(ValueError, match="loss_monitor"):
        LieselVI(
            model,
            optimizers=optax.adam(learning_rate=1e-3),
            loss_monitor="sometimes",  # ty: ignore[invalid-argument-type]
        )


def test_batches_and_batch_size_are_mutually_exclusive():
    model = _normal_model()
    batches = Batches(["y"], axis_size=6, batch_size=None)

    with pytest.raises(ValueError, match="batch_size or batches"):
        LieselVI(
            model,
            optimizers=optax.adam(learning_rate=1e-3),
            loss_monitor=LOSS_MONITOR,
            batches=batches,
            batch_size=2,
        )


def test_old_batch_keyword_rejected_also_with_batch_size():
    model = _normal_model()

    with pytest.raises(TypeError, match="batch_axis_size"):
        LieselVI(
            model,
            optimizers=optax.adam(learning_rate=1e-3),
            loss_monitor=LOSS_MONITOR,
            batch_size=2,
            batch_axis_size=2,  # ty: ignore[unknown-argument]
        )


def test_user_provided_batches_are_not_mutated():
    model = _normal_model(n=2)
    batches = Batches(["y"], axis_size=2, batch_size=None)

    vi = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor=LOSS_MONITOR,
        batches=batches,
        seed=1,
    )
    engine = vi.build_engine()

    assert vi.batches is batches
    assert engine.batches is batches
    assert batches.axis_size == 2


def test_multi_size_default_split_builds_batch_manager():
    model = _two_branch_model()

    engine = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor=LOSS_MONITOR,
        batch_size=None,
        seed=1,
    ).build_engine()

    assert isinstance(engine.split, PositionSplitManager)
    assert isinstance(engine.batches, BatchManager)
    assert engine.batches.axis_size == engine.split.train_axis_sizes
    assert isinstance(engine.loss, NegElboLoss)
    assert engine.loss.scale is True
    assert engine.loss.scalar == sum(engine.split.train_axis_sizes)


@pytest.mark.parametrize("epoch_size,expected", [("max", 4), ("min", 2), (3, 3)])
def test_multi_size_minibatches_use_joint_epoch_size(epoch_size, expected):
    model = _two_branch_model()
    split = PositionSplit.from_model(model, multi_size="manager", shuffle=False)
    batches = Batches.from_split(split, batch_size=2, epoch_size=epoch_size)
    engine = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor=LOSS_MONITOR,
        split=split,
        batches=batches,
        seed=1,
    ).build_engine()
    assert engine.batches.n_full_batches == expected


@pytest.mark.parametrize("managed", [False, True])
def test_weighted_vi_checkpoint_continues_same_run(tmp_path, managed):
    def engine():
        model = _two_branch_model() if managed else _normal_model()
        children = [
            Batches(
                [name],
                axis_size=len(var.value),
                batch_size=2,
                sample_with_replacement=True,
                sampling_weights=jnp.arange(1, len(var.value) + 1),
            )
            for name, var in model.observed.items()
        ]
        batches = BatchManager(children, epoch_size="max") if managed else children[0]
        return LieselVI(
            model,
            optimizers=optax.adam(learning_rate=1e-3),
            batches=batches,
            loss_monitor=LOSS_MONITOR,
            stopper=Stopper(epochs=3, patience=3),
            nsamples=1,
            seed=1,
            show_progress=False,
        ).build_engine()

    expected = engine().fit()
    path = tmp_path / "vi.pkl"
    engine().fit(checkpoint=path, pause_after=1)
    actual = engine().fit(checkpoint=path)
    for a, b in zip(
        jax.tree.leaves((actual.position_final, actual.history)),
        jax.tree.leaves((expected.position_final, expected.history)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_scale_loss_false_builds_unscaled_default_loss():
    model = _normal_model()

    engine = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor=LOSS_MONITOR,
        scale_loss=False,
        seed=1,
    ).build_engine()

    assert isinstance(engine.loss, NegElboLoss)
    assert engine.loss.scale is False
    assert engine.loss.scalar == 1.0


def test_custom_loss_and_conflicting_split_raise():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    other_split = PositionSplit.from_model(model)
    loss = NegElboLoss.mvn_diag(model, split=split)

    with pytest.raises(ValueError, match="loss.split"):
        LieselVI(
            model,
            optimizers=optax.adam(learning_rate=1e-3),
            loss_monitor=LOSS_MONITOR,
            loss=loss,
            split=other_split,
        )


def test_custom_loss_is_passed_through_unchanged():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    loss = NegElboLoss.mvn_diag(
        model, split=split, nsamples=3, scale=False, entropy="mc"
    )

    engine = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor=LOSS_MONITOR,
        loss=loss,
        nsamples=99,
        scale_loss=True,
        seed=1,
    ).build_engine()

    assert engine.loss is loss
    assert loss.nsamples == 3
    assert loss.scale is False
    assert loss.entropy == "mc"


@pytest.mark.parametrize("family", ["mvn_diag", "mvn_tril", "mvn_blocked"])
def test_entropy_option_is_forwarded(family):
    engine = LieselVI(
        _normal_model(),
        optimizers=optax.adam(learning_rate=1e-3),
        loss=family,
        entropy="mc",
        loss_monitor=LOSS_MONITOR,
    ).build_engine()
    assert isinstance(engine.loss, NegElboLoss)
    assert engine.loss.entropy == "mc"


def test_invalid_entropy_option_raises():
    with pytest.raises(ValueError, match="entropy"):
        LieselVI(
            _normal_model(),
            optimizers=optax.adam(learning_rate=1e-3),
            entropy="invalid",  # ty: ignore[invalid-argument-type]
            loss_monitor=LOSS_MONITOR,
        )


def test_unknown_loss_string_raises():
    model = _normal_model()

    with pytest.raises(ValueError, match="loss"):
        LieselVI(
            model,
            optimizers=optax.adam(learning_rate=1e-3),
            loss_monitor=LOSS_MONITOR,
            loss="mean_field",  # ty: ignore[invalid-argument-type]
        )


def test_lbfgs_string_shortcut_raises_with_vi_specific_message():
    model = _normal_model()

    with pytest.raises(ValueError, match="ELBO.*stochastic"):
        LieselVI(
            model,
            loss_monitor=LOSS_MONITOR,
            optimizers="lbfgs",  # ty: ignore[invalid-argument-type]
        )


def test_explicit_lbfgs_optimizer_sequence_is_accepted():
    model = _normal_model()
    split = PositionSplit.from_model(model)
    loss = NegElboLoss.mvn_diag(model, split=split)
    optimizer = LBFGS(list(loss.q.parameters))

    engine = LieselVI(
        model,
        loss_monitor=LOSS_MONITOR,
        loss=loss,
        optimizers=[optimizer],
        seed=1,
    ).build_engine()

    assert engine.optimizers[0] is optimizer


def test_progress_and_loss_monitor_are_passed_to_engine():
    model = _normal_model()
    loss_monitor = EmaTrainLossMonitor(0.5)

    setup = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        batch_size=1,
        loss_monitor=loss_monitor,
        show_progress=False,
        progress_update_every=3,
        show_step_progress=True,
        step_progress_update_every=4,
        seed=1,
    )

    engine = setup.build_engine()

    assert engine.loss_monitor is loss_monitor
    assert engine.show_progress is False
    assert engine.progress_update_every == 3
    assert engine.show_step_progress is True
    assert engine.step_progress_update_every == 4


def test_fit_returns_optim_result():
    model = _normal_model()

    result = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor=LOSS_MONITOR,
        stopper=Stopper(epochs=1, patience=1),
        nsamples=1,
        seed=1,
    ).fit()

    assert isinstance(result, OptimResult)
    assert result.monitor_source == "train_ema"
    assert result.n_epochs == 1
    assert result.position_final is not None
    assert result.position_min_monitor is not None


def test_fit_with_full_data_monitor_returns_minimum_monitor_position():
    model = _normal_model()

    result = LieselVI(
        model,
        optimizers=optax.adam(learning_rate=1e-3),
        loss_monitor="train_full_data",
        stopper=Stopper(epochs=1, patience=1),
        nsamples=1,
        seed=1,
    ).fit()

    assert result.monitor_source == "train_full_data"
    assert result.position_min_monitor is not None


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
            optimizers=optax.adam(learning_rate=1e-3),
            loss_monitor=LOSS_MONITOR,
            stopper=Stopper(epochs=1, patience=1),
            nsamples=1,
            seed=1,
        ).fit()

    assert isinstance(result, OptimResult)
    assert result.history.loss_train.dtype == jnp.float64


def test_optimizer_is_required():
    with pytest.raises(TypeError, match="optimizers"):
        LieselVI(_normal_model(), loss_monitor=LOSS_MONITOR)  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    "optimizers,error,match",
    [
        ("adam", ValueError, "configured Optax"),
        ("unknown", ValueError, "configured Optax"),
        (optax.adam, TypeError, "optimizer factory"),
        ([optax.adam(1e-3)], TypeError, "bare Optax"),
    ],
)
def test_invalid_optimizer_configuration_raises(optimizers, error, match):
    with pytest.raises(error, match=match):
        LieselVI(_normal_model(), loss_monitor=LOSS_MONITOR, optimizers=optimizers)


def test_configured_transform_fit_matches_explicit_q_optimizer():
    model = _normal_model()
    loss = NegElboLoss.mvn_diag(model, nsamples=2)
    transform = optax.adam(learning_rate=1e-3)
    options: dict[str, Any] = {
        "loss": loss,
        "loss_monitor": LOSS_MONITOR,
        "stopper": Stopper(epochs=3, patience=3),
        "seed": 11,
        "show_progress": False,
    }
    actual = LieselVI(model, optimizers=transform, **options).fit()
    expected = LieselVI(
        model, optimizers=[opt.Optimizer(list(loss.q.parameters), transform)], **options
    ).fit()
    for a, b in zip(
        jax.tree.leaves((actual.position_final, actual.history)),
        jax.tree.leaves((expected.position_final, expected.history)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_explicit_lbfgs_minibatches_rejected_at_construction():
    model = _normal_model()
    loss = NegElboLoss.mvn_diag(model)
    with pytest.raises(ValueError, match="full-data"):
        LieselVI(
            model,
            loss=loss,
            loss_monitor=LOSS_MONITOR,
            optimizers=[LBFGS(list(loss.q.parameters))],
            batch_size=2,
        )


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("axis_size", 6),
        ("split_axes", {"y": 0}),
        ("default_split_axis", 0),
        ("shuffle_batches", False),
        ("epoch_size", "max"),
    ],
)
def test_removed_data_factory_argument_raises(keyword, value):
    with pytest.raises(TypeError, match=keyword):
        LieselVI(
            _normal_model(),
            optimizers=optax.adam(1e-3),
            loss_monitor=LOSS_MONITOR,
            **{keyword: value},
        )


def test_default_seed_matches_explicit_zero_and_repeated_fit():
    options: dict[str, Any] = {
        "optimizers": optax.adam(1e-3),
        "loss_monitor": LOSS_MONITOR,
        "stopper": Stopper(epochs=2, patience=2),
        "nsamples": 2,
        "show_progress": False,
    }
    setup = LieselVI(_normal_model(), **options)
    assert setup.seed == 0
    actual = setup.fit()
    for expected in (setup.fit(), LieselVI(_normal_model(), seed=0, **options).fit()):
        for a, b in zip(
            jax.tree.leaves((actual.position_final, actual.history)),
            jax.tree.leaves((expected.position_final, expected.history)),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)


def test_default_stoppers_are_independent():
    first = LieselVI(
        _normal_model(), optimizers=optax.adam(1e-3), loss_monitor=LOSS_MONITOR
    )
    second = LieselVI(
        _normal_model(), optimizers=optax.adam(1e-3), loss_monitor=LOSS_MONITOR
    )
    first.stopper.epochs = 2
    assert second.stopper == Stopper(epochs=1000, patience=10, rtol=1e-6)


@pytest.mark.parametrize("scale_loss", ["auto", 1, None])
def test_scale_loss_requires_boolean(scale_loss):
    with pytest.raises(ValueError, match="scale_loss must be True or False"):
        LieselVI(
            _normal_model(),
            optimizers=optax.adam(1e-3),
            loss_monitor=LOSS_MONITOR,
            scale_loss=scale_loss,
        )


def test_history_control_keeps_losses_and_final_best_positions():
    options: dict[str, Any] = {
        "optimizers": optax.adam(1e-3),
        "loss_monitor": "train_full_data",
        "stopper": Stopper(epochs=3, patience=3),
        "nsamples": 2,
        "seed": 0,
        "show_progress": False,
    }
    actual = LieselVI(_normal_model(), save_position_history=False, **options).fit()
    expected = LieselVI(_normal_model(), **options).fit()
    assert actual.history.position is None
    assert expected.history.position is not None
    for a, b in zip(
        jax.tree.leaves(
            (
                actual.position_final,
                actual.position_min_monitor,
                actual.history.loss_train,
                actual.history.loss_monitor,
            )
        ),
        jax.tree.leaves(
            (
                expected.position_final,
                expected.position_min_monitor,
                expected.history.loss_train,
                expected.history.loss_monitor,
            )
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("batch_size", [None, 1])
def test_test_holdout_with_omitted_batch_key_matches_training_only_fit(batch_size):
    def fit(include_holdout):
        loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
        y1 = lsl.Var.new_obs(
            jnp.array([2.0] * 5 + ([100.0] * 5 if include_holdout else [])),
            lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
            name="y1",
        )
        y2 = lsl.Var.new_obs(
            jnp.array([1.0] * 3 + ([200.0] * 3 if include_holdout else [])),
            lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
            name="y2",
        )
        model = lsl.Model([y1, y2])
        split = PositionSplit.from_model(
            model,
            multi_size="manager",
            shuffle=False,
            test_axis_share=0.5 if include_holdout else 0.0,
        )
        batches = Batches.from_split(split, position_keys=["y1"], batch_size=batch_size)
        return LieselVI(
            model,
            split=split,
            batches=batches,
            optimizers=optax.adam(1e-3),
            loss_monitor="train_full_data",
            stopper=Stopper(epochs=3, patience=3),
            nsamples=2,
            seed=7,
            show_progress=False,
        ).fit()

    actual, expected = fit(True), fit(False)
    for a, b in zip(
        jax.tree.leaves((actual.position_final, actual.history)),
        jax.tree.leaves((expected.position_final, expected.history)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("batch_size", [None, 1])
def test_computed_basis_fit_matches_raw_training_only_fit(batch_size):
    def fit(computed, heldout):
        x_values = jnp.array([0.0, 1.0, 2.0] + ([30.0, 40.0, 50.0] if heldout else []))
        y_values = jnp.array(
            [1.0, 2.0, 3.0] + ([100.0, 200.0, 300.0] if heldout else [])
        )
        x = lsl.Var.new_obs(x_values, name="x")
        basis = lsl.Var.new_calc(
            lambda x: jnp.stack((jnp.ones_like(x), x), axis=-1), x, name="basis"
        )
        beta = lsl.Var.new_param(jnp.zeros(2), name="beta")
        mu = lsl.Var.new_calc(lambda matrix, coef: matrix @ coef, basis, beta)
        y = lsl.Var.new_obs(y_values, lsl.Dist(tfd.Normal, loc=mu, scale=1.0), name="y")
        model = lsl.Model([y])
        split = PositionSplit.from_model(
            model,
            position_keys=["basis" if computed else "x", "y"],
            test_axis_share=0.5 if heldout else 0.0,
            shuffle=False,
        )
        return LieselVI(
            model,
            split=split,
            batch_size=batch_size,
            optimizers=optax.adam(1e-3),
            loss_monitor="train_full_data",
            nsamples=2,
            seed=7,
            stopper=Stopper(epochs=3, patience=3),
            show_progress=False,
        ).fit()

    actual, expected = (
        fit(computed=True, heldout=True),
        fit(computed=False, heldout=False),
    )
    for a, b in zip(
        jax.tree.leaves((actual.position_final, actual.history)),
        jax.tree.leaves((expected.position_final, expected.history)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
