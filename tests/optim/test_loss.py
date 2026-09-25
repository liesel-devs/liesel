import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.experimental import io_callback

import liesel.model as lsl
from liesel.optim import (
    Batches,
    EmaTrainLossMonitor,
    LieselOptim,
    LossMixin,
    NegLogProbLoss,
    PositionSplit,
    PositionSplitManager,
    Split,
    Stopper,
)
from liesel.optim.state import OptimCarry
from liesel.optim.types import Position


@pytest.mark.parametrize("optimizer", [optax.sgd(0.1), "lbfgs"])
def test_loss_can_select_default_outer_parameters(optimizer):
    class LocationLoss(NegLogProbLoss):
        default_position_keys = ("loc",)

    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    scale = lsl.Var.new_param(jnp.array(1.0), name="scale")
    y = lsl.Var.new_obs(
        jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, loc, scale), name="y"
    )
    model = lsl.Model([y])
    loss = LocationLoss(model, PositionSplit.from_model(model))
    result = LieselOptim(
        model,
        loss=loss,
        optimizers=optimizer,
        loss_monitor="train_full_data",
        stopper=Stopper(epochs=1, patience=1),
        show_progress=False,
    ).fit()
    assert set(result.position_final) == {"loc"}
    assert float(result.position_final["loc"]) > 0.0
    assert set(model.parameters) == {"loc", "scale"}


def test_loss_mixin_differentiates_value_with_auxiliary_state():
    class Quadratic(LossMixin):
        def loss_train_batched(self, params, carry):
            return (params["x"] - 1.0) ** 2, {"evaluated_at": params["x"]}

    loss = Quadratic()
    params = Position({"x": jnp.array(3.0)})
    carry = OptimCarry.new(
        key=jax.random.key(0),
        epochs=1,
        position=params,
        batches=Batches([], axis_size=1, batch_size=None),
        optimizers=[],
        model_state={},
        save_position_history=False,
    )
    (value, proposed), gradient = jax.jit(loss.value_and_grad)(params, carry)
    assert float(value) == 4.0
    assert float(proposed["evaluated_at"]) == 3.0
    assert float(gradient["x"]) == 4.0
    assert float(loss.grad(params, carry)["x"]) == 4.0
    assert loss.init_state(params, carry) is None
    assert loss.default_position_keys is None


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


def _counted_basis_model(extra_branch=False, scale_fn=None):
    calls = []

    def square(values):
        calls.append(np.asarray(values).copy())
        return np.square(values)

    def basis_fn(values):
        return io_callback(
            square,
            jax.ShapeDtypeStruct(values.shape, values.dtype),
            values,
            ordered=True,
        )

    x = lsl.Var.new_value(jnp.arange(1.0, 7.0), name="x")
    basis = lsl.Var.new_calc(basis_fn, x, name="basis")
    beta = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="beta")
    loc = lsl.Var.new_calc(jnp.multiply, basis, beta)
    scale = 1.0 if scale_fn is None else lsl.Var.new_calc(scale_fn, x, name="scale")
    y = lsl.Var.new_obs(2 * x.value**2, lsl.Dist(tfd.Normal, loc, scale), name="y")
    if extra_branch:
        z = lsl.Var.new_obs(
            jnp.array([1.0, 1.0, 100.0, 100.0]),
            lsl.Dist(tfd.Normal, beta, 1.0),
            name="z",
        )
        model = lsl.Model([y, z])
    else:
        model = lsl.Model(y)
    jax.effects_barrier()
    return model, calls


@pytest.mark.parametrize(
    "optimizer,batch_size", [("lbfgs", None), ("sgd", None), ("sgd", 2)]
)
def test_fit_reuses_full_data_basis(optimizer, batch_size):
    model, calls = _counted_basis_model()
    optim = LieselOptim(
        model,
        split=PositionSplit.from_model(model, position_keys=["x", "y"], shuffle=False),
        batch_size=batch_size,
        optimizers="lbfgs" if optimizer == "lbfgs" else optax.sgd(0.0001),
        loss_monitor="train_full_data",
        stopper=Stopper(epochs=5, patience=5, min_epochs=5),
        show_progress=False,
    )
    calls.clear()
    result = optim.fit()
    jax.effects_barrier()
    assert result.n_epochs == 5
    assert sum(call.size == 6 for call in calls) == 1
    assert sum(call.size == 2 for call in calls) == (15 if batch_size == 2 else 0)
    assert result.history.position is not None
    beta = result.history.position["beta"]
    # Six Normal observations with basis x**2, and a standard Normal prior.
    expected = (
        7 * np.log(2 * np.pi) / 2
        + (2 - beta) ** 2 * np.sum(np.arange(1.0, 7.0) ** 4) / 2
        + beta**2 / 2
    ) / 6
    np.testing.assert_allclose(result.history.loss_monitor, expected, rtol=1e-5)


@pytest.mark.parametrize("holdout", ["validate", "test"])
@pytest.mark.parametrize("batch_size", [None, 1])
def test_prepared_partitions_keep_omitted_batch_keys_on_training_rows(
    holdout, batch_size
):
    model, calls = _counted_basis_model(extra_branch=True)
    split = PositionSplit.from_model(
        model,
        position_keys=[["x", "y"], ["z"]],
        multi_size="manager",
        shuffle=False,
        validate_axis_share=0.5 if holdout == "validate" else 0.0,
        test_axis_share=0.5 if holdout == "test" else 0.0,
    )
    optim = LieselOptim(
        model,
        split=split,
        batches=Batches.from_split(
            split, position_keys=["z"], batch_size=batch_size, shuffle=False
        ),
        optimizers=optax.sgd(0.01),
        loss_monitor="validation" if holdout == "validate" else "train_full_data",
        stopper=Stopper(epochs=5, patience=5, min_epochs=5),
        show_progress=False,
    )
    calls.clear()
    result = optim.fit()
    jax.effects_barrier()
    expected_calls = [[1.0, 2.0, 3.0]]
    if holdout == "validate":
        if batch_size == 1:
            # Without a training template, the omitted group is evaluated per batch.
            expected_calls = [[4.0, 5.0, 6.0]] + expected_calls * 10
        else:
            expected_calls += [[4.0, 5.0, 6.0]]
    np.testing.assert_array_equal(calls, expected_calls)

    # The same gradient applies at each step: the two training z values coincide.
    steps = np.arange(1, 6) * (1 if batch_size is None else 2)
    beta = (198 / 101) * (1 - (1 - 0.01 * 101 / 5) ** steps)
    assert result.history.position is not None
    np.testing.assert_allclose(result.history.position["beta"], beta, rtol=1e-5)
    x = np.arange(4.0, 7.0) if holdout == "validate" else np.arange(1.0, 4.0)
    z = 100.0 if holdout == "validate" else 1.0
    expected = (
        5 * np.log(2 * np.pi) / 2 + (2 - beta) ** 2 * np.sum(x**4) / 2 + (z - beta) ** 2
    )
    if holdout == "test":
        expected += np.log(2 * np.pi) / 2 + beta**2 / 2
    np.testing.assert_allclose(result.history.loss_monitor, expected / 5, rtol=1e-5)


@pytest.mark.parametrize(
    "batch_size,monitor",
    [(None, "validation"), (1, "validation"), (1, "ema"), (1, "train_full_data")],
)
def test_fit_batches_computed_basis_without_callbacks(batch_size, monitor):
    model, calls = _counted_basis_model(extra_branch=True)

    def fit(data_key):
        split = PositionSplit.from_model(
            model,
            position_keys=[[data_key, "y"], ["z"]],
            multi_size="manager",
            validate_axis_share=0.5,
            shuffle=False,
        )
        return LieselOptim(
            model,
            split=split,
            batches=Batches.from_split(
                split,
                position_keys=[data_key, "y"],
                batch_size=batch_size,
                shuffle=False,
            ),
            optimizers=optax.sgd(0.01),
            loss_monitor=(
                EmaTrainLossMonitor(effective_window=2) if monitor == "ema" else monitor
            ),
            stopper=Stopper(epochs=3, patience=3, min_epochs=3),
            seed=12,
            show_progress=False,
        ).fit()

    calls.clear()
    computed = fit("basis")
    jax.effects_barrier()
    assert not calls
    raw = fit("x")
    assert computed.history.position is not None
    assert raw.history.position is not None
    np.testing.assert_allclose(
        computed.history.position["beta"], raw.history.position["beta"], rtol=1e-6
    )
    np.testing.assert_allclose(
        computed.history.loss_monitor, raw.history.loss_monitor, rtol=1e-6
    )
    # Gaussian SGD reference: the omitted z group contributes only training [1, 1].
    expected = (
        (198 / 101) * (1 - 0.798 ** np.arange(1, 4))
        if batch_size is None
        else [1.082866944, 1.5709256790939157, 1.7908985300718774]
    )
    np.testing.assert_allclose(computed.history.position["beta"], expected, rtol=1e-5)


def test_fit_mixes_precomputed_matrix_rows_and_native_covariate_batches():
    calls = []

    def matrix(values):
        calls.append(np.asarray(values).copy())
        return np.column_stack((np.ones_like(values), values))

    def basis_fn(values):
        return io_callback(
            matrix,
            jax.ShapeDtypeStruct((values.size, 2), values.dtype),
            values,
            ordered=True,
        )

    x1 = lsl.Var.new_obs(jnp.arange(-2.0, 6.0), name="x1")
    x2 = lsl.Var.new_obs(
        jnp.array([1.0, 3.0, -1.0, 2.0, 0.0, 4.0, -2.0, 1.0]), name="x2"
    )
    basis = lsl.Var.new_calc(basis_fn, x1, name="basis")
    beta = lsl.Var.new_param(jnp.zeros(2), lsl.Dist(tfd.Normal, 0.0, 1.0), name="beta")
    gamma = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="gamma")
    loc = lsl.Var.new_calc(
        lambda basis, beta, x2, gamma: basis @ beta + gamma * x2**2,
        basis,
        beta,
        x2,
        gamma,
    )
    y = lsl.Var.new_obs(jnp.arange(8.0) / 5, lsl.Dist(tfd.Normal, loc, 1.0), name="y")
    model = lsl.Model(y)
    jax.effects_barrier()

    def fit(data_key):
        split = PositionSplit.from_model(
            model,
            position_keys=[data_key, "x2", "y"],
            validate_axis_share=0.25,
            seed=12,
        )
        return LieselOptim(
            model,
            split=split,
            batch_size=2,
            optimizers=optax.sgd(0.005),
            loss_monitor="validation",
            stopper=Stopper(epochs=3, patience=3, min_epochs=3),
            seed=13,
            show_progress=False,
        ).fit()

    calls.clear()
    computed = fit("basis")
    jax.effects_barrier()
    assert not calls
    raw = fit("x1")
    jax.effects_barrier()
    assert calls
    for actual, expected in zip(
        jax.tree.leaves(computed.history), jax.tree.leaves(raw.history), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("monitor", ["ema", "validation"])
def test_minibatches_skip_full_training_basis_and_keep_omitted_training_rows(
    tmp_path, debug, monitor
):
    model, calls = _counted_basis_model(extra_branch=True)
    split = PositionSplit.from_model(
        model,
        position_keys=[["x", "y"], ["z"]],
        multi_size="manager",
        shuffle=False,
        validate_axis_share=0.5,
    )
    engine = LieselOptim(
        model,
        split=split,
        batches=Batches.from_split(
            split, position_keys=["x", "y"], batch_size=1, shuffle=False
        ),
        optimizers=optax.sgd(0.01),
        loss_monitor=(
            EmaTrainLossMonitor(effective_window=2)
            if monitor == "ema"
            else "validation"
        ),
        stopper=Stopper(epochs=3, patience=3, min_epochs=3),
        show_progress=False,
    ).build_engine()
    engine.debug_nans = debug
    calls.clear()
    checkpoint = tmp_path / "fit.pkl"
    paused = engine.fit(checkpoint=checkpoint, pause_after=1)
    assert paused.status == "paused"
    result = engine.fit(checkpoint=checkpoint)
    jax.effects_barrier()
    assert not any(np.array_equal(call, [1.0, 2.0, 3.0]) for call in calls)
    assert result.n_epochs == 3
    assert result.history.position is not None
    # The three scalar Gaussian SGD steps compose to
    # beta_next = 0.450709792 * beta + 1.082866944, using only training z = [1, 1].
    np.testing.assert_allclose(
        result.history.position["beta"],
        [1.082866944, 1.5709256790939157, 1.7908985300718774],
        rtol=1e-5,
    )


@pytest.mark.parametrize("debug", [False, True])
def test_checkpoint_rebuilds_data_states_once(tmp_path, debug):
    def build_engine():
        model, calls = _counted_basis_model()
        split = PositionSplit.from_model(
            model, position_keys=["x", "y"], validate_axis_share=0.5, shuffle=False
        )
        engine = LieselOptim(
            model,
            split=split,
            optimizers=optax.sgd(0.0001),
            loss_monitor="validation",
            stopper=Stopper(epochs=5, patience=5, min_epochs=5),
            show_progress=False,
        ).build_engine()
        engine.debug_nans = debug
        calls.clear()
        return engine, calls

    engine, calls = build_engine()
    checkpoint = tmp_path / "fit.pkl"
    engine.fit(checkpoint=checkpoint, pause_after=2)
    jax.effects_barrier()
    assert len(calls) == 2
    engine, calls = build_engine()
    resumed = engine.fit(checkpoint=checkpoint)
    jax.effects_barrier()
    assert len(calls) == 2
    uninterrupted = engine.fit()
    for actual, expected in zip(
        jax.tree.leaves(resumed.history),
        jax.tree.leaves(uninterrupted.history),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_nan_reproduction_uses_prepared_training_data():
    # Training rows give a negative Normal scale; the full data give a valid scale.
    model, calls = _counted_basis_model(scale_fn=lambda x: x.mean() - 3)
    split = PositionSplit.from_model(
        model, position_keys=["x", "y"], test_axis_share=0.5, shuffle=False
    )
    engine = LieselOptim(
        model,
        split=split,
        optimizers=optax.sgd(0.0001),
        loss_monitor="train_full_data",
        stopper=Stopper(epochs=5, patience=5),
        show_progress=False,
    ).build_engine()
    engine.debug_nans = True
    calls.clear()
    result = engine.fit()
    assert result.status == "nan"
    assert result.nan_debug is not None
    assert jnp.isnan(result.nan_debug.reproduce_loss(engine))
    jax.effects_barrier()
    assert len(calls) == 1


def test_neg_log_prob_loss_train_uses_full_training_split_not_current_batch():
    model = _normal_obs_model()
    split = Split(
        ["y"], axis_size=6, validate_axis_size=2, shuffle=False
    ).split_position(model.extract_position(["y"]))
    loss = NegLogProbLoss(model, split)
    carry = _empty_carry(model)
    carry.batch = Position({"y": jnp.array([1000.0, 2000.0])})

    value = loss.loss_train(Position({}), carry)[0]
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

    unscaled = NegLogProbLoss(model, split).loss_train(Position({}), carry)[0]
    scaled_loss = NegLogProbLoss(model, split, scale=True)
    scaled = scaled_loss.loss_train(Position({}), carry)[0]

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

    unscaled = NegLogProbLoss(model, split).loss_train(Position({}), carry)[0]
    scaled_loss = NegLogProbLoss(model, split, scale=True)
    scaled = scaled_loss.loss_train(Position({}), carry)[0]

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

    unscaled = NegLogProbLoss(model, split).loss_train(Position({}), carry)[0]
    scaled_loss = NegLogProbLoss(model, split, scale=True)
    scaled = scaled_loss.loss_train(Position({}), carry)[0]
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

    unscaled = NegLogProbLoss(model, split).loss_train(Position({}), carry)[0]
    scaled_loss = NegLogProbLoss(model, split, scale=True)
    scaled = scaled_loss.loss_train(Position({}), carry)[0]
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
                    loss.loss_monitor(params, carry)[0], expected / loss.scalar
                )
        held_out_scores.append(scores)
        training_scores.append(float(loss.loss_train(params, carry)[0]))
        full_split = PositionSplit.from_model(
            model, position_keys=keys, split_axes={"z": None}, multi_size="manager"
        )
        fallback = NegLogProbLoss(model, full_split, validation_strategy="log_prob")
        assert jnp.allclose(
            fallback.loss_monitor(params, carry)[0],
            fallback.loss_train(params, carry)[0],
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
