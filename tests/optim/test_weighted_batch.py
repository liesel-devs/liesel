"""Weighted sampling and correction through the public optimization API."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
from liesel.optim import Batches, BatchManager
from liesel.optim.types import Position


def test_float16_weights_reach_all_indices_and_preserve_indicator_mean():
    batches = Batches(
        ["y"],
        axis_size=4096,
        batch_size=65536,
        sample_with_replacement=True,
        sampling_weights=np.ones(4096, dtype=np.float16),
    )
    sampled = jax.jit(lambda b: b.start_epoch(jax.random.key(42)))(batches)
    indices = np.asarray(sampled.indices)
    assert np.unique(indices).size == 4096
    estimate = np.mean((indices % 4 != 3) * np.asarray(sampled.correction_factors(0)))
    assert estimate == pytest.approx(0.75, abs=0.01)
    assert batches.sampling_probabilities is not None
    assert batches.sampling_probabilities.dtype == jnp.float32


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_probability_and_correction_precision_respects_jax_configuration(dtype):
    batches = Batches(
        ["y"],
        3,
        3,
        sample_with_replacement=True,
        sampling_weights=np.array([1, 2, 7], dtype=dtype),
    )
    expected_dtype = (
        jnp.float64
        if dtype == np.float64 and jax.config.read("jax_enable_x64")
        else jnp.float32
    )
    assert batches.sampling_probabilities is not None
    assert batches.sampling_probabilities.dtype == expected_dtype
    assert batches.correction_factors(0).dtype == expected_dtype


def test_weighted_replacement_sampling_uses_relative_priorities():
    batches = Batches(
        ["y"],
        axis_size=3,
        batch_size=20000,
        sample_with_replacement=True,
        sampling_weights=[1.0, 2.0, 7.0],
    )
    key = jax.random.key(13)
    sampled = jax.jit(lambda b: b.start_epoch(key))(batches)
    frequencies = np.bincount(np.asarray(sampled.indices), minlength=3) / 20000
    np.testing.assert_allclose(frequencies, [0.1, 0.2, 0.7], atol=0.015)
    np.testing.assert_array_equal(batches.permute_indices(key), sampled.indices)
    assert batches.sampling_probabilities is not None
    np.testing.assert_allclose(batches.sampling_probabilities, [0.1, 0.2, 0.7])
    selected = sampled.get_batched_position(Position({"y": jnp.arange(3)}), 0)
    np.testing.assert_array_equal(selected["y"], sampled.indices)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, 0.0],
        [1.0, -1.0],
        [1.0, float("nan")],
        [1.0, float("inf")],
        [1.0],
        [[1.0, 2.0]],
        np.array([1.0, 1e-45], dtype=np.float32),
    ],
)
def test_weights_must_be_positive_finite_and_representable(weights):
    with pytest.raises(ValueError, match="sampling_weights"):
        Batches(["y"], 2, 1, sample_with_replacement=True, sampling_weights=weights)


def test_weights_require_replacement_sampling():
    with pytest.raises(ValueError, match="sample_with_replacement"):
        Batches(["y"], 2, 1, sampling_weights=[1.0, 2.0])


@pytest.mark.parametrize("scale", [2.0, 6.0])
def test_expected_corrected_likelihood_and_gradient_preserve_full_objective(scale):
    loc = lsl.Var.new_param(jnp.array(0.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="loc")
    y = lsl.Var.new_obs(jnp.array([1.0, 4.0]), lsl.Dist(tfd.Normal, loc, 1.0), name="y")
    z = lsl.Var.new_obs(jnp.array([2.0]), lsl.Dist(tfd.Normal, loc, 2.0), name="z")
    model = lsl.Model([y, z])
    batches = Batches(
        ["y"],
        2,
        1,
        sample_with_replacement=True,
        sampling_weights=[1.0, 3.0],
        sample_size=scale,
        batch_sample_size=1.0,
    )

    def estimate(theta, index):
        current = jax.tree.map(lambda x: x, batches)
        current.indices = jnp.array([index, index])
        obs = current.get_batched_position(Position({"y": y.value}), 0)
        state = model.update_state({"loc": theta, **obs}, model.state)
        return (
            current.scaled_log_lik(model, state, batch_index=0)
            + state["_model_log_prior"].value
        )

    def exact(theta):
        return (
            (scale / 2) * tfd.Normal(theta, 1.0).log_prob(jnp.array([1.0, 4.0])).sum()
            + tfd.Normal(theta, 2.0).log_prob(2.0)
            + tfd.Normal(0.0, 1.0).log_prob(theta)
        )

    theta = jnp.array(0.7, dtype=loc.value.dtype)
    values, gradients = jax.vmap(jax.value_and_grad(estimate), in_axes=(None, 0))(
        theta, jnp.arange(2)
    )
    expected_value, expected_gradient = jax.value_and_grad(exact)(theta)
    np.testing.assert_allclose(
        jnp.dot(jnp.array([0.25, 0.75]), values), expected_value, rtol=1e-6
    )
    np.testing.assert_allclose(
        jnp.dot(jnp.array([0.25, 0.75]), gradients), expected_gradient, rtol=1e-6
    )


@pytest.mark.parametrize(
    "shape,axis,event_dim",
    [
        ((4, 3), 0, 1),
        ((3, 4), 1, 0),
        ((2, 4, 3), 1, 1),
        ((2, 3, 4, 5), 2, 1),
        ((2, 4, 3), -2, 1),
        ((4,), 0, 0),
    ],
)
def test_likelihood_axes_follow_event_reduction_and_broadcasting(
    shape, axis, event_dim
):
    data = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 10
    if event_dim:
        dist = lsl.Dist(
            tfd.MultivariateNormalDiag,
            loc=jnp.zeros(shape[-1]),
            scale_diag=jnp.ones(shape[-1]),
        )
    else:
        # Extra leading distribution batch axis exercises broadcasting.
        loc = jnp.zeros((2, 1)) if len(shape) == 1 else 0.0
        dist = lsl.Dist(tfd.Normal, loc, 1.0)
    y = lsl.Var.new_obs(data, dist, name="y")
    model = lsl.Model([y])
    batches = Batches(
        ["y"],
        4,
        2,
        batch_axes={"y": axis},
        sample_with_replacement=True,
        sampling_weights=[1, 2, 3, 4],
    )
    batches.indices = jnp.array([3, 0, 1, 2])
    state = model.update_state(
        batches.get_batched_position(Position({"y": data}), 0), model.state
    )
    values = state["y_log_prob"].value
    likelihood_axis = axis % len(shape) + values.ndim - (len(shape) - event_dim)
    expected = 2 * jnp.sum(
        jnp.moveaxis(values, likelihood_axis, 0)
        * jnp.array([0.625, 2.5]).reshape((2,) + (1,) * (values.ndim - 1))
    )
    actual = jax.jit(lambda b: b.scaled_log_lik(model, state, batch_index=0))(batches)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(batches.correction_factors(0), [0.625, 2.5])


class LeadingReducedNormal(tfd.Normal):
    def _log_prob(self, x):
        return super()._log_prob(x).sum(axis=0)


def test_custom_leading_reduction_requires_explicit_likelihood_axis():
    y = lsl.Var.new_obs(
        jnp.arange(12.0).reshape(3, 4),
        lsl.Dist(LeadingReducedNormal, 0.0, 1.0),
        name="y",
    )
    model = lsl.Model([y])
    batches = Batches(
        ["y"],
        4,
        2,
        batch_axes={"y": 1},
        sample_with_replacement=True,
        sampling_weights=[1, 2, 3, 4],
    )
    state = model.update_state(
        batches.get_batched_position(Position({"y": y.value}), 0), model.state
    )
    with pytest.raises(ValueError, match="likelihood_axes"):
        batches.scaled_log_lik(model, state, batch_index=0)
    explicit = Batches(
        ["y"],
        4,
        2,
        likelihood_axes={"y": 0},
        batch_axes={"y": 1},
        sample_with_replacement=True,
        sampling_weights=[1, 2, 3, 4],
    )
    expected = 2 * jnp.dot(state["y_log_prob"].value, jnp.array([2.5, 1.25]))
    np.testing.assert_allclose(
        explicit.scaled_log_lik(model, state, batch_index=0), expected
    )


@pytest.mark.parametrize("per_obs", [True, False])
def test_reduced_scalar_likelihoods_rejected(per_obs):
    dist = (
        lsl.Dist(tfd.MultivariateNormalDiag, jnp.zeros(4), jnp.ones(4))
        if per_obs
        else lsl.Dist(tfd.Normal, 0.0, 1.0)
    )
    y = lsl.Var.new_obs(
        jnp.arange(4.0),
        dist,
        name="y",
    )
    assert y.dist_node is not None
    y.dist_node.per_obs = per_obs
    model = lsl.Model([y])
    batches = Batches(
        ["y"], 4, 4, sample_with_replacement=True, sampling_weights=[1, 2, 3, 4]
    )
    with pytest.raises(ValueError, match="pointwise"):
        batches.scaled_log_lik(model, model.state, batch_index=0)


def test_manager_mixes_independently_weighted_and_uniform_groups():
    variables = [
        lsl.Var.new_obs(jnp.arange(4.0) + i, lsl.Dist(tfd.Normal, 0.0, 1.0), name=name)
        for i, name in enumerate(["a", "b", "c", "d"])
    ]
    model = lsl.Model(variables)
    manager = BatchManager(
        [
            Batches(
                ["a"], 4, 2, sample_with_replacement=True, sampling_weights=[1, 2, 3, 4]
            ),
            Batches(
                ["b"], 4, 2, sample_with_replacement=True, sampling_weights=[4, 3, 2, 1]
            ),
            Batches(["c"], 4, 2),
        ],
        epoch_size=3,
    ).start_epoch(jax.random.key(72))
    for index in range(3):
        state = model.update_state(
            manager.get_batched_position(
                model.extract_position(["a", "b", "c"]), index
            ),
            model.state,
        )
        expected = state["d_log_prob"].value.sum()
        for child, weights in zip(
            manager.batches, [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]]
        ):
            probabilities = jnp.array(weights) / sum(weights)
            factors = 1 / (4 * probabilities[child.batch_indices[index]])
            expected += 2 * jnp.dot(
                factors, state[child.position_keys[0] + "_log_prob"].value
            )
        np.testing.assert_allclose(
            manager.scaled_log_lik(model, state, batch_index=index), expected
        )
    assert len(manager.correction_factors(0)) == 3
    np.testing.assert_array_equal(manager.correction_factors(0)[2], [1, 1])


def make_weighted_engine(*, epochs=3, weights=None, debug=False, loss_type=None):
    import optax

    from liesel.optim import (
        NegLogProbLoss,
        OptimEngine,
        Optimizer,
        PositionSplit,
        Stopper,
    )

    loc = lsl.Var.new_param(jnp.array(0.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="loc")
    y = lsl.Var.new_obs(jnp.arange(8.0), lsl.Dist(tfd.Normal, loc, 1.0), name="y")
    model = lsl.Model([y])
    split = PositionSplit.from_model(model, position_keys=["y"])
    loss = (loss_type or NegLogProbLoss)(model, split, scale=True)
    return OptimEngine(
        loss=loss,
        batches=Batches(
            ["y"],
            8,
            2,
            sample_with_replacement=True,
            sampling_weights=(
                jnp.arange(1.0, 9.0, dtype=jnp.float32) if weights is None else weights
            ),
        ),
        optimizers=[Optimizer(["loc"], optax.sgd(0.01))],
        stopper=Stopper(epochs=epochs, min_epochs=epochs, patience=epochs),
        seed=21,
        initial_state=model.state,
        loss_monitor="train_full_data",
        show_progress=False,
        debug_nans=debug,
    )


@pytest.mark.parametrize("debug", [False, True])
def test_engine_uses_current_batch_correction_and_keeps_full_data_monitoring(debug):
    from liesel.optim import NegLogProbLoss

    class ReferenceLoss(NegLogProbLoss):
        def loss_train_batched(self, params, carry):
            y = carry.batch["y"]
            probabilities = (y + 1) / 36.0
            log_lik = jnp.sum(
                tfd.Normal(params["loc"], 1.0).log_prob(y) / (2 * probabilities)
            )
            prior = tfd.Normal(0.0, 1.0).log_prob(params["loc"])
            return -(log_lik + prior) / 8

    expected = make_weighted_engine(loss_type=ReferenceLoss, debug=debug).fit()
    actual = make_weighted_engine(debug=debug).fit()
    np.testing.assert_allclose(
        actual.position_final["loc"], expected.position_final["loc"], rtol=1e-6
    )
    np.testing.assert_allclose(
        actual.history.loss_train, expected.history.loss_train, rtol=1e-6
    )
    np.testing.assert_allclose(
        actual.history.loss_monitor, expected.history.loss_monitor, rtol=1e-6
    )


@pytest.mark.parametrize("progress", ["off", "epochs", "batches"])
def test_checkpoint_restores_sampling_probabilities_over_new_weights(
    tmp_path, progress
):
    expected = make_weighted_engine().fit()
    path = tmp_path / "weighted.pkl"
    paused = make_weighted_engine().fit(checkpoint=path, pause_after=1)
    assert paused.status == "paused"
    restored = make_weighted_engine(weights=jnp.ones(8))
    restored.show_progress = progress != "off"
    restored.show_step_progress = progress == "batches"
    actual = restored.fit(checkpoint=path)
    for got, want in zip(
        jax.tree.leaves(actual.history), jax.tree.leaves(expected.history), strict=True
    ):
        np.testing.assert_allclose(got, want, rtol=1e-6)
    np.testing.assert_allclose(
        actual.position_final["loc"], expected.position_final["loc"], rtol=1e-6
    )
    assert actual.status == expected.status


@pytest.mark.parametrize("allow_version_mismatch", [False, True])
def test_pre_alias_weighted_checkpoint_is_rejected(tmp_path, allow_version_mismatch):
    path = tmp_path / "legacy-weighted.pkl"
    checkpoint = make_weighted_engine().fit(pause_after=1).checkpoint
    # Reproduce the serialized state from before alias tables were introduced.
    vars(checkpoint._carry.batches).pop("_alias_table", None)
    checkpoint.save(path)
    with pytest.raises(ValueError, match="predates alias sampling"):
        make_weighted_engine().fit(
            checkpoint=path, allow_version_mismatch=allow_version_mismatch
        )


@pytest.mark.parametrize("axis", [0, -2, 2])
def test_explicit_likelihood_axis_must_be_valid_and_have_batch_length(axis):
    y = lsl.Var.new_obs(jnp.ones((3, 4)), lsl.Dist(tfd.Normal, 0.0, 1.0), name="y")
    model = lsl.Model([y])
    batches = Batches(
        ["y"],
        4,
        2,
        batch_axes={"y": 1},
        likelihood_axes={"y": axis},
        sample_with_replacement=True,
        sampling_weights=[1, 2, 3, 4],
    )
    state = model.update_state(
        batches.get_batched_position(Position({"y": y.value}), 0), model.state
    )
    with pytest.raises(ValueError, match="likelihood"):
        batches.scaled_log_lik(model, state, batch_index=0)


def test_likelihood_axis_names_must_match_observed_variables_in_group():
    y = lsl.Var.new_obs(jnp.arange(4.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="y")
    model = lsl.Model([y])
    batches = Batches(
        ["y"],
        4,
        2,
        likelihood_axes={"typo": 0},
        sample_with_replacement=True,
        sampling_weights=[1, 2, 3, 4],
    )
    state = model.update_state(
        batches.get_batched_position(Position({"y": y.value}), 0), model.state
    )
    with pytest.raises(ValueError, match="likelihood_axes.*typo"):
        batches.scaled_log_lik(model, state, batch_index=0)


def test_weighted_correction_requires_batch_index_and_pointwise_model():
    y = lsl.Var.new_obs(jnp.arange(4.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="y")
    model = lsl.Model([y])
    batches = Batches(
        ["y"], 4, 2, sample_with_replacement=True, sampling_weights=[1, 2, 3, 4]
    )
    with pytest.raises(ValueError, match="batch_index"):
        batches.scaled_log_lik(model, model.state)
    with pytest.raises(TypeError, match="liesel.model.Model"):
        batches.scaled_log_lik(object(), model.state, batch_index=0)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("weight", [3e38, 1e300])
def test_uniform_weights_preserve_existing_scaling_and_large_weights_normalize(weight):
    y = lsl.Var.new_obs(jnp.arange(4.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="y")
    model = lsl.Model([y])
    plain = Batches(
        ["y"], 4, 2, sample_with_replacement=True, sample_size=12, batch_sample_size=2
    )
    weighted = Batches(
        ["y"],
        4,
        2,
        sample_with_replacement=True,
        sample_size=12,
        batch_sample_size=2,
        sampling_weights=[weight] * 4,
    )
    plain.indices = weighted.indices = jnp.array([3, 3, 0, 2])
    state = model.update_state(
        plain.get_batched_position(Position({"y": y.value}), 0), model.state
    )
    assert weighted.sampling_probabilities is not None
    np.testing.assert_allclose(weighted.sampling_probabilities, jnp.full(4, 0.25))
    np.testing.assert_allclose(
        weighted.scaled_log_lik(model, state, batch_index=0),
        plain.scaled_log_lik(model, state),
    )


def test_weighted_replacement_can_repeat_full_size_batch():
    batches = Batches(
        ["y"], 3, 3, sample_with_replacement=True, sampling_weights=[1, 2, 10000]
    )
    assert not batches.is_full_data
    sampled = batches.start_epoch(jax.random.key(3))
    np.testing.assert_array_equal(
        sampled.get_batched_position(Position({"y": jnp.arange(3)}), 0)["y"], [2, 2, 2]
    )


def test_event_axis_cannot_be_inferred_from_coincidentally_matching_sizes():
    y = lsl.Var.new_obs(
        jnp.ones((2, 2)),
        lsl.Dist(tfd.MultivariateNormalDiag, jnp.zeros(2), jnp.ones(2)),
        name="y",
    )
    model = lsl.Model([y])
    batches = Batches(
        ["y"],
        2,
        2,
        batch_axes={"y": 1},
        sample_with_replacement=True,
        sampling_weights=[1, 2],
    )
    with pytest.raises(ValueError, match="Cannot infer"):
        batches.scaled_log_lik(model, model.state, batch_index=0)


def test_weighted_correction_rejects_custom_aggregate_log_likelihood():
    y = lsl.Var.new_obs(jnp.arange(4.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="y")
    gb = lsl.GraphBuilder().add(y)
    gb.log_lik_node = lsl.Calc(
        lambda value: 2 * value.sum(), y.dist_node, _name="custom_log_lik"
    )
    model = gb.build_model()
    batches = Batches(
        ["y"], 4, 2, sample_with_replacement=True, sampling_weights=[1, 2, 3, 4]
    )
    state = model.update_state(
        batches.get_batched_position(Position({"y": y.value}), 0), model.state
    )
    with pytest.raises(ValueError, match="custom log_lik_node"):
        batches.scaled_log_lik(model, state, batch_index=0)
