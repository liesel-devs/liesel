"""Weighted sampling and correction through the public optimization API."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
from liesel.optim import Batches, BatchManager
from liesel.optim.types import Position


@pytest.mark.parametrize(
    "strength, expected",
    [
        (0.0, [1, 1, 1, 1, 1]),
        (0.5, [0.5, 1, 0.5, 0.5, 0.5]),
        (1.0, [0.25, 1, 0.25, 0.25, 0.25]),
    ],
)
def test_balanced_weights_preserve_row_order_and_control_category_mass(
    strength, expected
):
    weights = Batches.weights_balanced(["a", "b", "a", "a", "a"], strength=strength)
    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    np.testing.assert_allclose(weights, expected)


@pytest.mark.parametrize(
    "labels",
    [
        [1, 2, 1],
        [True, False, True],
        [1.5, 2.5, 1.5],
        np.array(["a", "b", "a"], dtype=object),
        jnp.array([1, 2, 1]),
        [2**60, 2**60 + 1, 2**60],
    ],
)
def test_balanced_weights_accept_discrete_labels_without_losing_identity(labels):
    np.testing.assert_allclose(Batches.weights_balanced(labels), [0.5, 1, 0.5])


@pytest.mark.parametrize(
    "labels",
    [
        [],
        "abc",
        [[1, 2]],
        [1, "1"],
        [1, None],
        [float("nan"), 1],
        [float("inf"), 1],
        [1j, 2j],
        np.array([1, "1"], dtype=object),
    ],
)
def test_balanced_weights_reject_invalid_labels(labels):
    with pytest.raises(ValueError, match="labels"):
        Batches.weights_balanced(labels)


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan"), float("inf"), [0.5]])
def test_balanced_weights_reject_invalid_strength(strength):
    with pytest.raises(ValueError, match="strength"):
        Batches.weights_balanced([1, 2], strength=strength)


def test_weights_for_shares_allocate_category_mass_and_allow_relative_shares():
    labels = ["a", "b", "a", "a"]
    weights = Batches.weights_for_shares(labels, {"a": 0.6, "b": 0.4})
    assert weights.dtype == np.float64
    np.testing.assert_allclose(weights, [0.2, 0.4, 0.2, 0.2])
    relative = Batches.weights_for_shares(labels, {"a": 60, "b": 40}, check_sum=False)
    np.testing.assert_allclose(relative, [20, 40, 20, 20])
    with pytest.raises(ValueError, match="sum"):
        Batches.weights_for_shares(labels, {"a": 60, "b": 40})
    Batches.weights_for_shares(labels, {"a": 0.6, "b": 0.4000005})
    with pytest.raises(ValueError, match="sum"):
        Batches.weights_for_shares(labels, {"a": 0.6, "b": 0.400002})


@pytest.mark.parametrize("check_sum", [True, False])
@pytest.mark.parametrize(
    "shares",
    [
        {"a": 1.0},
        {"a": 0.5, "b": 0.5, "c": 0.2},
        {"a": 1.0, "b": 0.0},
        {"a": 2.0, "b": -1.0},
        {"a": float("nan"), "b": 0.5},
        {"a": float("inf"), "b": 0.5},
        {"a": "0.5", "b": "0.5"},
        {"a": [0.5], "b": [0.5]},
    ],
)
def test_weights_for_shares_always_require_exact_coverage_and_positive_masses(
    shares, check_sum
):
    with pytest.raises(ValueError, match="shares"):
        Batches.weights_for_shares(["a", "b"], shares, check_sum=check_sum)


def test_weights_for_shares_reject_unrepresentable_output_and_handle_large_masses():
    with pytest.raises(ValueError, match="representable"):
        Batches.weights_for_shares(["a", "a", "b"], {"a": 5e-324, "b": 1.0})
    np.testing.assert_allclose(
        Batches.weights_for_shares([1, 2, 1], {1: 1e308, 2: 1e308}, check_sum=False),
        [5e307, 1e308, 5e307],
    )
    with pytest.raises(ValueError, match="sum"):
        Batches.weights_for_shares([1, 2], {1: 1e308, 2: 1e308})


def test_binned_weights_balance_occupied_intervals_and_include_final_edge():
    values = [4.0, 0.0, 1.0, 0.5, 0.25]
    # Equal-width bins [0, 1), [1, 2), [2, 3), [3, 4].
    full = Batches.weights_binned(values, bins=4, strength=1.0)
    assert full.dtype == np.float64
    np.testing.assert_allclose(full, [1, 1 / 3, 1, 1 / 3, 1 / 3])
    partial = Batches.weights_binned([0, 0, 0, 0, 10], bins=2)
    np.testing.assert_allclose(partial, [0.5, 0.5, 0.5, 0.5, 1])
    # Unequal widths still balance counts; 1 is in the second bin.
    np.testing.assert_allclose(
        Batches.weights_binned(values, bins=[0, 1, 4], strength=1.0),
        [0.5, 1 / 3, 0.5, 1 / 3, 1 / 3],
    )
    np.testing.assert_array_equal(
        Batches.weights_binned(values, bins=4, strength=0), np.ones(5)
    )


@pytest.mark.parametrize(
    "values,bins,expected",
    [
        ([7], 20, [1]),
        ([7, 7, 7, 7], 20, [0.5] * 4),
        ([1e308] * 4, 20, [0.5] * 4),
        ([7] * 4, [0, 7], [0.5] * 4),
        ([-10, -5, 0, 1], [-np.inf, 0, np.inf], [0.5] * 4),
    ],
)
def test_binned_weights_support_constant_data_and_open_ended_bins(
    values, bins, expected
):
    strength = 1 if bins == [-np.inf, 0, np.inf] else 0.5
    np.testing.assert_allclose(
        Batches.weights_binned(values, bins=bins, strength=strength), expected
    )


@pytest.mark.parametrize(
    "bins",
    [
        0,
        -2,
        1.5,
        True,
        "auto",
        [],
        [0],
        [0, 0, 2],
        [2, 0],
        [0, np.nan, 2],
        [-np.inf, np.inf, np.inf],
        [[0, 2]],
        [0, 1],
        [1, 2],
        ["0", "2"],
    ],
)
def test_binned_weights_reject_invalid_or_noncovering_bins(bins):
    with pytest.raises(ValueError, match="bins"):
        Batches.weights_binned([0, 2], bins=bins)


@pytest.mark.parametrize(
    "values",
    [[], 1, [[0, 1]], [np.nan, 1], [np.inf, 1], [1j, 2j], ["0", "1"], [None, 1]],
)
def test_binned_weights_require_finite_one_dimensional_values(values):
    with pytest.raises(ValueError, match="values"):
        Batches.weights_binned(values, bins=[-np.inf, np.inf])


def test_binned_weights_reject_degenerate_generated_edges():
    with pytest.raises(ValueError, match="bins"):
        Batches.weights_binned([1, np.nextafter(1.0, 2.0)], bins=10)
    with pytest.raises(ValueError, match="bins"):
        Batches.weights_binned([-1e308, 1e308], bins=10)


@pytest.mark.parametrize("strategy", ["balanced", "shares", "binned"])
def test_helper_weights_preserve_expected_objective_and_gradient(strategy):
    labels = ["rare", "common", "common", "common"]
    values = jnp.array([10.0, 0.0, 1.0, 2.0])
    if strategy == "balanced":
        weights = Batches.weights_balanced(labels)
    elif strategy == "shares":
        weights = Batches.weights_for_shares(labels, {"rare": 0.5, "common": 0.5})
    else:
        weights = Batches.weights_binned(values, bins=[0, 3, 10], strength=1)
    batches = Batches(
        ["y"], 4, 1, sample_with_replacement=True, sampling_weights=weights
    )
    probabilities = jnp.array([0.5, 1 / 6, 1 / 6, 1 / 6])
    assert batches.sampling_probabilities is not None
    np.testing.assert_allclose(batches.sampling_probabilities, probabilities)

    # The initial indices enumerate every possible draw, allowing an exact
    # expectation check without statistical sampling error.
    def estimate(theta, index):
        value = batches.get_batched_position(Position({"y": values}), index)["y"][0]
        return 4 * batches.correction_factors(index)[0] * (value - theta) ** 2

    losses, gradients = jax.jit(
        jax.vmap(jax.value_and_grad(estimate), in_axes=(None, 0))
    )(jnp.array(3.0), jnp.arange(4))
    # Squared errors: 49 + 9 + 4 + 1. Gradient: -14 + 6 + 4 + 2.
    np.testing.assert_allclose(jnp.dot(probabilities, losses), 63, rtol=1e-6)
    np.testing.assert_allclose(jnp.dot(probabilities, gradients), -2, rtol=1e-6)


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


@pytest.mark.parametrize("factory", [Batches.from_model, BatchManager.from_model])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("value_nodes", [False, True])
def test_model_factories_route_custom_likelihood_axes(factory, grouped, value_nodes):
    variables = [
        lsl.Var.new_obs(
            jnp.arange(3.0 * size).reshape(3, size),
            lsl.Dist(LeadingReducedNormal, 0.0, 1.0),
            name=name,
        )
        for name, size in ([("y", 4), ("z", 6)] if grouped else [("y", 4)])
    ]
    model = lsl.Model(variables)
    kwargs = {"multi_size": "manager"} if factory == Batches.from_model else {}
    axes = {var.name: 0 for var in variables}
    keys = [var.value_node.name if value_nodes else var.name for var in variables]

    def build(likelihood_axes):
        return factory(
            model,
            batch_size=2,
            position_keys=keys,
            default_batch_axis=1,
            sample_with_replacement=True,
            sampling_weights={
                key: jnp.arange(1, var.value.shape[1] + 1)
                for key, var in zip(keys, variables, strict=True)
            },
            likelihood_axes=likelihood_axes,
            **kwargs,
        )

    batches = build(axes)
    children = batches.batches if isinstance(batches, BatchManager) else (batches,)
    assert [child.likelihood_axes for child in children] == [
        {var.name: 0} for var in variables
    ]
    state = model.update_state(
        batches.get_batched_position(model.extract_position(keys), 0), model.state
    )
    expected = sum(
        child.batch_sample_scale
        * jnp.sum(state[f"{var.name}_log_prob"].value * child.correction_factors(0))
        for child, var in zip(children, variables, strict=True)
    )
    assert float(batches.scaled_log_lik(model, state, batch_index=0)) == pytest.approx(
        float(expected)
    )
    with pytest.raises(ValueError, match="likelihood_axes.*typo"):
        invalid = build({**axes, "typo": 0})
        invalid.scaled_log_lik(model, state, batch_index=0)


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


@pytest.mark.parametrize("constructor", ["batches", "manager", "init"])
@pytest.mark.parametrize("as_mapping", [False, True])
def test_from_model_sampling_weights(constructor, as_mapping):
    y = lsl.Var.new_obs(
        jnp.arange(12.0).reshape(2, 6),
        lsl.Dist(tfd.Normal, 0.0, 1.0),
        name="y",
    )
    model = lsl.Model([y])
    weights = np.arange(1.0, 7.0)
    supplied = {"y": weights} if as_mapping else weights
    expected = Batches.from_model(
        model, batch_size=2, batch_axes={"y": 1}, sample_with_replacement=True
    )
    if constructor == "init":
        result = BatchManager([expected], sampling_weights=supplied)
    else:
        cls = Batches if constructor == "batches" else BatchManager
        result = cls.from_model(
            model,
            batch_size=2,
            batch_axes={"y": 1},
            sample_with_replacement=True,
            sampling_weights=supplied,
        )
    child = result if isinstance(result, Batches) else result.batches[0]
    assert child.sampling_probabilities is not None
    np.testing.assert_allclose(child.sampling_probabilities, weights / weights.sum())
    assert child.sample_size == 12
    assert child.batch_sample_size == 4
    assert child.batch_axes == {"y": 1}
    result.start_epoch(jax.random.key(37))
    state = model.update_state(
        result.get_batched_position(model.extract_position(["y"]), 0), model.state
    )
    factors = child.correction_factors(0)
    expected_loss = 3 * (state["y_log_prob"].value * factors[None, :]).sum()
    np.testing.assert_allclose(
        result.scaled_log_lik(model, state, batch_index=0), expected_loss
    )
    # Pytree/JIT reconstruction must keep probabilities and alias tables.
    restored = jax.jit(lambda batches: batches)(result)
    for actual, expected_indices in zip(
        jax.tree.leaves(restored.permute_indices(jax.random.key(12))),
        jax.tree.leaves(result.permute_indices(jax.random.key(12))),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected_indices)


@pytest.mark.parametrize("constructor", ["batches", "manager", "init"])
@pytest.mark.parametrize("same_size", [False, True])
def test_grouped_sampling_weights(constructor, same_size):
    n_b = 4 if same_size else 6
    variables = [
        lsl.Var.new_obs(jnp.arange(float(n)), lsl.Dist(tfd.Normal, 0.0, 1.0), name=name)
        for name, n in [("a", 4), ("x", 4), ("b", n_b), ("c", 8)]
    ]
    model = lsl.Model(variables)
    groups = [["a", "x"], ["b"], ["c"]]
    weights = {"x": np.arange(1.0, 5.0), "b": np.arange(n_b, 0, -1.0)}
    expected = BatchManager(
        [
            Batches(
                group,
                n,
                2,
                sample_with_replacement=True,
                sampling_weights=weights.get(group[-1]),
            )
            for group, n in zip(groups, [4, n_b, 8], strict=True)
        ],
        epoch_size="max",
    ).start_epoch(jax.random.key(7))
    if constructor == "init":
        result = BatchManager(
            [
                Batches(group, n, 2, sample_with_replacement=True)
                for group, n in zip(groups, [4, n_b, 8], strict=True)
            ],
            epoch_size="max",
            sampling_weights=weights,
        )
    else:
        cls = Batches if constructor == "batches" else BatchManager
        extra = {"multi_size": "manager"} if constructor == "batches" else {}
        result = cls.from_model(
            model,
            batch_size=2,
            position_keys=groups if same_size else ["a", "x", "b", "c"],
            sample_with_replacement=True,
            sampling_weights=weights,
            **extra,
        )
    assert isinstance(result, BatchManager)
    result.start_epoch(jax.random.key(7))
    assert result.batches[2].sampling_probabilities is None
    for actual, reference in zip(result.batches, expected.batches, strict=True):
        np.testing.assert_array_equal(actual.batch_indices, reference.batch_indices)
    position = model.extract_position(["a", "x", "b", "c"])
    for i in range(result.n_full_batches):
        state = model.update_state(
            result.get_batched_position(position, i), model.state
        )
        np.testing.assert_allclose(
            result.scaled_log_lik(model, state, batch_index=i),
            expected.scaled_log_lik(model, state, batch_index=i),
        )


@pytest.mark.parametrize("constructor", ["batches", "manager", "init"])
@pytest.mark.parametrize(
    "weights, message",
    [
        ([1, 2, 3, 4], "Multiple batch groups"),
        ({"unknown": [1, 2, 3, 4]}, "Unknown sampling_weights"),
        ({"a": [1, 2, 3, 4], "x": [1, 2, 3, 4]}, "only one position key"),
        ({"a": [1, 2]}, "length axis_size"),
        ({"a": [1, 2, 0, 4]}, "finite, positive"),
        ({"a": [1, 2, float("nan"), 4]}, "finite, positive"),
        ({"a": None}, "must be weight vectors"),
    ],
)
def test_grouped_sampling_weights_validation(constructor, weights, message):
    groups = [["a", "x"], ["b"]]
    with pytest.raises(ValueError, match=message):
        if constructor == "init":
            BatchManager(
                [
                    Batches(group, 4, 2, sample_with_replacement=True)
                    for group in groups
                ],
                sampling_weights=weights,
            )
        else:
            model = lsl.Model(
                [
                    lsl.Var.new_obs(jnp.arange(4.0), name=name)
                    for name in ["a", "x", "b"]
                ]
            )
            cls = Batches if constructor == "batches" else BatchManager
            extra = {"multi_size": "manager"} if constructor == "batches" else {}
            cls.from_model(
                model,
                2,
                position_keys=groups,
                sample_with_replacement=True,
                sampling_weights=weights,
                **extra,
            )


@pytest.mark.parametrize("constructor", ["batches", "manager", "init"])
def test_factory_weights_require_replacement(constructor):
    with pytest.raises(ValueError, match="requires sample_with_replacement=True"):
        if constructor == "init":
            BatchManager([Batches(["y"], 4, 2)], sampling_weights=[1, 2, 3, 4])
        else:
            model = lsl.Model([lsl.Var.new_obs(jnp.arange(4.0), name="y")])
            cls = Batches if constructor == "batches" else BatchManager
            cls.from_model(model, 2, sampling_weights=[1, 2, 3, 4])


def test_manager_weight_overrides_preserve_originals_and_unspecified_children():
    a = Batches(
        ["a"], 4, 2, sample_with_replacement=True, sampling_weights=[4, 3, 2, 1]
    )
    b = Batches(
        ["b"], 4, 2, sample_with_replacement=True, sampling_weights=[1, 3, 1, 3]
    )
    c = Batches(["c"], 4, 2)
    manager = BatchManager([a, b, c], sampling_weights={"a": [1, 2, 3, 4]})
    assert a.sampling_probabilities is not None
    assert manager.batches[0].sampling_probabilities is not None
    np.testing.assert_allclose(a.sampling_probabilities, [0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(
        manager.batches[0].sampling_probabilities, [0.1, 0.2, 0.3, 0.4]
    )
    assert manager.batches[1] is not b
    assert manager.batches[2] is not c
    np.testing.assert_array_equal(
        manager.batches[1].sampling_probabilities, b.sampling_probabilities
    )
    np.testing.assert_array_equal(manager.batches[2].indices, c.indices)
