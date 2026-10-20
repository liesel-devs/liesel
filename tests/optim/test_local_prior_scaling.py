"""Local-prior minibatches must estimate the same posterior as full data."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
from liesel.goose import LieselInterface
from liesel.optim._log_lik import log_prob_sources
from liesel.optim.types import Position


class TermsAt(tfd.Normal):
    def __init__(self, idx, loc=0.0):
        super().__init__(loc, 1.0)
        self.idx = idx

    def _log_prob(self, x):
        return super()._log_prob(x[self.idx])


def local_model(n=6, *, likelihood=True, indexed=True, global_prior=True):
    idx = lsl.Var.new_obs(jnp.arange(n), name="idx")
    # An indirect input exercises traversal beyond the distribution's parents.
    indices = lsl.Var.new_calc(lambda x: x, idx, name="indices")
    prior = lsl.Dist(TermsAt, indices) if indexed else lsl.Dist(tfd.Normal, 0.0, 1.0)
    z = lsl.Var.new_param(jnp.linspace(-0.5, 0.5, n), prior, name="z")
    variables = [idx, z]
    if likelihood:
        mean = lsl.Var.new_calc(lambda z, i: z[i], z, idx)
        variables.append(
            lsl.Var.new_obs(
                jnp.arange(n, dtype=float), lsl.Dist(tfd.Normal, mean, 1.0), name="y"
            )
        )
    if global_prior:
        variables.append(
            lsl.Var.new_param(0.7, lsl.Dist(tfd.Normal, 0.0, 1.0), name="global")
        )
    return lsl.Model(variables)


def carry(model, **kwargs):
    return SimpleNamespace(
        model_state=model.state, fixed_position=Position({}), **kwargs
    )


def assert_tree_close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_batch_objective_and_gradients_recover_full_posterior(indexed, weighted):
    model = local_model(indexed=indexed)
    split = opt.PositionSplit.from_model(model, shuffle=False)
    batches = opt.Batches.from_split(
        split,
        batch_size=1,
        sample_with_replacement=weighted,
        sampling_weights=jnp.arange(1.0, 7.0) if weighted else None,
    )
    assert isinstance(batches, opt.Batches)
    batches.indices = jnp.arange(6)
    loss = opt.NegLogProbLoss(model, split)
    params = model.extract_position(list(model.parameters))
    original = model.state

    def average(position):
        values = []
        for i in range(6):
            batch = batches.get_batched_position(split.train, i)
            values.append(
                loss.loss_train_batched(
                    position, carry(model, batch=batch, batches=batches, i_batch=i)
                )
            )
        probabilities = batches.sampling_probabilities if weighted else jnp.ones(6) / 6
        assert probabilities is not None
        return jnp.dot(probabilities, jnp.stack(values))

    def full(position):
        return -model.update_state(position, original)["_model_log_prob"].value

    assert_tree_close(
        jax.jit(jax.value_and_grad(average))(params), jax.value_and_grad(full)(params)
    )
    mode = Position(params | {"z": model.vars["y"].value / 2})
    np.testing.assert_allclose(jax.grad(average)(mode)["z"], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        jax.hessian(average)(mode)["z"]["z"], 2 * jnp.eye(6), atol=1e-6
    )
    assert_tree_close(model.state, original)


@pytest.mark.parametrize("manager", [False, True])
@pytest.mark.parametrize("part", ["validate", "test"])
@pytest.mark.parametrize("strategy", ["log_lik", "log_prob"])
def test_holdout_with_local_priors_requires_custom_loss(manager, part, strategy):
    model = local_model(n=8)
    split = opt.PositionSplit.from_model(
        model,
        shuffle=False,
        validate_axis_share=0.25 if part == "validate" else 0.0,
        test_axis_share=0.25 if part == "test" else 0.0,
    )
    assert isinstance(split, opt.PositionSplit)
    if manager:
        split = opt.PositionSplitManager([split])
    with pytest.raises(ValueError, match="held-out.*custom Loss"):
        opt.LieselOptim(
            model,
            split=split,
            optimizers=optax.adam(0.05),
            validation_strategy=strategy,
            loss_monitor="train_full_data",
        )


@pytest.mark.parametrize("manager", [False, True])
@pytest.mark.parametrize("part", ["validate", "test"])
def test_split_helpers_score_inferred_heldout_latents(manager, part):
    model = local_model(n=8)
    split = opt.PositionSplit.from_model(
        model,
        shuffle=False,
        validate_axis_share=0.25 if part == "validate" else 0.0,
        test_axis_share=0.25 if part == "test" else 0.0,
    )
    assert isinstance(split, opt.PositionSplit)
    if manager:
        split = opt.PositionSplitManager([split])
    heldout = getattr(split, part)
    # Supply the conditional modes, as a custom loss can before scoring.
    z = model.vars["y"].value / 2
    state = model.update_state(Position(heldout | {"z": z}), model.state)

    def score(state):
        return (
            split.scaled_log_lik(model, state, part=part),
            split.scaled_log_prior(model, state, part=part),
        )

    expected_lik = 3 * tfd.Normal(z[heldout["idx"]], 1.0).log_prob(heldout["y"]).sum()
    expected_prior = 3 * tfd.Normal(0.0, 1.0).log_prob(z[heldout["idx"]]).sum()
    expected_prior += tfd.Normal(0.0, 1.0).log_prob(model.vars["global"].value)
    assert_tree_close(jax.jit(score)(state), (expected_lik, expected_prior))


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("shape", [(), (3,)])
def test_uniform_batch_rejects_indirect_nonpointwise_factors(observed, shape):
    data = lsl.Var.new_obs(jnp.arange(6.0), name="data")
    scale = lsl.Var.new_calc(lambda x: 1.0 + 2.5 * jnp.std(x), data)
    factory = lsl.Var.new_obs if observed else lsl.Var.new_param
    mu = factory(jnp.zeros(shape), lsl.Dist(tfd.Normal, 0.0, scale), name="mu")
    model = lsl.Model(mu)
    batches = opt.Batches(["data"], axis_size=6, batch_size=2, shuffle=False)
    state = model.update_state(
        batches.extract_batched_position(model, model.state, 0), model.state
    )
    method = batches.scaled_log_lik if observed else batches.scaled_log_prior
    with pytest.raises(ValueError, match="mu.*pointwise.*custom Loss"):
        method(model, state)
    full = opt.Batches(["data"], axis_size=6, batch_size=None)
    method = full.scaled_log_lik if observed else full.scaled_log_prior
    assert_tree_close(method(model, model.state), model.log_prob)


@pytest.mark.parametrize("factory", [opt.Batches, opt.BatchManager])
def test_prior_only_group_and_full_data(factory):
    model = local_model(likelihood=False)
    split = opt.PositionSplit.from_model(model)
    assert split.train_sample_size == 6
    for batch_size in (2, None):
        batches = factory.from_model(model, batch_size=batch_size, shuffle=False)
        data = batches.extract_batched_position(model, model.state, 0)
        state = model.update_state(data, model.state)
        scale = 3 if batch_size else 1
        expected = scale * state["z_log_prob"].value.sum()
        expected += state["global_log_prob"].value
        assert_tree_close(batches.scaled_log_prior(model, state), expected)
        assert_tree_close(batches.scaled_log_lik(model, state), 0.0)


def test_node_keys_and_distribution_only_observed_dependency():
    model = local_model()
    key = model.vars["idx"].value_node.name
    assert log_prob_sources(model, [key]) == {"y": [key], "z": [key]}
    assert log_prob_sources(model, ["idx"]) == {"y": ["idx"], "z": ["idx"]}


def test_cross_group_priors_rejected_in_factories_losses_and_manual_batches():
    idx = lsl.Var.new_obs(jnp.arange(4), name="idx")
    loc = lsl.Var.new_obs(jnp.zeros(4), name="loc")
    z = lsl.Var.new_param(jnp.zeros(4), lsl.Dist(TermsAt, idx, loc), name="z")
    model = lsl.Model(z)
    with pytest.raises(ValueError, match="factor 'z'.*group.*idx.*loc"):
        opt.PositionSplit.from_model(model, position_keys=[["idx"], ["loc"]])
    splits = opt.PositionSplitManager(
        [
            opt.Split([key], axis_size=4).split_position(model.extract_position([key]))
            for key in ("idx", "loc")
        ]
    )
    with pytest.raises(ValueError, match="factor 'z'"):
        opt.NegLogProbLoss(model, splits)
    split = opt.PositionSplit.from_model(model)
    batches = opt.BatchManager(
        [opt.Batches([key], axis_size=4, batch_size=2) for key in ("idx", "loc")]
    )
    with pytest.raises(ValueError, match="factor 'z'"):
        opt.LieselOptim(
            model,
            split=split,
            batches=batches,
            optimizers=optax.sgd(0.01),
            loss_monitor="train_full_data",
        ).build_engine()
    with pytest.raises(ValueError, match="factor 'z'"):
        batches.scaled_log_prior(model, model.state)


def test_custom_prior_requires_custom_loss_only_when_covered():
    model = local_model()
    builder = lsl.GraphBuilder().add(*model.copy_vars().values())
    builder.log_prior_node = lsl.Calc(lambda: jnp.array(7.0))
    custom = builder.build_model()
    batches = opt.Batches(["idx", "y"], axis_size=6, batch_size=2)
    state = custom.update_state(
        batches.extract_batched_position(custom, custom.state, 0), custom.state
    )
    with pytest.raises(ValueError, match="custom log_prior_node"):
        batches.scaled_log_prior(custom, state)
    full = opt.Batches([], axis_size=6, batch_size=None)
    assert_tree_close(full.scaled_log_prior(custom, custom.state), 7.0)


def test_custom_prior_reading_batch_without_parameter_factor_is_rejected():
    idx = lsl.Var.new_obs(jnp.arange(6), name="idx")
    z = lsl.Var.new_param(jnp.zeros(6), name="z")
    builder = lsl.GraphBuilder().add(idx, z)
    builder.log_prior_node = lsl.Calc(
        lambda z, i: tfd.Normal(0.0, 1.0).log_prob(z[i]).sum(), z, idx
    )
    model = builder.build_model()
    batches = opt.Batches(["idx"], axis_size=6, batch_size=2)
    with pytest.raises(ValueError, match="custom log_prior_node"):
        batches.scaled_log_prior(model, model.state)


def test_generic_interfaces_keep_aggregate_prior():
    model = local_model()
    batches = opt.Batches(["y"], axis_size=6, batch_size=2)
    split = opt.PositionSplit.from_model(model)
    assert isinstance(split, opt.PositionSplit)
    for config in (
        batches,
        opt.BatchManager([batches]),
        split,
        opt.PositionSplitManager([split]),
    ):
        assert_tree_close(
            config.scaled_log_prior(LieselInterface(model), model.state),
            model.log_prior,
        )


class ReplicatedTermsAt(TermsAt):
    def _log_prob(self, x):
        return tfd.Normal(0.0, 1.0).log_prob(x[:, self.idx])


@pytest.mark.parametrize("from_split", [False, True])
def test_weighted_manager_scales_unequal_prior_groups_and_routes_axes(from_split):
    variables = [lsl.Var.new_param(0.7, lsl.Dist(tfd.Normal, 0.0, 1.0), name="g")]
    for suffix, n in [("a", 4), ("b", 6)]:
        idx = lsl.Var.new_obs(jnp.arange(n), name=f"idx_{suffix}")
        variables.append(
            lsl.Var.new_param(
                jnp.arange(2 * n, dtype=float).reshape(2, n) / n,
                lsl.Dist(ReplicatedTermsAt, idx),
                name=f"z_{suffix}",
            )
        )
    model = lsl.Model(variables)
    split = opt.PositionSplit.from_model(model, multi_size="manager")
    kwargs: dict[str, Any] = {
        "batch_size": 2,
        "sample_with_replacement": True,
        "sampling_weights": {
            "idx_a": jnp.arange(1.0, 5.0),
            "idx_b": jnp.arange(1.0, 7.0),
        },
        "likelihood_axes": {"z_a": 1, "z_b": 1},
    }
    batches = (
        opt.Batches.from_split(split, **kwargs)
        if from_split
        else opt.BatchManager.from_model(model, **kwargs)
    ).start_epoch(jax.random.key(4))
    assert isinstance(batches, opt.BatchManager)
    for i in range(batches.n_full_batches):
        state = model.update_state(
            batches.get_batched_position(split.train, i), model.state
        )
        expected = state["g_log_prob"].value
        for child in batches.batches:
            name = "z_" + child.position_keys[0][-1]
            expected += child.batch_sample_scale * jnp.sum(
                state[name + "_log_prob"].value * child.correction_factors(i)
            )
        assert_tree_close(
            jax.jit(
                lambda b, s, index: b.scaled_log_prior(model, s, batch_index=index)
            )(batches, state, i),
            expected,
        )


@pytest.mark.parametrize("problem", ["ambiguous", "scalar", "axis", "length"])
def test_weighted_prior_rejects_missing_pointwise_axis(problem):
    idx = lsl.Var.new_obs(jnp.arange(6), name="idx")
    prior = lsl.Dist(ReplicatedTermsAt, idx)
    z = lsl.Var.new_param(jnp.zeros((3, 6)), prior, name="z")
    if problem == "scalar":
        prior.per_obs = False
    model = lsl.Model(z)
    axes = {"axis": {"z": 2}, "length": {"z": 0}}.get(problem, {})
    batches = opt.Batches.from_model(
        model,
        batch_size=2,
        sample_with_replacement=True,
        sampling_weights=jnp.arange(1.0, 7.0),
        likelihood_axes=axes,
    ).start_epoch(jax.random.key(4))
    state = model.update_state(
        batches.extract_batched_position(model, model.state, 0), model.state
    )
    message = {
        "ambiguous": "Set likelihood_axes",
        "scalar": "pointwise",
        "axis": "invalid likelihood_axes",
        "length": "batch_size entries",
    }[problem]
    with pytest.raises(ValueError, match=message):
        batches.scaled_log_prior(model, state, batch_index=0)
