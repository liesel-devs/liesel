"""Explicit observation groups through public split and batch factories."""

from types import SimpleNamespace
from typing import assert_type

import jax
import jax.numpy as jnp
import pytest

import liesel.model as lsl
import liesel.optim.split as split_module
from liesel.optim import (
    Batches,
    BatchManager,
    EmaTrainLossMonitor,
    LieselOptim,
    PositionSplit,
    PositionSplitManager,
    Split,
    SplitManager,
)


def _model(*, shared=False):
    a = jnp.arange(32.0)
    b = a + 100
    variables = [
        lsl.Var.new_obs(jnp.stack([a, a + 10]), name="x_a"),
        lsl.Var.new_obs(a, name="y_a"),
        lsl.Var.new_obs(b[:, None], name="x_b"),
        lsl.Var.new_obs(b, name="y_b"),
    ]
    if shared:
        variables.append(lsl.Var.new_obs(jnp.array(7.0), name="shared"))
    return lsl.Model(variables)


def test_split_recipe_infers_multiple_groups_and_materializes_them():
    model = lsl.Model(
        [
            lsl.Var.new_obs(jnp.arange(12.0), name="a"),
            lsl.Var.new_obs(jnp.arange(8.0), name="b"),
        ]
    )
    recipe = Split.from_model(
        model, validate_axis_share=0.25, shuffle=False, multi_size="manager"
    )
    assert isinstance(recipe, SplitManager)
    data = recipe.split_position(model.extract_position(recipe.position_keys))
    assert data.train["a"].tolist() == list(range(9))
    assert data.train["b"].tolist() == list(range(6))
    batches = Batches.from_split(data, batch_size=3)
    assert isinstance(batches, BatchManager)
    assert {tuple(child.position_keys) for child in batches.batches} == {("a",), ("b",)}


@pytest.mark.parametrize("factory", [SplitManager, Split])
def test_explicit_split_groups_preserve_alignment_and_equal_size_boundaries(factory):
    model = _model()
    groups = [["x_b", "y_b"], ["x_a", "y_a"]]
    kwargs = {"multi_size": "manager"} if factory is Split else {}
    manager = factory.from_model(
        model,
        position_keys=groups,
        split_axes={"x_a": 1},
        validate_axis_share=0.25,
        test_axis_share=0.125,
        shuffle=True,
        seed=42,
        **kwargs,
    )
    split = manager.split_position(model.extract_position(manager.position_keys))

    assert [child.position_keys for child in manager.splits] == groups
    assert split.train_axis_sizes == (20, 20)
    for part in (split.train, split.validate, split.test):
        assert jnp.array_equal(part["x_a"][0], part["y_a"])
        assert jnp.array_equal(part["x_b"][:, 0], part["y_b"])
    assert not jnp.array_equal(split.train["y_a"], split.train["y_b"] - 100)


@pytest.mark.parametrize("factory", [SplitManager, BatchManager, Split])
@pytest.mark.parametrize(
    "keys, error, message",
    [
        (["y_a", ["y_b"]], TypeError, "flat.*nested"),
        ([["y_a"], []], ValueError, "empty"),
        ([["y_a", 1]], TypeError, "strings"),
        ([["y_a"], ["y_a"]], ValueError, "Duplicate"),
        ([["y_a", "y_a"]], ValueError, "Duplicate"),
        ([["x_a", "y_a"]], ValueError, "axis lengths"),
    ],
)
def test_explicit_groups_reject_invalid_keys(factory, keys, error, message):
    kwargs = {"batch_size": 4} if factory is BatchManager else {}
    with pytest.raises(error, match=message):
        factory.from_model(_model(), position_keys=keys, **kwargs)


def test_single_split_factory_accepts_one_explicit_group_by_default():
    recipe = Split.from_model(_model(), position_keys=[["y_a"]])
    assert isinstance(recipe, Split)
    assert recipe.position_keys == ["y_a"]


def test_factory_overloads_preserve_scalar_defaults_and_positional_calls():
    model = lsl.Model([lsl.Var.new_obs(jnp.arange(4.0), name="y")])
    assert_type(Split.from_model(model), Split)
    assert_type(Batches.from_model(model, 2), Batches)
    assert_type(Split.from_model(model, multi_size="manager"), Split | SplitManager)
    assert_type(
        Batches.from_model(model, 2, multi_size="manager"), Batches | BatchManager
    )
    recipe = assert_type(
        Split.from_model(
            model, None, None, 0.0, 0.0, None, 0, True, 0, None, "manager"
        ),
        Split | SplitManager,
    )
    batches = assert_type(
        Batches.from_model(model, 2, None, None, True, None, 0, "manager"),
        Batches | BatchManager,
    )
    assert isinstance(recipe, Split)
    assert isinstance(batches, Batches)


@pytest.mark.parametrize("factory", [PositionSplit, PositionSplitManager, Split])
def test_groups_and_passthrough_survive_position_splits_and_explicit_batches(factory):
    model = _model(shared=True)
    kwargs = {"multi_size": "manager"} if factory in (PositionSplit, Split) else {}
    split = factory.from_model(
        model,
        position_keys=[["x_a", "y_a", "shared"], ["x_b", "y_b"]],
        split_axes={"x_a": 1, "shared": None},
        validate_axis_share=0.25,
        shuffle=True,
        seed=42,
        **kwargs,
    )
    if isinstance(split, SplitManager):
        split = split.split_position(model.extract_position(split.position_keys))
    assert isinstance(split, PositionSplitManager)
    for part in (split.train, split.validate, split.test):
        assert part["shared"] == 7
    quick = LieselOptim(
        model,
        split=split,
        optimizers=[],
        batches=Batches.from_split(split, batch_size=4, batch_axes={"x_a": 1}),
        loss_monitor=EmaTrainLossMonitor(1),
        seed=42,
    )
    batches = quick.batches
    assert isinstance(batches, BatchManager)
    assert [child.position_keys for child in batches.batches] == [
        ["x_a", "y_a"],
        ["x_b", "y_b"],
    ]
    started = jax.jit(lambda b: b.start_epoch(jax.random.key(42)))(batches)
    batch = started.get_batched_position(split.train, batch_index=0)
    assert "shared" not in batch
    assert jnp.array_equal(batch["x_a"][0], batch["y_a"])
    assert jnp.array_equal(batch["x_b"][:, 0], batch["y_b"])


@pytest.mark.parametrize("factory", [Batches, BatchManager])
def test_batch_factories_preserve_explicit_groups_under_jit(factory):
    model = _model()
    kwargs = {"multi_size": "manager"} if factory is Batches else {}
    groups = [["x_a", "y_a"], ["x_b", "y_b"]]
    batches = factory.from_model(
        model, position_keys=groups, batch_axes={"x_a": 1}, batch_size=4, **kwargs
    )
    assert isinstance(batches, BatchManager)
    assert [child.position_keys for child in batches.batches] == groups
    started = jax.jit(lambda b: b.start_epoch(jax.random.key(12)))(batches)
    data = model.extract_position(batches.position_keys)
    batch = started.get_batched_position(data, batch_index=0)
    assert jnp.array_equal(batch["x_a"][0], batch["y_a"])
    assert jnp.array_equal(batch["x_b"][:, 0], batch["y_b"])


@pytest.mark.parametrize("keys", [[["y_a", "shared"]], [["y_a"], ["shared"]]])
def test_passthrough_requires_a_real_group(keys):
    if len(keys) == 1:
        manager = SplitManager.from_model(
            _model(shared=True), position_keys=keys, split_axes={"shared": None}
        )
        assert tuple(manager.passthrough_position_keys) == ("shared",)
        assert manager.splits[0].position_keys == ["y_a"]
    else:
        with pytest.raises(ValueError, match="passthrough"):
            SplitManager.from_model(
                _model(shared=True), position_keys=keys, split_axes={"shared": None}
            )


@pytest.mark.parametrize("factory", [Batches, PositionSplit, Split])
def test_single_group_return_types_and_multiple_group_opt_in(factory):
    model = _model()
    kwargs = {"batch_size": 4} if factory is Batches else {}
    for keys in (["y_a", "y_b"], [["y_a", "y_b"]]):
        single = factory.from_model(
            model, position_keys=keys, multi_size="manager", **kwargs
        )
        assert isinstance(single, factory)
        assert single.position_keys == ["y_a", "y_b"]
    with pytest.raises(ValueError, match="multi_size"):
        factory.from_model(model, position_keys=[["y_a"], ["y_b"]], **kwargs)


@pytest.mark.parametrize("factory", [SplitManager, BatchManager])
@pytest.mark.parametrize("keys", [None, ["x_a", "y_a", "x_b", "y_b"]])
def test_flat_and_omitted_keys_keep_automatic_grouping(factory, keys):
    if factory is BatchManager:
        manager = factory.from_model(
            _model(), position_keys=keys, batch_axes={"x_a": 1}, batch_size=4
        )
        assert len(manager.batches) == 1
    else:
        manager = factory.from_model(
            _model(), position_keys=keys, split_axes={"x_a": 1}
        )
        assert len(manager.splits) == 1


@pytest.mark.parametrize("option", ["axis_size", "sample_size", "batch_sample_size"])
def test_multiple_equal_size_groups_reject_scalar_batch_overrides(option):
    with pytest.raises(ValueError, match="cannot configure multiple"):
        Batches.from_model(
            _model(),
            batch_size=4,
            position_keys=[["y_a"], ["y_b"]],
            multi_size="manager",
            axis_size=32 if option == "axis_size" else None,
            sample_size=32 if option == "sample_size" else None,
            batch_sample_size=32 if option == "batch_sample_size" else None,
        )


def test_multiple_equal_size_groups_reject_scalar_split_axis_override():
    with pytest.raises(ValueError, match="cannot configure multiple"):
        PositionSplit.from_model(
            _model(),
            position_keys=[["y_a"], ["y_b"]],
            multi_size="manager",
            axis_size=32,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"multi_size": "invalid"}, "multi_size must"),
        ({"multi_size": "manager", "axis_size": 32}, "axis_size"),
        (
            {"multi_size": "manager", "sample_sizes": {"train": 64}},
            "sample_sizes",
        ),
    ],
)
def test_split_recipe_rejects_invalid_mode_and_ambiguous_group_overrides(
    kwargs, message
):
    with pytest.raises(ValueError, match=message):
        Split.from_model(_model(), position_keys=[["y_a"], ["y_b"]], **kwargs)


SPLIT_FACTORIES = [
    Split,
    Split.from_axis_shares,
    Split.from_model,
    PositionSplit.from_model,
    SplitManager.from_model,
    PositionSplitManager.from_model,
]


def _split_via(factory, *, share=0.25, **kwargs):
    model = _model()
    if factory is Split:
        result = factory(
            ["y_a"], axis_size=32, validate_axis_size=int(32 * share), **kwargs
        )
    elif factory == Split.from_axis_shares:
        result = factory(["y_a"], axis_size=32, validate_axis_share=share, **kwargs)
    else:
        result = factory(
            model, position_keys=["y_a"], validate_axis_share=share, **kwargs
        )
    if isinstance(result, (Split, SplitManager)):
        result = result.split_position(model.extract_position(result.position_keys))
    return result


@pytest.mark.parametrize("factory", SPLIT_FACTORIES)
def test_split_factories_shuffle_holdouts_by_default_with_reproducible_seed(factory):
    default = _split_via(factory, seed=42)
    explicit = _split_via(factory, shuffle=True, seed=42)
    ordered = _split_via(factory, shuffle=False)
    assert jnp.array_equal(default.train["y_a"], explicit.train["y_a"])
    assert not jnp.array_equal(default.train["y_a"], ordered.train["y_a"])
    assert ordered.train["y_a"].tolist() == list(range(24))


@pytest.mark.parametrize("factory", SPLIT_FACTORIES)
@pytest.mark.parametrize("share", [0.0, 0.01])
def test_full_data_split_preserves_order_without_generating_seed(
    factory, share, monkeypatch
):
    def unexpected_clock_read():
        raise AssertionError("Full-data splitting must not generate a seed")

    monkeypatch.setattr(
        split_module, "time", SimpleNamespace(time=unexpected_clock_read)
    )
    split = _split_via(factory, share=share, shuffle=True)
    assert split.train["y_a"].tolist() == list(range(32))


def test_full_data_reserved_rows_do_not_reorder_observations():
    split = Split(["y_a"], axis_size=32, shuffle=False, keep_in_train=[31])
    assert split.indices_train.tolist() == list(range(32))
