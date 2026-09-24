import math

import jax
import jax.numpy as jnp
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim.split as split_module
from liesel.optim import (
    Batches,
    BatchManager,
    EmaTrainLossMonitor,
    LieselOptim,
    NegLogProbLoss,
    PositionSplit,
    PositionSplitManager,
    Split,
    SplitManager,
)
from liesel.optim.state import OptimCarry
from liesel.optim.types import Position


def _two_branch_model():
    loc = lsl.Var.new_param(0.0, name="loc")
    y1 = lsl.Var.new_obs(
        jnp.arange(10.0),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y1",
    )
    y2 = lsl.Var.new_obs(
        jnp.arange(6.0),
        lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y2",
    )
    return lsl.Model([y1, y2]), y1, y2


def _matrix_obs_model(shape=(4, 8)):
    y = lsl.Var.new_obs(
        jnp.arange(math.prod(shape), dtype=float).reshape(shape),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="y",
    )
    return lsl.Model([y]), y


class TestSplit:
    def test_inferred_key_recipe_is_reusable_with_passthrough(self):
        recipe = Split(axis_size=4, validate_axis_size=1, split_axes={"tab": None})
        tab = jnp.arange(2)
        for name in ("x", "y"):
            split = recipe.split_position(Position({name: jnp.arange(4), "tab": tab}))
            assert split.split_position_keys == [name]
            assert split.train[name].size == 3
            assert jnp.array_equal(split.validate["tab"], tab)
        assert recipe.position_keys is None

    @pytest.mark.parametrize("share, expected", [(0.29, 29), (0.289, 28), (0.0, 0)])
    def test_share_counts_tolerate_roundoff(self, share, expected):
        for part in ("validate", "test"):
            recipe = Split.from_axis_shares(
                ["x"], axis_size=100, **{f"{part}_axis_share": share}
            )
            assert getattr(recipe, f"{part}_axis_size") == expected
            assert recipe.train_axis_size == 100 - expected

    def test_share_counts_and_grouped_holdout_detection_agree(self):
        model = lsl.Model(
            [
                lsl.Var.new_obs(jnp.arange(49), name="x"),
                lsl.Var.new_obs(jnp.arange(49), name="y"),
            ]
        )
        # This product is one ULP below 1.0, but both groups need a holdout.
        manager = SplitManager.from_model(
            model, position_keys=[["x"], ["y"]], validate_axis_share=1 / 49, seed=7
        )
        assert manager.validate_axis_sizes == (1, 1)
        assert not jnp.array_equal(manager.splits[0].indices, manager.splits[1].indices)

    def test_split_position_keeps_none_axis_keys_unchanged(self):
        response = jnp.arange(24.0).reshape(4, 6)
        land = jnp.arange(6.0).reshape(6, 1)
        splitter = Split(
            ["response", "land"],
            axis_size=4,
            validate_axis_size=1,
            test_axis_size=1,
            split_axes={"response": 0, "land": None},
        )

        split = splitter.split_position(Position({"response": response, "land": land}))

        assert split.train["response"].shape == (2, 6)
        assert split.validate["response"].shape == (1, 6)
        assert split.test["response"].shape == (1, 6)
        assert split.position_keys == ["response", "land"]
        assert split.split_position_keys == ["response"]
        for part in (split.train, split.validate, split.test):
            assert jnp.array_equal(part["land"], land)

    def test_split_requires_at_least_one_non_passthrough_key(self):
        with pytest.raises(ValueError, match="at least one.*split"):
            Split(["land"], axis_size=4, split_axes={"land": None})

    def test_split_position_infers_position_keys_when_omitted(self):
        splitter = Split(axis_size=4, validate_axis_size=1, shuffle=False)

        split = splitter.split_position(
            Position({"x": jnp.arange(4), "y": jnp.arange(4) + 10})
        )

        assert split.position_keys == ["x", "y"]
        assert split.train["x"].tolist() == [0, 1, 2]
        assert split.validate["y"].tolist() == [13]

    def test_keep_in_train_reserves_rows_without_changing_split_sizes(self):
        splitter = Split(
            axis_size=10,
            validate_axis_size=2,
            test_axis_size=2,
            keep_in_train=[8, 9],
            shuffle=True,
            seed=1,
        )

        assert {8, 9}.issubset(set(splitter.indices_train.tolist()))
        assert splitter.indices_train.size == 6
        assert splitter.indices_validate.size == 2
        assert splitter.indices_test.size == 2
        assert sorted(splitter.indices.tolist()) == list(range(10))

    def test_keep_in_train_rejects_more_rows_than_training_size(self):
        with pytest.raises(ValueError, match="train_axis_size"):
            Split(
                axis_size=5,
                validate_axis_size=3,
                keep_in_train=[0, 1, 2],
            )

    def test_from_model_defaults_to_all_observed_keys(self):
        x = lsl.Var.new_obs(
            jnp.arange(8.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="x"
        )
        y = lsl.Var.new_obs(
            jnp.arange(8.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="y"
        )

        split = Split.from_model(lsl.Model([x, y]), validate_axis_share=0.25)

        assert split.position_keys is not None
        assert set(split.position_keys) == {"x", "y"}
        assert split.axis_size == 8
        assert split.train_axis_size == 6
        assert split.validate_axis_size == 2

    def test_from_model_infers_selected_nonleading_axis_and_forwards_configuration(
        self,
    ):
        model, _ = _matrix_obs_model(shape=(4, 8))

        split = Split.from_model(
            model,
            position_keys=["y"],
            validate_axis_share=0.25,
            test_axis_share=0.25,
            split_axes={"y": 1},
            shuffle=True,
            seed=7,
            sample_sizes={"train": 20, "validate": 5, "test": 5},
        )

        assert split.axis_size == 8
        assert split.train_axis_size == 4
        assert split.validate_axis_size == 2
        assert split.test_axis_size == 2
        assert split.sample_sizes == {"train": 20.0, "validate": 5.0, "test": 5.0}
        assert jnp.array_equal(
            split.indices,
            Split.from_model(
                model,
                position_keys=["y"],
                validate_axis_share=0.25,
                test_axis_share=0.25,
                split_axes={"y": 1},
                shuffle=True,
                seed=7,
            ).indices,
        )

    def test_from_model_preserves_passthrough_keys(self):
        y = lsl.Var.new_obs(
            jnp.arange(8.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="y"
        )
        shared = lsl.Var.new_obs(
            jnp.arange(3.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="shared"
        )
        model = lsl.Model([y, shared])

        splitter = Split.from_model(model, split_axes={"shared": None})
        assert splitter.position_keys is not None
        split = splitter.split_position(model.extract_position(splitter.position_keys))

        assert set(splitter.position_keys) == {"y", "shared"}
        assert splitter.split_position_keys == ["y"]
        assert splitter.passthrough_position_keys == ["shared"]
        assert jnp.array_equal(split.train["shared"], shared.value)

    def test_from_model_trusts_explicit_axis_size_for_another_position(self):
        y = lsl.Var.new_obs(
            jnp.arange(8.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="y"
        )
        splitter = Split.from_model(
            lsl.Model([y]), axis_size=6, validate_axis_share=0.5
        )

        split = splitter.split_position(Position({"y": jnp.arange(6)}))

        assert splitter.axis_size == 6
        assert split.train_axis_size == 3
        assert split.validate_axis_size == 3

    def test_from_model_rejects_invalid_model_groups(self):
        x = lsl.Var.new_obs(
            jnp.arange(8.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="x"
        )
        y = lsl.Var.new_obs(
            jnp.arange(5.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="y"
        )
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="at least one position key"):
            Split.from_model(model, position_keys=[])
        with pytest.raises(ValueError, match="at least one position key to be split"):
            Split.from_model(model, position_keys=["x"], split_axes={"x": None})
        with pytest.raises(ValueError, match="SplitManager.from_model"):
            Split.from_model(model, axis_size=8)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"validate_axis_share": -0.1},
            {"validate_axis_share": 0.6, "test_axis_share": 0.5},
            {"axis_size": 0},
        ],
    )
    def test_from_model_uses_existing_share_and_size_validation(self, kwargs):
        model, _ = _matrix_obs_model()

        with pytest.raises(ValueError):
            Split.from_model(model, **kwargs)

    def test_no_split(self):
        m = lsl.Var.new_param(0.0, name="m")
        x = lsl.Var.new_obs(
            jnp.arange(30),
            distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        split = Split.from_axis_shares(
            position_keys=["x"],
            axis_size=x.value.size,
            validate_axis_share=0.0,
            test_axis_share=0.0,
        )

        assert split.indices_train.size == x.value.size
        assert split.indices_test.size == 0
        assert split.indices_validate.size == 0

        pos = model.extract_position(["x"])
        split_pos = split.split_position(pos)
        assert split_pos.train["x"].size == x.value.size
        assert split_pos.validate["x"].size == 0
        assert split_pos.test["x"].size == 0

    def test_split(self):
        m = lsl.Var.new_param(0.0, name="m")
        x = lsl.Var.new_obs(
            jnp.arange(100),
            distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        split = Split.from_axis_shares(
            position_keys=["x"],
            axis_size=x.value.size,
            validate_axis_share=0.2,
            test_axis_share=0.2,
        )

        assert split.indices_train.size == 60
        assert split.indices_test.size == 20
        assert split.indices_validate.size == 20

        pos = model.extract_position(["x"])
        split_pos = split.split_position(pos)
        assert split_pos.train["x"].size == 60
        assert split_pos.validate["x"].size == 20
        assert split_pos.test["x"].size == 20

    def test_split_incoherent(self):
        with pytest.raises(ValueError):
            Split.from_axis_shares(
                position_keys=["x"],
                axis_size=100,
                validate_axis_share=0.9,
                test_axis_share=0.2,
            )

        with pytest.raises(ValueError):
            Split.from_axis_shares(
                position_keys=["x"],
                axis_size=100,
                validate_axis_share=1.1,
                test_axis_share=0.0,
            )

        with pytest.raises(ValueError):
            Split.from_axis_shares(
                position_keys=["x"],
                axis_size=100,
                validate_axis_share=0.0,
                test_axis_share=1.1,
            )

        with pytest.raises(ValueError):
            Split.from_axis_shares(
                position_keys=["x"],
                axis_size=100,
                validate_axis_share=-0.3,
                test_axis_share=0.5,
            )

        with pytest.raises(ValueError):
            Split.from_axis_shares(
                position_keys=["x"],
                axis_size=2,
                validate_axis_share=0.6,
                test_axis_share=0.6,
            )

        with pytest.raises(ValueError, match="train_axis_size"):
            Split.from_axis_shares(
                position_keys=["x"],
                axis_size=10,
                validate_axis_share=1.0,
                test_axis_share=0.0,
            )

        with pytest.raises(ValueError):
            Split.from_axis_shares(
                position_keys=["x"],
                axis_size=0,
                validate_axis_share=0.0,
                test_axis_share=0.0,
            )

        with pytest.raises(ValueError):
            Split(
                position_keys=["x"],
                axis_size=10,
                train_axis_size=5,
                validate_axis_size=2,
                test_axis_size=1,
            )

        with pytest.raises(ValueError):
            Split(
                position_keys=["x"],
                axis_size=10,
                train_axis_size=-1,
                validate_axis_size=10,
                test_axis_size=1,
            )

        with pytest.raises(ValueError, match="train_axis_size"):
            Split(
                position_keys=["x"],
                axis_size=10,
                train_axis_size=0,
                validate_axis_size=10,
                test_axis_size=0,
            )

        with pytest.raises(ValueError, match="Duplicate position_keys"):
            Split(position_keys=["x", "x"], axis_size=10)


class TestPositionSplit:
    @pytest.mark.parametrize("part", ["train", "validate", "test"])
    def test_empty_parts_can_contain_only_passthrough(self, part):
        positions = {name: Position({}) for name in ("train", "validate", "test")}
        positions[part] = Position({"x": jnp.arange(3)})
        sizes = {f"{name}_axis_size": 3 if name == part else 0 for name in positions}
        split = PositionSplit(
            positions["train"],
            positions["validate"],
            positions["test"],
            sizes["train_axis_size"],
            sizes["validate_axis_size"],
            sizes["test_axis_size"],
            passthrough=Position({"tab": jnp.arange(2)}),
        )
        assert split.split_position_keys == ["x"]
        for position in (split.train, split.validate, split.test):
            assert position["tab"].tolist() == [0, 1]
        sizes[f"{next(name for name in positions if name != part)}_axis_size"] = 1
        with pytest.raises(ValueError, match="must contain position entries"):
            PositionSplit(
                positions["train"],
                positions["validate"],
                positions["test"],
                sizes["train_axis_size"],
                sizes["validate_axis_size"],
                sizes["test_axis_size"],
                passthrough=Position({"tab": jnp.arange(2)}),
            )

    def test_from_model_keeps_position_keys_authoritative(self):
        model, _, _ = _two_branch_model()

        split = PositionSplit.from_model(
            model,
            position_keys=["y1"],
            split_axes={"y2": None},
        )

        assert split.position_keys == ["y1"]

    def test_integrity_checks(self):
        pos = Position({"x": jnp.arange(1)})

        with pytest.raises(ValueError, match="non-negative"):
            PositionSplit(pos, Position({}), Position({}), -1, 0, 0)

        with pytest.raises(ValueError, match="train"):
            PositionSplit(Position({}), Position({}), Position({}), 1, 0, 0)

        with pytest.raises(ValueError, match="declared split size"):
            PositionSplit(
                Position({"x": jnp.arange(1)}), Position({}), Position({}), 2, 0, 0
            )

        with pytest.raises(ValueError, match="validate"):
            PositionSplit(pos, Position({"y": jnp.arange(1)}), Position({}), 1, 1, 0)

        with pytest.raises(ValueError, match="test"):
            PositionSplit(pos, Position({}), Position({"y": jnp.arange(1)}), 1, 0, 1)

    def test_manual_sample_sizes_scale_position_split(self):
        split = PositionSplit(
            Position({"y": jnp.arange(6)}),
            Position({"y": jnp.arange(2)}),
            Position({}),
            train_axis_size=6,
            validate_axis_size=2,
            test_axis_size=0,
            sample_sizes={"train": 24, "validate": 4},
        )

        assert split.validate_sample_scale == 6.0

    def test_manual_sample_sizes_reject_zero_train_size(self):
        with pytest.raises(ValueError, match="train"):
            PositionSplit(
                Position({"y": jnp.arange(6)}),
                Position({"y": jnp.arange(2)}),
                Position({}),
                train_axis_size=6,
                validate_axis_size=2,
                test_axis_size=0,
                sample_sizes={"train": 0, "validate": 4},
            )

    def test_manual_sample_sizes_pass_through_split(self):
        split = Split(
            ["y"],
            axis_size=8,
            validate_axis_size=2,
            sample_sizes={"train": 18, "validate": 3},
        ).split_position(Position({"y": jnp.arange(8)}))

        assert split.validate_sample_scale == 6.0

        with pytest.raises(ValueError, match="train"):
            Split(
                ["y"],
                axis_size=8,
                validate_axis_size=2,
                sample_sizes={"train": 0, "validate": 3},
            ).split_position(Position({"y": jnp.arange(8)}))

    def test_from_model_infers_sample_sizes_from_pointwise_log_probs(self):
        model, _ = _matrix_obs_model(shape=(4, 8))

        split = PositionSplit.from_model(
            model,
            position_keys=["y"],
            validate_axis_share=0.25,
            split_axes={"y": 1},
        )

        assert split.train_axis_size == 6
        assert split.validate_axis_size == 2
        assert split.sample_sizes == {"train": 24.0, "validate": 8.0}
        assert split.validate_sample_scale == 3.0

    def test_from_model_infers_sample_sizes_from_multivariate_log_probs(self):
        y = lsl.Var.new_obs(
            jnp.arange(16.0).reshape(8, 2),
            lsl.Dist(
                tfd.MultivariateNormalDiag,
                loc=jnp.zeros(2),
                scale_diag=jnp.ones(2),
            ),
            name="y",
        )
        model = lsl.Model([y])

        split = PositionSplit.from_model(
            model,
            position_keys=["y"],
            validate_axis_share=0.25,
        )

        assert split.train_axis_size == 6
        assert split.validate_axis_size == 2
        assert split.sample_sizes == {"train": 6.0, "validate": 2.0}
        assert split.validate_sample_scale == 3.0

    def test_add_inferred_sample_sizes_from_model_mutates_split(self):
        model, _ = _matrix_obs_model(shape=(4, 8))
        splitter = Split(["y"], axis_size=8, validate_axis_size=2, split_axes={"y": 1})
        split = splitter.split_position(model.extract_position(["y"]))

        result = split.add_inferred_sample_sizes_from_model(model)

        assert result is split
        assert split.sample_sizes == {"train": 24.0, "validate": 8.0}

    def test_from_model_can_disable_likelihood_size_inference(self):
        model, _ = _matrix_obs_model(shape=(4, 8))

        split = PositionSplit.from_model(
            model,
            position_keys=["y"],
            validate_axis_share=0.25,
            split_axes={"y": 1},
            infer_sample_sizes=False,
        )

        assert split.sample_sizes is None
        assert split.validate_sample_scale == 3.0

    def test_from_model_rejects_empty_position_keys(self):
        model, _ = _matrix_obs_model(shape=(4, 8))

        with pytest.raises(ValueError, match="at least one position key"):
            PositionSplit.from_model(model, position_keys=[], axis_size=8)

    def test_from_model_rejects_per_obs_false_for_likelihood_size_inference(self):
        dist = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
        dist.per_obs = False
        y = lsl.Var.new_obs(jnp.arange(8.0), dist, name="y")
        model = lsl.Model([y])

        with pytest.raises(ValueError, match="per_obs=False"):
            PositionSplit.from_model(model, position_keys=["y"])

    def test_from_model_rejects_custom_log_lik_node_for_likelihood_size_inference(
        self,
    ):
        y = lsl.Var.new_obs(
            jnp.arange(8.0),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y",
        )
        assert y.dist_node is not None
        gb = lsl.GraphBuilder()
        gb.add(y)
        gb.log_lik_node = lsl.Calc(
            lambda log_prob: log_prob.sum(),
            y.dist_node,
            _name="custom_log_lik",
        )
        model = gb.build_model()

        with pytest.raises(ValueError, match="custom log_lik_node"):
            PositionSplit.from_model(model, position_keys=["y"])

    def test_from_model_rejects_incompatible_pointwise_sizes(self):
        y1 = lsl.Var.new_obs(
            jnp.arange(32.0).reshape(4, 8),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y1",
        )
        y2 = lsl.Var.new_obs(
            jnp.arange(8.0),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y2",
        )
        model = lsl.Model([y1, y2])

        with pytest.raises(ValueError, match="incompatible pointwise"):
            PositionSplit.from_model(
                model,
                position_keys=["y1", "y2"],
                validate_axis_share=0.25,
                split_axes={"y1": 1, "y2": 0},
            )


class TestSplitManager:
    def test_from_model_keeps_shared_passthrough_keys(self):
        loc = lsl.Var.new_param(0.0, name="loc")
        y1 = lsl.Var.new_obs(
            jnp.arange(10.0),
            lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
            name="y1",
        )
        y2 = lsl.Var.new_obs(
            jnp.arange(6.0),
            lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
            name="y2",
        )
        shared = lsl.Var.new_obs(jnp.arange(3.0), name="shared")
        model = lsl.Model([y1, y2, shared])

        split = PositionSplitManager.from_model(
            model,
            position_keys=["y1", "y2", "shared"],
            validate_axis_share=0.2,
            split_axes={"shared": None},
        )

        assert split.position_keys == ["y1", "y2", "shared"]
        assert split.split_position_keys == ["y1", "y2"]
        for part in (split.train, split.validate, split.test):
            assert jnp.array_equal(part["shared"], shared.value)

    def test_rejects_children_with_deferred_position_keys(self):
        with pytest.raises(ValueError, match="position_keys"):
            SplitManager([Split(axis_size=3)])

    def test_combines_different_size_branches(self):
        manager = SplitManager(
            [
                Split(
                    ["x"],
                    axis_size=10,
                    validate_axis_size=2,
                    test_axis_size=1,
                    shuffle=False,
                ),
                Split(
                    ["y"],
                    axis_size=6,
                    validate_axis_size=1,
                    test_axis_size=1,
                    shuffle=False,
                ),
            ]
        )
        position = Position({"x": jnp.arange(10), "y": jnp.arange(6)})

        split = manager.split_position(position)

        assert split.position_keys == ["x", "y"]
        assert split.train_axis_sizes == (7, 4)
        assert split.validate_axis_sizes == (2, 1)
        assert split.test_axis_sizes == (1, 1)
        assert split.train["x"].tolist() == list(range(7))
        assert split.validate["y"].tolist() == [4]
        assert split.test["x"].tolist() == [9]

    def test_duplicate_position_keys_raise(self):
        with pytest.raises(ValueError, match="Position keys"):
            SplitManager([Split(["x"], axis_size=3), Split(["x"], axis_size=3)])

    def test_mixed_validation_and_test_availability_raise(self):
        with pytest.raises(ValueError, match="validation data"):
            SplitManager(
                [
                    Split(["x"], axis_size=5, validate_axis_size=1),
                    Split(["y"], axis_size=5, validate_axis_size=0),
                ]
            )

        with pytest.raises(ValueError, match="test data"):
            SplitManager(
                [
                    Split(["x"], axis_size=5, test_axis_size=1),
                    Split(["y"], axis_size=5, test_axis_size=0),
                ]
            )

    def test_axis_handling_is_independent_per_child(self):
        manager = SplitManager(
            [
                Split(["x"], axis_size=4, validate_axis_size=1, split_axes={"x": 1}),
                Split(["y"], axis_size=6, validate_axis_size=2),
            ]
        )
        position = Position(
            {
                "x": jnp.arange(8).reshape(2, 4),
                "y": jnp.arange(12).reshape(6, 2),
            }
        )

        split = manager.split_position(position)

        assert split.train["x"].shape == (2, 3)
        assert split.validate["x"].shape == (2, 1)
        assert split.train["y"].shape == (4, 2)
        assert split.validate["y"].shape == (2, 2)

    def test_shuffling_is_deterministic_with_fixed_seed(self):
        model, _, _ = _two_branch_model()

        manager1 = SplitManager.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            shuffle=True,
            seed=7,
        )
        manager2 = SplitManager.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            shuffle=True,
            seed=7,
        )

        assert len(manager1.splits) == 2
        for split1, split2 in zip(manager1.splits, manager2.splits, strict=True):
            assert jnp.allclose(split1.indices, split2.indices)
            assert jnp.all(split1.indices < split1.axis_size)

    def test_from_model_with_seed_none_fans_out_child_seeds(self, monkeypatch):
        model, _, _ = _two_branch_model()
        monkeypatch.setattr(split_module.time, "time", lambda: 1234.0)

        manager_none = SplitManager.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            shuffle=True,
            seed=None,
        )
        manager_int = SplitManager.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            shuffle=True,
            seed=1234,
        )

        assert len(manager_none.splits) == 2
        assert len(manager_int.splits) == 2
        for split1, split2 in zip(manager_none.splits, manager_int.splits, strict=True):
            assert jnp.allclose(split1.indices, split2.indices)

    def test_from_model_rejects_mixed_availability_from_axis_shares_rounding(self):
        y1 = lsl.Var.new_obs(
            jnp.arange(10.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="y1"
        )
        y2 = lsl.Var.new_obs(
            jnp.arange(4.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="y2"
        )
        model = lsl.Model([y1, y2])

        with pytest.raises(ValueError, match="zero validation observations"):
            SplitManager.from_model(
                model,
                position_keys=["y1", "y2"],
                validate_axis_share=0.2,
            )

    def test_scalar_aliases_work_for_equal_sizes_and_raise_for_unequal_sizes(self):
        equal_position = Position({"x": jnp.arange(10), "y": jnp.arange(10)})
        equal = SplitManager(
            [
                Split(["x"], axis_size=10, validate_axis_size=2),
                Split(["y"], axis_size=10, validate_axis_size=2),
            ]
        ).split_position(equal_position)

        assert equal.train_axis_size == 8
        assert equal.validate_axis_size == 2
        assert equal.validate_sample_scale == 4.0

        unequal = SplitManager(
            [
                Split(["x"], axis_size=10, validate_axis_size=2),
                Split(["y"], axis_size=6, validate_axis_size=1),
            ]
        ).split_position(Position({"x": jnp.arange(10), "y": jnp.arange(6)}))

        with pytest.raises(ValueError, match="train_axis_sizes"):
            _ = unequal.train_axis_size

    def test_from_model_groups_observed_variables_by_sample_size(self):
        model, _, _ = _two_branch_model()

        manager = SplitManager.from_model(
            model, position_keys=["y1", "y2"], validate_axis_share=0.2
        )

        assert manager.position_keys == ["y1", "y2"]
        assert manager.axis_sizes == (10, 6)
        assert manager.validate_axis_sizes == (2, 1)

    def test_position_split_from_model_multi_size_modes(self):
        model, _, _ = _two_branch_model()

        with pytest.raises(ValueError, match="multi_size"):
            PositionSplit.from_model(model)

        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            multi_size="manager",
        )

        assert isinstance(split, PositionSplitManager)
        assert split.validate_axis_sizes == (2, 1)
        assert split.sample_sizes == (
            {"train": 8.0, "validate": 2.0},
            {"train": 5.0, "validate": 1.0},
        )
        assert split.sample_size("train") == 13.0
        assert split.sample_size("validate") == 3.0

        with pytest.raises(ValueError, match="single axis_size value"):
            PositionSplit.from_model(
                model,
                position_keys=["y1", "y2"],
                axis_size=10,
                validate_axis_share=0.2,
                multi_size="manager",
            )

    def test_position_split_from_model_manager_sample_sizes_are_totals(self):
        model, _, _ = _two_branch_model()

        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            multi_size="manager",
            sample_sizes={"train": 130, "validate": 30},
        )

        assert isinstance(split, PositionSplitManager)
        assert split.sample_sizes == (
            {"train": 80.0, "validate": 20.0},
            {"train": 50.0, "validate": 10.0},
        )
        assert split.sample_size("train") == pytest.approx(130.0)
        assert split.sample_size("validate") == pytest.approx(30.0)

    def test_position_split_from_model_manager_rejects_sample_size_for_empty_part(
        self,
    ):
        model, _, _ = _two_branch_model()

        with pytest.raises(ValueError, match="validate_axis_size == 0"):
            PositionSplit.from_model(
                model,
                position_keys=["y1", "y2"],
                multi_size="manager",
                sample_sizes={"validate": 1},
            )

    def test_position_split_from_model_manager_mode_returns_scalar_for_one_size(self):
        x = lsl.Var.new_obs(
            jnp.arange(8.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="x"
        )
        y = lsl.Var.new_obs(
            jnp.arange(8.0), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="y"
        )
        model = lsl.Model([x, y])

        split = PositionSplit.from_model(
            model,
            position_keys=["x", "y"],
            validate_axis_share=0.25,
            multi_size="manager",
        )

        assert isinstance(split, PositionSplit)
        assert split.train_axis_size == 6
        assert split.validate_axis_size == 2

    def test_position_split_manager_exposes_manual_sample_sizes(self):
        position = Position({"x": jnp.arange(8), "y": jnp.arange(6)})
        split = SplitManager(
            [
                Split(
                    ["x"],
                    axis_size=8,
                    validate_axis_size=2,
                    sample_sizes={"train": 30, "validate": 10},
                ),
                Split(
                    ["y"],
                    axis_size=6,
                    validate_axis_size=1,
                    sample_sizes={"train": 15, "validate": 3},
                ),
            ]
        ).split_position(position)

        assert split.sample_sizes == (
            {"train": 30.0, "validate": 10.0},
            {"train": 15.0, "validate": 3.0},
        )
        assert split.train_sample_sizes == (30.0, 15.0)
        assert split.validate_sample_sizes == (10.0, 3.0)
        assert split.sample_size("train") == 45.0
        assert split.sample_size("validate") == 13.0

    def test_position_split_manager_scaled_log_lik_matches_manual_calculation(self):
        model, y1, y2 = _two_branch_model()
        position = model.extract_position(["y1", "y2"])
        split = SplitManager(
            [
                Split(["y1"], axis_size=10, validate_axis_size=2),
                Split(["y2"], axis_size=6, validate_axis_size=1),
            ]
        ).split_position(position)
        state = model.update_state(split.validate, model.state)

        assert y1.dist_node is not None
        assert y2.dist_node is not None
        manual = (
            split.splits[0].validate_sample_scale * state[y1.dist_node.name].value.sum()
            + split.splits[1].validate_sample_scale
            * state[y2.dist_node.name].value.sum()
        )

        assert jnp.allclose(split.scaled_log_lik(model, state), manual)

    def test_scalar_position_split_scaled_log_lik_matches_current_behavior(self):
        y = lsl.Var.new_obs(
            jnp.arange(10.0),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y",
        )
        model = lsl.Model([y])
        split = Split(["y"], axis_size=10, validate_axis_size=2).split_position(
            model.extract_position(["y"])
        )
        state = model.update_state(split.validate, model.state)

        assert jnp.allclose(
            split.scaled_log_lik(model, state),
            split.validate_sample_scale * state["_model_log_lik"].value,
        )

    def test_neg_log_prob_loss_monitor_uses_per_branch_scaling(self):
        model, _, _ = _two_branch_model()
        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            multi_size="manager",
        )
        loss = NegLogProbLoss(model, split)
        carry = OptimCarry.new(
            key=jax.random.key(0),
            epochs=1,
            position=Position({}),
            batches=Batches([], axis_size=1, batch_size=None),
            optimizers=[],
            model_state=model.state,
            save_position_history=False,
        )

        value = loss.loss_monitor(Position({}), carry)
        state = model.update_state(split.validate, model.state)
        manual = -split.scaled_log_lik(model, state)

        assert jnp.allclose(value, manual)

    def test_lieseloptim_builds_full_data_batch_manager_for_position_split_manager(
        self,
    ):
        model, _, _ = _two_branch_model()
        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            multi_size="manager",
        )
        assert isinstance(split, PositionSplitManager)

        quick = LieselOptim(
            model,
            optimizers=optax.adam(0.02),
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            split=split,
        )

        assert isinstance(quick.batches, BatchManager)
        assert quick.batches.axis_size == split.train_axis_sizes
        batch = quick.batches.get_batched_position(split.train, batch_index=0)
        assert batch["y1"].shape == (8,)
        assert batch["y2"].shape == (5,)
        assert split.train is split.train

    def test_lieseloptim_rejects_single_batches_for_position_split_manager(self):
        model, _, _ = _two_branch_model()
        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            multi_size="manager",
        )
        batches = Batches(["y1"], axis_size=8, batch_size=None)

        with pytest.raises(ValueError, match="BatchManager"):
            LieselOptim(
                model,
                optimizers=optax.adam(0.02),
                loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
                split=split,
                batches=batches,
            ).build_engine()


@pytest.mark.parametrize(
    "factory",
    [
        Split.from_model,
        SplitManager.from_model,
        PositionSplit.from_model,
        PositionSplitManager.from_model,
        Batches.from_model,
        BatchManager.from_model,
    ],
)
@pytest.mark.parametrize("value_node_keys", [False, True])
def test_model_factories_require_explicit_groups_without_likelihood(
    factory, value_node_keys
):
    lookup = lsl.Var.new_obs(jnp.arange(5.0), name="lookup")
    model = lsl.Model([lookup])
    keys = [lookup.value_node.name if value_node_keys else lookup.name]
    kwargs = (
        {"batch_size": None}
        if factory in (Batches.from_model, BatchManager.from_model)
        else {}
    )
    with pytest.raises(ValueError, match="explicit nested position_keys"):
        factory(model, position_keys=keys, **kwargs)
    configured = factory(model, position_keys=[keys], **kwargs)
    assert list(configured.position_keys) == keys


@pytest.mark.parametrize("has_likelihood", [False, True])
def test_scalar_observations_require_explicit_passthrough(has_likelihood):
    fixed = lsl.Var.new_obs(
        jnp.array(0.5),
        lsl.Dist(tfd.Normal, loc=0.0, scale=1.0) if has_likelihood else None,
        name="fixed",
    )
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    y = lsl.Var.new_obs(
        jnp.arange(4.0), lsl.Dist(tfd.Normal, loc=loc, scale=fixed), name="y"
    )
    model = lsl.Model([y])
    with pytest.raises(ValueError, match="fixed.*shape.*axis"):
        LieselOptim(model, optimizers=optax.adam(0.02), loss_monitor="train_full_data")
    split = PositionSplit.from_model(model, split_axes={"fixed": None})
    quick = LieselOptim(
        model, optimizers=optax.adam(0.02), split=split, loss_monitor="train_full_data"
    )
    assert quick.batches.position_keys == ["y"]
    assert split.train["fixed"] == 0.5
    assert jnp.allclose(
        split.scaled_log_lik(model, model.state, part="train"), model.log_lik
    )


def test_model_factory_invalid_axis_is_informative():
    model, _ = _matrix_obs_model()
    with pytest.raises(ValueError, match="shape.*axis 2"):
        PositionSplit.from_model(model, split_axes={"y": 2})


@pytest.mark.parametrize("holdout", [0.0, 0.2])
@pytest.mark.parametrize("batch_size", [None, 4])
def test_lookup_model_uses_manual_passthrough(holdout, batch_size):
    beta = lsl.Var.new_param(jnp.array(1.0), name="beta")
    z = lsl.Var.new_obs(jnp.arange(1.0, 6.0), name="z")
    g = lsl.Var.new_obs(jnp.tile(jnp.arange(5), 4), name="g")
    mu = lsl.Var.new_calc(lambda beta, z, g: beta * z[g], beta, z, g, name="mu")
    y = lsl.Var.new_obs(
        2 * z.value[g.value], lsl.Dist(tfd.Normal, loc=mu, scale=1.0), name="y"
    )
    model = lsl.Model([y])
    with pytest.raises(ValueError, match="no observed likelihood"):
        LieselOptim(
            model,
            optimizers=optax.adam(0.02),
            batch_size=batch_size,
            loss_monitor="train_full_data",
        )
    split = PositionSplit.from_model(
        model, split_axes={"z": None}, validate_axis_share=holdout
    )
    engine = LieselOptim(
        model,
        optimizers=optax.adam(0.02),
        split=split,
        batch_size=batch_size,
        loss_monitor="train_full_data",
    ).build_engine()
    loss = engine.loss
    assert isinstance(loss, NegLogProbLoss)
    carry = engine._init_carry(2)
    carry.batch = engine._observed_batch(carry.batches)
    # Full-data runs may use the model-state template instead of an explicit batch.
    rows = split.train | carry.batch

    def expected(beta):
        return (
            -tfd.Normal(loc=beta * z.value[rows["g"]], scale=1.0)
            .log_prob(rows["y"])
            .sum()
            * carry.batches.batch_sample_scale
            / loss.scalar
        )

    def actual(beta):
        return engine.loss.loss_train_batched(Position({"beta": beta}), carry)

    assert jnp.allclose(actual(1.0), expected(1.0))
    assert jnp.allclose(jax.grad(actual)(1.0), jax.grad(expected)(1.0))
    for part in (split.train, split.validate, split.test):
        assert jnp.array_equal(part["z"], z.value)
