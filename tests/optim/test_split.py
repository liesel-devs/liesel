import math
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim.split as split_module
from liesel.optim import (
    Batches,
    BatchManager,
    LieselOptim,
    NegLogProbLoss,
    PositionSplit,
    PositionSplitManager,
    Split,
    SplitManager,
)
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

        with pytest.raises(ValueError, match="Duplicate position_keys"):
            Split(position_keys=["x", "x"], axis_size=10)


class TestPositionSplit:
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

    def test_manual_sample_sizes_pass_through_split(self):
        split = Split(
            ["y"],
            axis_size=8,
            validate_axis_size=2,
            sample_sizes={"train": 18, "validate": 3},
        ).split_position({"y": jnp.arange(8)})

        assert split.validate_sample_scale == 6.0

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
    def test_combines_different_size_branches(self):
        manager = SplitManager(
            [
                Split(["x"], axis_size=10, validate_axis_size=2, test_axis_size=1),
                Split(["y"], axis_size=6, validate_axis_size=1, test_axis_size=1),
            ]
        )
        position = {"x": jnp.arange(10), "y": jnp.arange(6)}

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
        position = {
            "x": jnp.arange(8).reshape(2, 4),
            "y": jnp.arange(12).reshape(6, 2),
        }

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
        y1 = lsl.Var.new_obs(jnp.arange(10.0), name="y1")
        y2 = lsl.Var.new_obs(jnp.arange(4.0), name="y2")
        model = lsl.Model([y1, y2])

        with pytest.raises(ValueError, match="zero validation observations"):
            SplitManager.from_model(
                model,
                position_keys=["y1", "y2"],
                validate_axis_share=0.2,
            )

    def test_scalar_aliases_work_for_equal_sizes_and_raise_for_unequal_sizes(self):
        equal_position = {"x": jnp.arange(10), "y": jnp.arange(10)}
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
        ).split_position({"x": jnp.arange(10), "y": jnp.arange(6)})

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

    def test_position_split_from_model_manager_mode_returns_scalar_for_one_size(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(8.0), name="y")
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
        position = {"x": jnp.arange(8), "y": jnp.arange(6)}
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

    def test_neg_log_prob_loss_validate_uses_per_branch_scaling(self):
        model, _, _ = _two_branch_model()
        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            validate_axis_share=0.2,
            multi_size="manager",
        )
        loss = NegLogProbLoss(model, split)
        carry = SimpleNamespace(model_state=model.state, fixed_position=Position({}))

        value = loss.loss_validate(Position({}), carry)
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

        quick = LieselOptim(model, split=split)

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
            LieselOptim(model, split=split, batches=batches).build_engine()
