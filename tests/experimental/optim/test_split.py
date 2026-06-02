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


class TestSplit:
    def test_no_split(self):
        m = lsl.Var.new_param(0.0, name="m")
        x = lsl.Var.new_obs(
            jnp.arange(30),
            distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        split = Split.from_share(
            position_keys=["x"],
            n=x.value.size,
            share_validate=0.0,
            share_test=0.0,
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

        split = Split.from_share(
            position_keys=["x"],
            n=x.value.size,
            share_validate=0.2,
            share_test=0.2,
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
            Split.from_share(
                position_keys=["x"],
                n=100,
                share_validate=0.9,
                share_test=0.2,
            )

        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=100,
                share_validate=1.1,
                share_test=0.0,
            )

        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=100,
                share_validate=0.0,
                share_test=1.1,
            )

        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=100,
                share_validate=-0.3,
                share_test=0.5,
            )

        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=2,
                share_validate=0.6,
                share_test=0.6,
            )

        with pytest.raises(ValueError):
            Split.from_share(
                position_keys=["x"],
                n=0,
                share_validate=0.0,
                share_test=0.0,
            )

        with pytest.raises(ValueError):
            Split(
                position_keys=["x"],
                n=10,
                n_train=5,
                n_validate=2,
                n_test=1,
            )

        with pytest.raises(ValueError):
            Split(
                position_keys=["x"],
                n=10,
                n_train=-1,
                n_validate=10,
                n_test=1,
            )

        with pytest.raises(ValueError, match="Duplicate position_keys"):
            Split(position_keys=["x", "x"], n=10)


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


class TestSplitManager:
    def test_combines_different_size_branches(self):
        manager = SplitManager(
            [
                Split(["x"], n=10, n_validate=2, n_test=1),
                Split(["y"], n=6, n_validate=1, n_test=1),
            ]
        )
        position = {"x": jnp.arange(10), "y": jnp.arange(6)}

        split = manager.split_position(position)

        assert split.position_keys == ["x", "y"]
        assert split.n_trains == (7, 4)
        assert split.n_validates == (2, 1)
        assert split.n_tests == (1, 1)
        assert split.train["x"].tolist() == list(range(7))
        assert split.validate["y"].tolist() == [4]
        assert split.test["x"].tolist() == [9]

    def test_duplicate_position_keys_raise(self):
        with pytest.raises(ValueError, match="Position keys"):
            SplitManager([Split(["x"], n=3), Split(["x"], n=3)])

    def test_mixed_validation_and_test_availability_raise(self):
        with pytest.raises(ValueError, match="validation data"):
            SplitManager(
                [
                    Split(["x"], n=5, n_validate=1),
                    Split(["y"], n=5, n_validate=0),
                ]
            )

        with pytest.raises(ValueError, match="test data"):
            SplitManager(
                [
                    Split(["x"], n=5, n_test=1),
                    Split(["y"], n=5, n_test=0),
                ]
            )

    def test_axis_handling_is_independent_per_child(self):
        manager = SplitManager(
            [
                Split(["x"], n=4, n_validate=1, axes={"x": 1}),
                Split(["y"], n=6, n_validate=2),
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
            share_validate=0.2,
            shuffle=True,
            seed=7,
        )
        manager2 = SplitManager.from_model(
            model,
            position_keys=["y1", "y2"],
            share_validate=0.2,
            shuffle=True,
            seed=7,
        )

        assert len(manager1.splits) == 2
        for split1, split2 in zip(manager1.splits, manager2.splits, strict=True):
            assert jnp.allclose(split1.indices, split2.indices)
            assert jnp.all(split1.indices < split1.n)

    def test_from_model_with_seed_none_fans_out_child_seeds(self, monkeypatch):
        model, _, _ = _two_branch_model()
        monkeypatch.setattr(split_module.time, "time", lambda: 1234.0)

        manager_none = SplitManager.from_model(
            model,
            position_keys=["y1", "y2"],
            share_validate=0.2,
            shuffle=True,
            seed=None,
        )
        manager_int = SplitManager.from_model(
            model,
            position_keys=["y1", "y2"],
            share_validate=0.2,
            shuffle=True,
            seed=1234,
        )

        assert len(manager_none.splits) == 2
        assert len(manager_int.splits) == 2
        for split1, split2 in zip(manager_none.splits, manager_int.splits, strict=True):
            assert jnp.allclose(split1.indices, split2.indices)

    def test_from_model_rejects_mixed_availability_from_share_rounding(self):
        y1 = lsl.Var.new_obs(jnp.arange(10.0), name="y1")
        y2 = lsl.Var.new_obs(jnp.arange(4.0), name="y2")
        model = lsl.Model([y1, y2])

        with pytest.raises(ValueError, match="zero validation observations"):
            SplitManager.from_model(
                model,
                position_keys=["y1", "y2"],
                share_validate=0.2,
            )

    def test_scalar_aliases_work_for_equal_sizes_and_raise_for_unequal_sizes(self):
        equal_position = {"x": jnp.arange(10), "y": jnp.arange(10)}
        equal = SplitManager(
            [
                Split(["x"], n=10, n_validate=2),
                Split(["y"], n=10, n_validate=2),
            ]
        ).split_position(equal_position)

        assert equal.n_train == 8
        assert equal.n_validate == 2
        assert equal.scale_validate == 4.0

        unequal = SplitManager(
            [
                Split(["x"], n=10, n_validate=2),
                Split(["y"], n=6, n_validate=1),
            ]
        ).split_position({"x": jnp.arange(10), "y": jnp.arange(6)})

        with pytest.raises(ValueError, match="n_trains"):
            _ = unequal.n_train

    def test_from_model_groups_observed_variables_by_sample_size(self):
        model, _, _ = _two_branch_model()

        manager = SplitManager.from_model(
            model, position_keys=["y1", "y2"], share_validate=0.2
        )

        assert manager.position_keys == ["y1", "y2"]
        assert manager.ns == (10, 6)
        assert manager.n_validates == (2, 1)

    def test_position_split_from_model_multi_size_modes(self):
        model, _, _ = _two_branch_model()

        with pytest.raises(ValueError, match="multi_size"):
            PositionSplit.from_model(model)

        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            share_validate=0.2,
            multi_size="manager",
        )

        assert isinstance(split, PositionSplitManager)
        assert split.n_validates == (2, 1)

        with pytest.raises(ValueError, match="single n value"):
            PositionSplit.from_model(
                model,
                position_keys=["y1", "y2"],
                n=10,
                share_validate=0.2,
                multi_size="manager",
            )

    def test_position_split_from_model_manager_mode_returns_scalar_for_one_size(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(8.0), name="y")
        model = lsl.Model([x, y])

        split = PositionSplit.from_model(
            model,
            position_keys=["x", "y"],
            share_validate=0.25,
            multi_size="manager",
        )

        assert isinstance(split, PositionSplit)
        assert split.n_train == 6
        assert split.n_validate == 2

    def test_position_split_manager_scaled_log_lik_matches_manual_calculation(self):
        model, y1, y2 = _two_branch_model()
        position = model.extract_position(["y1", "y2"])
        split = SplitManager(
            [
                Split(["y1"], n=10, n_validate=2),
                Split(["y2"], n=6, n_validate=1),
            ]
        ).split_position(position)
        state = model.update_state(split.validate, model.state)

        assert y1.dist_node is not None
        assert y2.dist_node is not None
        manual = (
            split.splits[0].scale_validate * state[y1.dist_node.name].value.sum()
            + split.splits[1].scale_validate * state[y2.dist_node.name].value.sum()
        )

        assert jnp.allclose(split.scaled_log_lik(model, state), manual)

    def test_scalar_position_split_scaled_log_lik_matches_current_behavior(self):
        y = lsl.Var.new_obs(
            jnp.arange(10.0),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y",
        )
        model = lsl.Model([y])
        split = Split(["y"], n=10, n_validate=2).split_position(
            model.extract_position(["y"])
        )
        state = model.update_state(split.validate, model.state)

        assert jnp.allclose(
            split.scaled_log_lik(model, state),
            split.scale_validate * state["_model_log_lik"].value,
        )

    def test_neg_log_prob_loss_validate_uses_per_branch_scaling(self):
        model, _, _ = _two_branch_model()
        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            share_validate=0.2,
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
            share_validate=0.2,
            multi_size="manager",
        )

        quick = LieselOptim(model, split=split)

        assert isinstance(quick.batches, BatchManager)
        assert quick.batches.n == split.n_trains
        batch = quick.batches.get_batched_position(split.train, batch_index=0)
        assert batch["y1"].shape == (8,)
        assert batch["y2"].shape == (5,)
        assert split.train is split.train

    def test_lieseloptim_rejects_single_batches_for_position_split_manager(self):
        model, _, _ = _two_branch_model()
        split = PositionSplit.from_model(
            model,
            position_keys=["y1", "y2"],
            share_validate=0.2,
            multi_size="manager",
        )
        batches = Batches(["y1"], n=8, batch_size=None)

        with pytest.raises(ValueError, match="BatchManager"):
            LieselOptim(model, split=split, batches=batches).build_engine()
