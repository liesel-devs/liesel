import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.random import key, uniform

import liesel.model as lsl
from liesel.optim import Batches, BatchManager


class TestBatches:
    def test_runs(self):
        Bi = Batches(["x"], n=30, batch_size=4, shuffle=True)
        assert Bi.batch_indices.shape == (7, 4)
        assert jnp.unique(Bi.batch_indices).size == 28
        assert jnp.unique(Bi.indices).size == 30

    def test_no_batching(self):
        Bi = Batches(["x"], n=30, batch_size=None, shuffle=False)
        Bi.indices = Bi.permute_indices(key(0))
        assert jnp.allclose(Bi.indices, jnp.arange(30))
        assert jnp.allclose(Bi.batch_indices, jnp.arange(30))

        Bi = Batches(["x"], n=30, batch_size=None, shuffle=True)
        Bi.indices = Bi.permute_indices(key(0))
        idx = Bi.batch_indices
        assert idx.shape[0] == 1
        assert idx.shape[1] == 30
        assert jnp.unique(idx).size == idx.size

    def test_batched_position(self):
        Bi = Batches(["x"], n=30, batch_size=4, shuffle=True)
        Bi.indices = Bi.permute_indices(key(0))
        pos = {"x": jnp.arange(30)}
        batched_pos = Bi.get_batched_position(pos, batch_index=0)
        assert batched_pos["x"].shape == (4,)

    def test_batching_axis(self):
        Bi = Batches(["x"], n=30, batch_size=4, shuffle=True, default_axis=1)
        Bi.indices = Bi.permute_indices(key(0))

        x = uniform(key(1), shape=(3, 30))
        pos = {"x": x}

        batched_pos = Bi.get_batched_position(pos, batch_index=0)
        assert batched_pos["x"].shape == (3, 4)

    def test_different_batching_axes(self):
        Bi = Batches(
            ["x", "y"], n=30, batch_size=4, shuffle=True, axes={"x": 1, "y": 0}
        )
        Bi.indices = Bi.permute_indices(key(0))

        x = uniform(key(1), shape=(3, 30))
        y = uniform(key(1), shape=(30, 6))
        pos = {"x": x, "y": y}

        batched_pos = Bi.get_batched_position(pos, batch_index=0)
        assert batched_pos["x"].shape == (3, 4)
        assert batched_pos["y"].shape == (4, 6)

    def test_duplicate_position_keys_raise(self):
        with pytest.raises(ValueError, match="Duplicate position_keys"):
            Batches(["x", "x"], n=10, batch_size=2)

    def test_scaled_log_lik_matches_old_all_observed_scaling(self):
        y = lsl.Var.new_obs(
            jnp.arange(6.0),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y",
        )
        model = lsl.Model([y])
        batches = Batches(["y"], n=6, batch_size=2, shuffle=False)
        batched = batches.get_batched_position(model.extract_position(["y"]), 0)
        state = model.update_state(batched, model.state)

        assert jnp.allclose(
            batches.scaled_log_lik(model, state),
            batches.batch_share * state["_model_log_lik"].value,
        )

    def test_from_model_rejects_multi_size_by_default(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="multi_size"):
            Batches.from_model(model, batch_size=2, position_keys=["x", "y"])

    def test_from_model_can_return_batch_manager_for_multi_size_data(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        manager = Batches.from_model(
            model,
            batch_size=2,
            position_keys=["x", "y"],
            multi_size="manager",
        )

        assert isinstance(manager, BatchManager)
        assert manager.mode == "resample"
        assert manager.n == (8, 5)
        assert manager.batch_size == (2, 2)
        assert manager.n_full_batches == 4

        started = manager.start_epoch(key(3))
        assert jnp.all(started.batch_numbers[:, 0] < 4)
        assert jnp.all(started.batch_numbers[:, 1] < 2)

    def test_from_model_multi_size_manager_returns_batches_for_one_size(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(8.0), name="y")
        model = lsl.Model([x, y])

        batches = Batches.from_model(
            model,
            batch_size=2,
            position_keys=["x", "y"],
            multi_size="manager",
        )

        assert isinstance(batches, Batches)
        assert batches.position_keys == ["x", "y"]

    def test_from_model_rejects_scalar_n_for_multi_size_manager(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="single n"):
            Batches.from_model(
                model,
                batch_size=2,
                position_keys=["x", "y"],
                n=8,
                multi_size="manager",
            )

    def test_sample_with_replacement_allows_oversized_batch(self):
        batches = Batches(
            ["x"],
            n=5,
            batch_size=8,
            shuffle=True,
            sample_with_replacement=True,
        ).start_epoch(key(1))

        assert batches.n_full_batches == 1
        assert batches.batch_indices.shape == (1, 8)
        assert jnp.all(batches.batch_indices < 5)


class TestBatchManager:
    def test_strict_combines_equal_count_batches(self):
        manager = BatchManager(
            [
                Batches(["x"], n=6, batch_size=2, shuffle=False),
                Batches(["y"], n=9, batch_size=3, shuffle=False),
            ]
        )
        position = {"x": jnp.arange(6), "y": jnp.arange(9)}

        batched = manager.get_batched_position(position, batch_index=1)

        assert manager.position_keys == ["x", "y"]
        assert manager.n == (6, 9)
        assert manager.batch_size == (2, 3)
        assert manager.n_full_batches == 3
        assert batched["x"].tolist() == [2, 3]
        assert batched["y"].tolist() == [3, 4, 5]

    def test_duplicate_position_keys_raise(self):
        with pytest.raises(ValueError, match="Position keys"):
            BatchManager(
                [
                    Batches(["x"], n=6, batch_size=2),
                    Batches(["x"], n=6, batch_size=2),
                ]
            )

    def test_strict_rejects_unequal_number_of_batches(self):
        with pytest.raises(ValueError, match="same n_full_batches"):
            BatchManager(
                [
                    Batches(["x"], n=6, batch_size=2),
                    Batches(["y"], n=8, batch_size=4),
                ]
            )

    def test_resample_epoch_sizes(self):
        max_manager = BatchManager(
            [
                Batches(["x"], n=6, batch_size=2, shuffle=False),
                Batches(["y"], n=8, batch_size=4, shuffle=False),
            ],
            mode="resample",
            epoch_size="max",
        )
        min_manager = BatchManager(
            [
                Batches(["x"], n=6, batch_size=2, shuffle=False),
                Batches(["y"], n=8, batch_size=4, shuffle=False),
            ],
            mode="resample",
            epoch_size="min",
        )
        manual_manager = BatchManager(
            [
                Batches(["x"], n=6, batch_size=2, shuffle=False),
                Batches(["y"], n=8, batch_size=4, shuffle=False),
            ],
            mode="resample",
            epoch_size=5,
        )

        assert max_manager.n_full_batches == 3
        assert min_manager.n_full_batches == 2
        assert manual_manager.n_full_batches == 5

    def test_resampled_batch_numbers_are_bounded_and_deterministic(self):
        def make_manager():
            return BatchManager(
                [
                    Batches(["x"], n=6, batch_size=2, shuffle=False),
                    Batches(["y"], n=8, batch_size=4, shuffle=False),
                ],
                mode="resample",
                epoch_size=5,
            ).start_epoch(key(17))

        manager1 = make_manager()
        manager2 = make_manager()

        assert manager1.batch_numbers.shape == (5, 2)
        assert jnp.all(manager1.batch_numbers[:, 0] < 3)
        assert jnp.all(manager1.batch_numbers[:, 1] < 2)
        assert jnp.allclose(manager1.batch_numbers, manager2.batch_numbers)

    def test_from_model_groups_observed_variables_by_sample_size(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        manager = BatchManager.from_model(
            model,
            batch_size=2,
            position_keys=["x", "y"],
        )

        assert manager.position_keys == ["x", "y"]
        assert manager.n == (8, 5)
        assert manager.batch_size == (2, 2)
        assert manager.n_full_batches == 4

    def test_from_model_supports_full_data_multi_size_batches(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        manager = BatchManager.from_model(
            model,
            batch_size=None,
            position_keys=["x", "y"],
        )

        assert manager.is_full_data
        assert manager.n == (8, 5)
        assert manager.batch_size == (8, 5)
        assert manager.n_full_batches == 1
        assert all(not batch.shuffle for batch in manager.batches)

    def test_from_model_strict_mode_rejects_unequal_child_batch_counts(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="same n_full_batches"):
            BatchManager.from_model(
                model,
                batch_size=2,
                position_keys=["x", "y"],
                mode="strict",
            )

    def test_from_model_allows_oversized_child_batch_in_resample_mode(self):
        x = lsl.Var.new_obs(jnp.arange(12.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        manager = BatchManager.from_model(
            model,
            batch_size=6,
            position_keys=["x", "y"],
        ).start_epoch(key(2))
        position = model.extract_position(["x", "y"])
        batched = manager.get_batched_position(position, batch_index=0)

        assert manager.n == (12, 5)
        assert manager.batch_size == (6, 6)
        assert manager.n_full_batches == 2
        assert manager.batches[1].sample_with_replacement
        assert batched["x"].shape == (6,)
        assert batched["y"].shape == (6,)
        assert jnp.all(batched["y"] < 5)

    def test_from_model_warns_for_resample_without_shuffle(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.warns(UserWarning, match="shuffle=True"):
            BatchManager.from_model(
                model,
                batch_size=2,
                position_keys=["x", "y"],
                shuffle=False,
            )

    def test_from_model_uses_axes_when_grouping_observed_variables(self):
        x = lsl.Var.new_obs(jnp.arange(16.0).reshape(2, 8), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        manager = BatchManager.from_model(
            model,
            batch_size=2,
            position_keys=["x", "y"],
            axes={"x": 1},
        )

        position = model.extract_position(["x", "y"])
        batched = manager.get_batched_position(position, batch_index=0)

        assert manager.n == (8, 5)
        assert batched["x"].shape == (2, 2)
        assert batched["y"].shape == (2,)

    def test_axis_handling_is_independent_per_child(self):
        manager = BatchManager(
            [
                Batches(["x"], n=4, batch_size=2, default_axis=1, shuffle=False),
                Batches(["y"], n=6, batch_size=3, default_axis=0, shuffle=False),
            ],
            mode="resample",
            epoch_size="min",
        )
        position = {
            "x": jnp.arange(12).reshape(3, 4),
            "y": jnp.arange(12).reshape(6, 2),
        }

        batched = manager.get_batched_position(position, batch_index=0)

        assert batched["x"].shape == (3, 2)
        assert batched["y"].shape == (3, 2)

    def test_batch_share_requires_equal_shares(self):
        equal_manager = BatchManager(
            [
                Batches(["x"], n=6, batch_size=2),
                Batches(["y"], n=9, batch_size=3),
            ]
        )
        unequal_manager = BatchManager(
            [
                Batches(["x"], n=10, batch_size=5),
                Batches(["y"], n=9, batch_size=4),
            ]
        )

        assert equal_manager.batch_share == 3.0
        with pytest.raises(ValueError, match="per-branch scaling"):
            _ = unequal_manager.batch_share

    def test_scaled_log_lik_scales_each_branch(self):
        y1 = lsl.Var.new_obs(
            jnp.arange(6.0),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y1",
        )
        y2 = lsl.Var.new_obs(
            jnp.arange(8.0),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y2",
        )
        model = lsl.Model([y1, y2])
        manager = BatchManager(
            [
                Batches(["y1"], n=6, batch_size=2, shuffle=False),
                Batches(["y2"], n=8, batch_size=4, shuffle=False),
            ],
            mode="resample",
            epoch_size="max",
        )
        batch = manager.get_batched_position(model.extract_position(["y1", "y2"]), 0)
        state = model.update_state(batch, model.state)

        assert y1.dist_node is not None
        assert y2.dist_node is not None
        manual = (
            3.0 * state[y1.dist_node.name].value.sum()
            + 2.0 * state[y2.dist_node.name].value.sum()
        )

        assert jnp.allclose(manager.scaled_log_lik(model, state), manual)

    def test_start_epoch_works_under_jit(self):
        manager = BatchManager(
            [
                Batches(["x"], n=6, batch_size=2, shuffle=True),
                Batches(["y"], n=8, batch_size=4, shuffle=True),
            ],
            mode="resample",
            epoch_size="max",
        )

        started = jax.jit(lambda b: b.start_epoch(key(1)))(manager)

        assert started.batch_numbers.shape == (3, 2)

    def test_get_batched_position_accepts_traced_batch_index(self):
        manager = BatchManager(
            [
                Batches(["x"], n=6, batch_size=2, shuffle=False),
                Batches(["y"], n=9, batch_size=3, shuffle=False),
            ]
        )
        position = {"x": jnp.arange(6), "y": jnp.arange(9)}

        def loop(batch_manager, pos):
            def body_fun(i, total):
                batched = batch_manager.get_batched_position(pos, i)
                return total + batched["x"].sum() + batched["y"].sum()

            return jax.lax.fori_loop(0, batch_manager.n_full_batches, body_fun, 0)

        assert jax.jit(loop)(manager, position) == 51
