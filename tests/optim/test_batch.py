import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.random import key, uniform

import liesel.model as lsl
from liesel.optim import Batches, BatchManager, PositionSplit, PositionSplitManager
from liesel.optim.types import Position


class TestBatches:
    def test_removed_construction_mode_is_rejected(self):
        split = PositionSplit(
            Position({"x": jnp.arange(6)}), Position({}), Position({}), 6, 0, 0
        )
        model = lsl.Model([lsl.Var.new_obs(jnp.arange(6.0), name="x")])

        with pytest.raises(TypeError):
            BatchManager([Batches(["x"], axis_size=6, batch_size=2)], mode="strict")  # ty: ignore[unknown-argument]
        with pytest.raises(TypeError):
            Batches.from_split(split, batch_size=2, mode="strict")  # ty: ignore[unknown-argument]
        with pytest.raises(TypeError):
            Batches.from_model(  # ty: ignore[no-matching-overload]
                model, batch_size=2, position_keys=["x"], mode="strict"
            )
        with pytest.raises(TypeError):
            BatchManager.from_model(
                model,
                batch_size=2,
                position_keys=["x"],
                mode="strict",  # ty: ignore[unknown-argument]
            )

    def test_from_split_builds_training_batches(self):
        split = PositionSplit(
            train=Position({"x": jnp.arange(6)}),
            validate=Position({"x": jnp.arange(6, 8)}),
            test=Position({}),
            train_axis_size=6,
            validate_axis_size=2,
            test_axis_size=0,
            sample_sizes={"train": 24},
        )

        batches = Batches.from_split(split, batch_size=2, shuffle=False)

        assert isinstance(batches, Batches)
        assert batches.position_keys == ["x"]
        assert batches.axis_size == 6
        assert batches.batch_size == 2
        assert batches.sample_size == 24.0

    def test_from_split_builds_batch_manager(self):
        split = PositionSplitManager(
            [
                PositionSplit(
                    Position({"x": jnp.arange(6)}),
                    Position({}),
                    Position({}),
                    6,
                    0,
                    0,
                ),
                PositionSplit(
                    Position({"y": jnp.arange(4)}),
                    Position({}),
                    Position({}),
                    4,
                    0,
                    0,
                ),
            ]
        )

        batches = Batches.from_split(split, batch_size=2, shuffle=True)

        assert isinstance(batches, BatchManager)
        assert batches.position_keys == ["x", "y"]
        assert batches.axis_size == (6, 4)
        assert batches.n_full_batches == 3

    def test_runs(self):
        Bi = Batches(["x"], axis_size=30, batch_size=4, shuffle=True)
        assert Bi.batch_indices.shape == (7, 4)
        assert jnp.unique(Bi.batch_indices).size == 28
        assert jnp.unique(Bi.indices).size == 30

    def test_default_epoch_keeps_incomplete_tail_shape(self):
        batches = Batches(["x"], axis_size=10, batch_size=4, shuffle=True)
        started = batches.start_epoch(key(0))

        assert started.indices.shape == (10,)
        assert started.batch_indices.shape == (2, 4)
        assert jnp.unique(started.indices).size == 10

    def test_default_epoch_resets_after_extended_epoch(self):
        batches = Batches(["x"], axis_size=10, batch_size=4, shuffle=True)
        batches.start_epoch(key(0), n_batches=3)
        started = batches.start_epoch(key(1))

        assert started.indices.shape == (10,)
        assert started.batch_indices.shape == (2, 4)
        assert jnp.array_equal(jnp.sort(started.indices), jnp.arange(10))

    def test_permute_indices_stays_natural_after_extended_epoch(self):
        batches = Batches(["x"], axis_size=10, batch_size=4, shuffle=True)
        batches.start_epoch(key(0), n_batches=3)

        indices = batches.permute_indices(key(1))

        assert indices.shape == (10,)
        assert jnp.array_equal(jnp.sort(indices), jnp.arange(10))

    def test_old_batch_axis_size_keyword_still_works(self):
        batches = Batches(["x"], axis_size=10, batch_axis_size=4, shuffle=False)

        assert batches.batch_size == 4
        assert batches.batch_indices.shape == (2, 4)

    def test_batch_size_and_old_keyword_are_mutually_exclusive(self):
        with pytest.raises(TypeError, match="batch_size or batch_axis_size"):
            Batches(["x"], axis_size=10, batch_size=4, batch_axis_size=4)

    def test_no_batching(self):
        Bi = Batches(["x"], axis_size=30, batch_size=None, shuffle=False)
        Bi.indices = Bi.permute_indices(key(0))
        assert jnp.allclose(Bi.indices, jnp.arange(30))
        assert jnp.allclose(Bi.batch_indices, jnp.arange(30))

        Bi = Batches(["x"], axis_size=30, batch_size=None, shuffle=True)
        Bi.indices = Bi.permute_indices(key(0))
        idx = Bi.batch_indices
        assert idx.shape[0] == 1
        assert idx.shape[1] == 30
        assert jnp.unique(idx).size == idx.size

    def test_empty_position_keys_allow_full_data_adapter(self):
        batches = Batches([], axis_size=30, batch_size=None, shuffle=True)

        assert batches.position_keys == []
        assert batches.batch_size == 30
        assert batches.is_full_data

    def test_empty_position_keys_reject_mini_batches(self):
        with pytest.raises(ValueError, match="position_keys"):
            Batches([], axis_size=30, batch_size=4)

    def test_batched_position(self):
        Bi = Batches(["x"], axis_size=30, batch_size=4, shuffle=True)
        Bi.indices = Bi.permute_indices(key(0))
        pos = Position({"x": jnp.arange(30)})
        batched_pos = Bi.get_batched_position(pos, batch_index=0)
        assert batched_pos["x"].shape == (4,)

    def test_batching_axis(self):
        Bi = Batches(
            ["x"], axis_size=30, batch_size=4, shuffle=True, default_batch_axis=1
        )
        Bi.indices = Bi.permute_indices(key(0))

        x = uniform(key(1), shape=(3, 30))
        pos = Position({"x": x})

        batched_pos = Bi.get_batched_position(pos, batch_index=0)
        assert batched_pos["x"].shape == (3, 4)

    def test_different_batching_axes(self):
        Bi = Batches(
            ["x", "y"],
            axis_size=30,
            batch_size=4,
            shuffle=True,
            batch_axes={"x": 1, "y": 0},
        )
        Bi.indices = Bi.permute_indices(key(0))

        x = uniform(key(1), shape=(3, 30))
        y = uniform(key(1), shape=(30, 6))
        pos = Position({"x": x, "y": y})

        batched_pos = Bi.get_batched_position(pos, batch_index=0)
        assert batched_pos["x"].shape == (3, 4)
        assert batched_pos["y"].shape == (4, 6)

    def test_duplicate_position_keys_raise(self):
        with pytest.raises(ValueError, match="Duplicate position_keys"):
            Batches(["x", "x"], axis_size=10, batch_size=2)

    def test_scaled_log_lik_matches_old_all_observed_scaling(self):
        y = lsl.Var.new_obs(
            jnp.arange(6.0),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y",
        )
        model = lsl.Model([y])
        batches = Batches(["y"], axis_size=6, batch_size=2, shuffle=False)
        batched = batches.get_batched_position(model.extract_position(["y"]), 0)
        state = model.update_state(batched, model.state)

        assert jnp.allclose(
            batches.scaled_log_lik(model, state),
            batches.batch_sample_scale * state["_model_log_lik"].value,
        )

    def test_from_model_rejects_multi_size_by_default(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="multi_size"):
            Batches.from_model(model, batch_size=2, position_keys=["x", "y"])

    def test_from_model_accepts_old_batch_axis_size_keyword(self):
        y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        model = lsl.Model([y])

        batches = Batches.from_model(
            model, batch_axis_size=2, position_keys=["y"], shuffle=False
        )

        assert isinstance(batches, Batches)
        assert batches.batch_size == 2

    def test_from_model_empty_position_keys_allow_full_data_adapter(self):
        y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        model = lsl.Model([y])

        batches = Batches.from_model(model, batch_size=None, position_keys=[])

        assert isinstance(batches, Batches)
        assert batches.position_keys == []
        assert batches.axis_size == 6
        assert batches.batch_size == 6
        assert batches.is_full_data

    def test_from_model_empty_position_keys_reject_mini_batches(self):
        y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        model = lsl.Model([y])

        with pytest.raises(ValueError, match="position_keys"):
            Batches.from_model(model, batch_size=2, position_keys=[])

    def test_from_model_empty_position_keys_multi_size_requires_axis_size(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="axis_size"):
            Batches.from_model(model, batch_size=None, position_keys=[])

        batches = Batches.from_model(
            model, batch_size=None, position_keys=[], axis_size=8
        )

        assert batches.axis_size == 8
        assert batches.is_full_data

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
        assert manager.axis_size == (8, 5)
        assert manager.batch_size == (2, 2)
        assert manager.n_full_batches == 4

        started = manager.start_epoch(key(3))
        assert tuple(index.shape for index in started.batch_indices) == ((4, 2), (4, 2))

    def test_from_model_global_replacement_reaches_every_child(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        manager = Batches.from_model(
            model,
            batch_size=2,
            position_keys=["x", "y"],
            multi_size="manager",
            sample_with_replacement=True,
        ).start_epoch(key(0))

        assert isinstance(manager, BatchManager)
        assert all(batch.sample_with_replacement for batch in manager.batches)
        assert all(
            jnp.all((indices >= 0) & (indices < batch.axis_size))
            for indices, batch in zip(
                manager.batch_indices, manager.batches, strict=True
            )
        )

    def test_manager_from_model_global_replacement_reaches_every_child(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        manager = BatchManager.from_model(
            lsl.Model([x, y]),
            batch_size=2,
            position_keys=["x", "y"],
            sample_with_replacement=True,
        )
        assert all(batch.sample_with_replacement for batch in manager.batches)

    def test_manager_from_model_oversized_child_requires_shuffle(self):
        x = lsl.Var.new_obs(jnp.arange(12.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        with pytest.raises(ValueError, match="shuffle=True"):
            BatchManager.from_model(
                lsl.Model([x, y]),
                batch_size=6,
                position_keys=["x", "y"],
                shuffle=False,
            )

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

    def test_from_model_rejects_scalar_axis_size_for_multi_size_manager(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="Single axis"):
            Batches.from_model(
                model,
                batch_size=2,
                position_keys=["x", "y"],
                axis_size=8,
                multi_size="manager",
            )

    def test_sample_with_replacement_allows_oversized_batch(self):
        batches = Batches(
            ["x"],
            axis_size=5,
            batch_size=8,
            shuffle=True,
            sample_with_replacement=True,
        ).start_epoch(key(1))

        assert batches.n_full_batches == 1
        assert batches.batch_indices.shape == (1, 8)
        assert jnp.all(batches.batch_indices < 5)

    @pytest.mark.parametrize("n_batches", [1, 2, 5])
    def test_start_epoch_assembles_requested_shuffled_passes(self, n_batches):
        batches = Batches(["x"], axis_size=5, batch_size=2, shuffle=True)

        started = batches.start_epoch(key(3), n_batches=n_batches)

        assert started.n_full_batches == 2
        assert started.batch_indices.shape == (n_batches, 2)
        for passed in range((n_batches + 1) // 2):
            rows = started.batch_indices[passed * 2 : min((passed + 1) * 2, n_batches)]
            assert jnp.unique(rows).size == rows.size

    def test_start_epoch_multpass_matches_fixed_key_and_reproducibility(self):
        batches = Batches(["x"], axis_size=5, batch_size=2, shuffle=True)
        started = batches.start_epoch(key(3), n_batches=5)
        keys = jax.random.split(key(3), 3)
        expected = jnp.concatenate([jax.random.permutation(k, 5)[:4] for k in keys])[
            :10
        ].reshape(5, 2)

        assert jnp.array_equal(started.batch_indices, expected)
        assert jnp.array_equal(
            batches.start_epoch(key(3), n_batches=5).batch_indices, expected
        )

    def test_pytree_round_trip_preserves_assembled_indices(self):
        batches = Batches(["x"], axis_size=5, batch_size=2).start_epoch(
            key(3), n_batches=5
        )

        leaves, treedef = jax.tree_util.tree_flatten(batches)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.array_equal(restored.indices, batches.indices)
        assert jnp.array_equal(restored.batch_indices, batches.batch_indices)

    def test_replacement_applies_to_every_requested_batch(self):
        batches = Batches(
            ["x"], axis_size=5, batch_size=2, sample_with_replacement=True
        )

        started = batches.start_epoch(key(3), n_batches=4)

        expected = jax.random.randint(key(3), (8,), 0, 5).reshape(4, 2)
        assert started.batch_indices.shape == (4, 2)
        assert jnp.array_equal(started.batch_indices, expected)
        assert not started.is_full_data

    @pytest.mark.parametrize("batch_size", [2, 5, 8])
    def test_replacement_batch_sizes(self, batch_size):
        batches = Batches(
            ["x"], axis_size=5, batch_size=batch_size, sample_with_replacement=True
        ).start_epoch(key(4), n_batches=3)

        expected = jax.random.randint(key(4), (3 * batch_size,), 0, 5).reshape(
            3, batch_size
        )
        assert jnp.array_equal(batches.batch_indices, expected)
        assert batches.batch_indices.shape == (3, batch_size)
        assert not batches.is_full_data

    @pytest.mark.parametrize("n_batches", [True, 0, -1, 1.5])
    def test_start_epoch_rejects_invalid_requested_count(self, n_batches):
        batches = Batches(["x"], axis_size=5, batch_size=2)

        with pytest.raises((TypeError, ValueError)):
            batches.start_epoch(key(0), n_batches=n_batches)

    def test_replacement_rejects_unshuffled_and_unbatched(self):
        with pytest.raises(ValueError, match="shuffle"):
            Batches(
                ["x"],
                axis_size=5,
                batch_size=2,
                shuffle=False,
                sample_with_replacement=True,
            )
        with pytest.raises(ValueError, match="batch_size"):
            Batches(["x"], axis_size=5, batch_size=None, sample_with_replacement=True)

    def test_manual_sample_sizes_scale_batch_likelihood(self):
        batches = Batches(
            ["y"],
            axis_size=6,
            batch_size=2,
            sample_size=30,
            batch_sample_size=5,
        )

        assert batches.batch_sample_scale == 6.0

    def test_from_model_infers_sample_sizes_for_nonleading_batch_axis(self):
        y = lsl.Var.new_obs(
            jnp.arange(32.0).reshape(4, 8),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="y",
        )
        model = lsl.Model([y])

        batches = Batches.from_model(
            model,
            batch_size=2,
            position_keys=["y"],
            batch_axes={"y": 1},
            shuffle=False,
        )

        assert isinstance(batches, Batches)
        assert batches.axis_size == 8
        assert batches.batch_size == 2
        assert batches.sample_size == 32.0
        assert batches.batch_sample_size == 8.0
        assert batches.batch_sample_scale == 4.0


class TestBatchManager:
    def test_strict_combines_equal_count_batches(self):
        manager = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2, shuffle=False),
                Batches(["y"], axis_size=9, batch_size=3, shuffle=False),
            ]
        )
        position = Position({"x": jnp.arange(6), "y": jnp.arange(9)})

        batched = manager.get_batched_position(position, batch_index=1)

        assert manager.position_keys == ["x", "y"]
        assert manager.axis_size == (6, 9)
        assert manager.batch_size == (2, 3)
        assert manager.n_full_batches == 3
        assert batched["x"].tolist() == [2, 3]
        assert batched["y"].tolist() == [3, 4, 5]

    def test_duplicate_position_keys_raise(self):
        with pytest.raises(ValueError, match="Position keys"):
            BatchManager(
                [
                    Batches(["x"], axis_size=6, batch_size=2),
                    Batches(["x"], axis_size=6, batch_size=2),
                ]
            )

    def test_strict_rejects_unequal_number_of_batches(self):
        with pytest.raises(ValueError, match="same n_full_batches"):
            BatchManager(
                [
                    Batches(["x"], axis_size=6, batch_size=2),
                    Batches(["y"], axis_size=8, batch_size=4),
                ]
            )

    def test_epoch_sizes(self):
        max_manager = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
                Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
            ],
            epoch_size="max",
        )
        min_manager = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
                Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
            ],
            epoch_size="min",
        )
        manual_manager = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
                Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
            ],
            epoch_size=5,
        )

        assert max_manager.n_full_batches == 3
        assert min_manager.n_full_batches == 2
        assert manual_manager.n_full_batches == 5

    @pytest.mark.parametrize("epoch_size", ["bad", True, 0, -1])
    def test_invalid_epoch_size_raises(self, epoch_size):
        with pytest.raises((TypeError, ValueError)):
            BatchManager(
                [Batches(["x"], axis_size=6, batch_size=2)],
                epoch_size=epoch_size,
            )

    def test_construction_prepares_resolved_deterministic_rows(self):
        children = [
            Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
            Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
        ]
        manager = BatchManager(children, epoch_size="max")

        assert [batch.n_full_batches for batch in manager.batches] == [3, 2]
        assert manager.batch_indices[0].tolist() == [[0, 1], [2, 3], [4, 5]]
        assert manager.batch_indices[1].tolist() == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 2, 3],
        ]

    def test_pytree_round_trip_preserves_dynamic_indices(self):
        manager = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
                Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
            ],
            epoch_size="max",
        ).start_epoch(key(7))

        leaves, treedef = jax.tree_util.tree_flatten(manager)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert all(
            jnp.array_equal(first, second)
            for first, second in zip(
                manager.batch_indices, restored.batch_indices, strict=True
            )
        )

    def test_assembled_rows_are_deterministic_for_a_fixed_key(self):
        def make_manager():
            return BatchManager(
                [
                    Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
                    Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
                ],
                epoch_size=5,
            ).start_epoch(key(17))

        manager1 = make_manager()
        manager2 = make_manager()

        assert tuple(index.shape for index in manager1.batch_indices) == (
            (5, 2),
            (5, 4),
        )
        assert all(
            jnp.allclose(first, second)
            for first, second in zip(
                manager1.batch_indices, manager2.batch_indices, strict=True
            )
        )

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
        assert manager.axis_size == (8, 5)
        assert manager.batch_size == (2, 2)
        assert manager.n_full_batches == 4

    def test_from_model_accepts_old_batch_axis_size_keyword(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        manager = BatchManager.from_model(
            model,
            batch_axis_size=2,
            position_keys=["x", "y"],
        )

        assert manager.batch_size == (2, 2)

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
        assert manager.axis_size == (8, 5)
        assert manager.batch_size == (8, 5)
        assert manager.n_full_batches == 1
        assert all(not batch.shuffle for batch in manager.batches)

    def test_from_model_strict_epoch_size_rejects_unequal_child_batch_counts(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="same n_full_batches"):
            BatchManager.from_model(
                model,
                batch_size=2,
                position_keys=["x", "y"],
                epoch_size="strict",
            )

    def test_from_model_allows_oversized_child_batch(self):
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

        assert manager.axis_size == (12, 5)
        assert manager.batch_size == (6, 6)
        assert manager.n_full_batches == 2
        assert manager.batches[1].sample_with_replacement
        assert batched["x"].shape == (6,)
        assert batched["y"].shape == (6,)
        assert jnp.all(batched["y"] < 5)

    def test_from_model_rejects_additional_batches_without_shuffle(self):
        x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        model = lsl.Model([x, y])

        with pytest.raises(ValueError, match="additional batches"):
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
            batch_axes={"x": 1},
        )

        position = model.extract_position(["x", "y"])
        batched = manager.get_batched_position(position, batch_index=0)

        assert manager.axis_size == (8, 5)
        assert batched["x"].shape == (2, 2)
        assert batched["y"].shape == (2,)

    def test_axis_handling_is_independent_per_child(self):
        manager = BatchManager(
            [
                Batches(
                    ["x"],
                    axis_size=4,
                    batch_size=2,
                    default_batch_axis=1,
                    shuffle=False,
                ),
                Batches(
                    ["y"],
                    axis_size=6,
                    batch_size=3,
                    default_batch_axis=0,
                    shuffle=False,
                ),
            ],
            epoch_size="min",
        )
        position = Position(
            {
                "x": jnp.arange(12).reshape(3, 4),
                "y": jnp.arange(12).reshape(6, 2),
            }
        )

        batched = manager.get_batched_position(position, batch_index=0)

        assert batched["x"].shape == (3, 2)
        assert batched["y"].shape == (3, 2)

    def test_batch_sample_scale_requires_equal_shares(self):
        equal_manager = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2),
                Batches(["y"], axis_size=9, batch_size=3),
            ]
        )
        unequal_manager = BatchManager(
            [
                Batches(["x"], axis_size=10, batch_size=5),
                Batches(["y"], axis_size=9, batch_size=4),
            ]
        )

        assert equal_manager.batch_sample_scale == 3.0
        with pytest.raises(ValueError, match="per-branch scaling"):
            _ = unequal_manager.batch_sample_scale

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
                Batches(["y1"], axis_size=6, batch_size=2, shuffle=True),
                Batches(["y2"], axis_size=8, batch_size=4, shuffle=True),
            ],
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

    def test_scaled_log_lik_uses_manual_sample_sizes(self):
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
                Batches(
                    ["y1"],
                    axis_size=6,
                    batch_size=2,
                    shuffle=True,
                    sample_size=60,
                    batch_sample_size=10,
                ),
                Batches(
                    ["y2"],
                    axis_size=8,
                    batch_size=4,
                    shuffle=True,
                    sample_size=24,
                    batch_sample_size=6,
                ),
            ],
            epoch_size="max",
        )
        batch = manager.get_batched_position(model.extract_position(["y1", "y2"]), 0)
        state = model.update_state(batch, model.state)

        assert y1.dist_node is not None
        assert y2.dist_node is not None
        manual = (
            6.0 * state[y1.dist_node.name].value.sum()
            + 4.0 * state[y2.dist_node.name].value.sum()
        )

        assert manager.batch_sample_scales == (6.0, 4.0)
        assert jnp.allclose(manager.scaled_log_lik(model, state), manual)

    def test_start_epoch_works_under_jit(self):
        manager = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
                Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
            ],
            epoch_size="max",
        )

        started = jax.jit(lambda b: b.start_epoch(key(1)))(manager)
        expected = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
                Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
            ],
            epoch_size="max",
        ).start_epoch(key(1))

        assert tuple(index.shape for index in started.batch_indices) == ((3, 2), (3, 4))
        assert all(
            jnp.array_equal(first, second)
            for first, second in zip(
                started.batch_indices, expected.batch_indices, strict=True
            )
        )

    def test_selective_replacement_uses_each_child_key(self):
        manager = BatchManager(
            [
                Batches(
                    ["x"],
                    axis_size=3,
                    batch_size=2,
                    shuffle=True,
                    sample_with_replacement=True,
                ),
                Batches(["y"], axis_size=6, batch_size=2, shuffle=True),
            ],
            epoch_size=3,
        ).start_epoch(key(11))
        child_keys = jax.random.split(key(11), 2)
        expected_x = jax.random.randint(child_keys[0], (6,), 0, 3).reshape(3, 2)
        expected_y = jax.random.permutation(
            jax.random.split(child_keys[1], 1)[0], 6
        ).reshape(3, 2)

        assert jnp.array_equal(manager.batch_indices[0], expected_x)
        assert jnp.array_equal(manager.batch_indices[1], expected_y)

    def test_get_batched_position_accepts_traced_batch_index(self):
        manager = BatchManager(
            [
                Batches(["x"], axis_size=6, batch_size=2, shuffle=False),
                Batches(["y"], axis_size=9, batch_size=3, shuffle=False),
            ]
        )
        position = {"x": jnp.arange(6), "y": jnp.arange(9)}

        def loop(batch_manager, pos):
            def body_fun(i, total):
                batched = batch_manager.get_batched_position(pos, i)
                return total + batched["x"].sum() + batched["y"].sum()

            return jax.lax.fori_loop(0, batch_manager.n_full_batches, body_fun, 0)

        assert jax.jit(loop)(manager, position) == 51
