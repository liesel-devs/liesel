import jax.numpy as jnp
from jax.random import key, uniform

from liesel.experimental.optim import Batches


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
