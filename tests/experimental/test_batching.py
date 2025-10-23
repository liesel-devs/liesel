import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.random import key, uniform

import liesel.model as lsl
from liesel.experimental.batching import (
    BatchedLieselInterface,
    BatchedLogProb,
    BatchIndices,
)


class TestBatchIndices:
    def test_runs(self):
        Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
        idx = Bi.permute_indices(key(0))

        assert idx.shape == (7, 4)
        assert jnp.unique(idx).size == idx.size

    def test_no_batching(self):
        Bi = BatchIndices(["x"], n=30, batch_size=None, shuffle=False)
        idx = Bi.permute_indices(key(0))
        assert jnp.allclose(idx, jnp.arange(30))

        Bi = BatchIndices(["x"], n=30, batch_size=None, shuffle=True)
        idx = Bi.permute_indices(key(0))
        assert idx.shape[0] == 1
        assert idx.shape[1] == 30
        assert jnp.unique(idx).size == idx.size

    def test_batched_position(self):
        Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
        Bi.indices = Bi.permute_indices(key(0))
        pos = {"x": jnp.arange(30)}
        batched_pos = Bi.get_batched_position(pos)
        assert batched_pos["x"].shape == (4,)

    def test_batching_axis(self):
        Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True, default_axis=1)
        Bi.indices = Bi.permute_indices(key(0))

        x = uniform(key(1), shape=(3, 30))
        pos = {"x": x}

        batched_pos = Bi.get_batched_position(pos)
        assert batched_pos["x"].shape == (3, 4)

    def test_different_batching_axes(self):
        Bi = BatchIndices(
            ["x", "y"], n=30, batch_size=4, shuffle=True, axes={"x": 1, "y": 0}
        )
        Bi.indices = Bi.permute_indices(key(0))

        x = uniform(key(1), shape=(3, 30))
        y = uniform(key(1), shape=(30, 6))
        pos = {"x": x, "y": y}

        batched_pos = Bi.get_batched_position(pos)
        assert batched_pos["x"].shape == (3, 4)
        assert batched_pos["y"].shape == (4, 6)


class TestBatchedLieselInterface:
    def test_batched_state(self):
        x = lsl.Var.new_obs(
            jnp.arange(30),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
        interface = BatchedLieselInterface(model)

        Bi.indices = Bi.permute_indices(key(0))

        batched_state = interface.batched_state(
            position={}, batch_indices=Bi, model_state=model.state
        )

        assert batched_state["x_log_prob"].value.size == Bi.batch_size
        assert batched_state["x_value"].value.size == Bi.batch_size

    def test_batched_log_prob(self):
        x = lsl.Var.new_obs(
            jnp.arange(30),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
        Bi.indices = Bi.permute_indices(key(0))
        interface = BatchedLieselInterface(model)

        Bi.batch_size = Bi.n
        lp_unbatched = interface.batched_log_prob(
            position={}, batch_indices=Bi, model_state=model.state
        )

        Bi.batch_size = 4
        lp_batched = interface.batched_log_prob(
            position={}, batch_indices=Bi, model_state=model.state
        )

        assert not jnp.allclose(lp_unbatched, lp_batched)

    def test_jit_batched_log_prob(self):
        m = lsl.Var.new_param(0.0, name="m")
        x = lsl.Var.new_obs(
            jnp.arange(30),
            distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
        interface = BatchedLieselInterface(model)

        def log_prob(pos, seed):
            Bi.indices = Bi.permute_indices(seed)
            lp_batched = interface.batched_log_prob(
                position=pos, batch_indices=Bi, model_state=model.state
            )
            return lp_batched

        assert not jnp.isnan(jax.jit(log_prob)({"m": 1.0}, key(1)))

    def test_grad_batched_log_prob(self):
        m = lsl.Var.new_param(0.0, name="m")
        x = lsl.Var.new_obs(
            jnp.arange(30),
            distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
        interface = BatchedLieselInterface(model)

        def log_prob(pos, seed):
            Bi.indices = Bi.permute_indices(seed)
            lp_batched = interface.batched_log_prob(
                position=pos, batch_indices=Bi, model_state=model.state
            )
            return lp_batched

        grad = jax.grad(log_prob, argnums=0)({"m": 1.0}, key(1))
        assert not jnp.isnan(grad["m"])

    def test_loop(self):
        """
        Tests a full jitted loop with batched gradient computations

        grads = jnp.zeros(max_iter, Bi.n_full_batches)

        # outer loop (over iterations)
        for i in range(max_iter):
            key, subkey = jax.random.split(key)
            Bi.indices = Bi.permute_indices(subkey)

            # inner loop (over batches)
            for j in range(Bi.n_full_batches):
                Bi.batch_number = j

                grad_ij = log_prob_grad(position, Bi)
                grads = grads.at[i, j].set(grad_ij)
        """
        m = lsl.Var.new_param(0.0, name="m")
        x = lsl.Var.new_obs(
            jnp.arange(30),
            distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
            name="x",
        )
        model = lsl.Model([x])

        Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
        interface = BatchedLieselInterface(model)

        max_iter = 5

        def inner_loop_over_batches(j, val):
            grads, i, pos, Bi = val

            # update the batch number
            Bi.batch_number = j

            # grad_ij = log_prob_grad(pos, Bi)
            grad_ij = BatchedLogProb(interface, model.state, Bi).grad(pos)

            grad_ij_arr = jax.flatten_util.ravel_pytree(grad_ij)[0].squeeze()

            grads = grads.at[i, j].set(grad_ij_arr)
            return grads, i, pos, Bi

        def outer_loop_over_iterations(i, val):
            grads, Bi, key = val

            # permuting the batch indices for each outer iteration
            key, subkey = jax.random.split(key)
            Bi.indices = Bi.permute_indices(subkey)

            # just for testing, using the same position throughout
            # helps discover problems
            pos = {"m": 1.0}

            # run all full batches once
            grads, _, pos, Bi = jax.lax.fori_loop(
                lower=0,
                upper=Bi.n_full_batches,
                body_fun=inner_loop_over_batches,
                init_val=(grads, i, pos, Bi),
            )

            return grads, Bi, key

        grads_init = jnp.zeros((max_iter, Bi.n_full_batches))
        grads, _, _ = jax.lax.fori_loop(
            lower=0,
            upper=max_iter,
            body_fun=outer_loop_over_iterations,
            init_val=(grads_init, Bi, key(0)),
        )

        assert not jnp.any(jnp.isnan(grads))

        for i in range(max_iter - 1):
            assert not jnp.allclose(grads[i], grads[i + 1])

        for i in range(Bi.n_full_batches - 1):
            assert not jnp.allclose(grads[:, i], grads[:, i + 1])
