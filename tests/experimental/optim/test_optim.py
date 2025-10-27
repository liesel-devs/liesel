"""
Currently broken tests from development process.
"""

# import jax
# import jax.numpy as jnp
# import optax
# import pytest
# import tensorflow_probability.substrates.jax.distributions as tfd
# from jax.random import key, uniform

# import liesel.model as lsl
# from liesel.experimental.batching import (
#     BatchIndices,
#     OptimEngine,
#     Optimizer,
#     OptimLieselInterface,
#     TrainValidationTestSplit,
# )
# from liesel.goose.optim import Stopper


# class TestBatchIndices:
#     def test_runs(self):
#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
#         idx = Bi.permute_indices(key(0))

#         assert idx.shape == (7, 4)
#         assert jnp.unique(idx).size == idx.size

#     def test_no_batching(self):
#         Bi = BatchIndices(["x"], n=30, batch_size=None, shuffle=False)
#         idx = Bi.permute_indices(key(0))
#         assert jnp.allclose(idx, jnp.arange(30))

#         Bi = BatchIndices(["x"], n=30, batch_size=None, shuffle=True)
#         idx = Bi.permute_indices(key(0))
#         assert idx.shape[0] == 1
#         assert idx.shape[1] == 30
#         assert jnp.unique(idx).size == idx.size

#     def test_batched_position(self):
#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
#         Bi.indices = Bi.permute_indices(key(0))
#         pos = {"x": jnp.arange(30)}
#         batched_pos = Bi.get_batched_position(pos)
#         assert batched_pos["x"].shape == (4,)

#     def test_batching_axis(self):
#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True, default_axis=1)
#         Bi.indices = Bi.permute_indices(key(0))

#         x = uniform(key(1), shape=(3, 30))
#         pos = {"x": x}

#         batched_pos = Bi.get_batched_position(pos)
#         assert batched_pos["x"].shape == (3, 4)

#     def test_different_batching_axes(self):
#         Bi = BatchIndices(
#             ["x", "y"], n=30, batch_size=4, shuffle=True, axes={"x": 1, "y": 0}
#         )
#         Bi.indices = Bi.permute_indices(key(0))

#         x = uniform(key(1), shape=(3, 30))
#         y = uniform(key(1), shape=(30, 6))
#         pos = {"x": x, "y": y}

#         batched_pos = Bi.get_batched_position(pos)
#         assert batched_pos["x"].shape == (3, 4)
#         assert batched_pos["y"].shape == (4, 6)


# class TestOptimLieselInterface:
#     def test_batched_state(self):
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
#         interface = OptimLieselInterface(model)

#         Bi.indices = Bi.permute_indices(key(0))

#         batched_state = interface.batched_state(
#             position={}, batch_indices=Bi, model_state=model.state
#         )

#         assert batched_state["x_log_prob"].value.size == Bi.batch_size
#         assert batched_state["x_value"].value.size == Bi.batch_size

#     def test_batched_log_prob(self):
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
#         Bi.indices = Bi.permute_indices(key(0))
#         interface = OptimLieselInterface(model)

#         Bi.batch_size = Bi.n
#         lp_unbatched = interface.batched_log_prob(
#             position={}, batch_indices=Bi, model_state=model.state
#         )

#         Bi.batch_size = 4
#         lp_batched = interface.batched_log_prob(
#             position={}, batch_indices=Bi, model_state=model.state
#         )

#         assert not jnp.allclose(lp_unbatched, lp_batched)

#     def test_jit_batched_log_prob(self):
#         m = lsl.Var.new_param(0.0, name="m")
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
#         interface = OptimLieselInterface(model)

#         def log_prob(pos, seed):
#             Bi.indices = Bi.permute_indices(seed)
#             lp_batched = interface.batched_log_prob(
#                 position=pos, batch_indices=Bi, model_state=model.state
#             )
#             return lp_batched

#         assert not jnp.isnan(jax.jit(log_prob)({"m": 1.0}, key(1)))

#     def test_grad_batched_log_prob(self):
#         m = lsl.Var.new_param(0.0, name="m")
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
#         interface = OptimLieselInterface(model)

#         def log_prob(pos, seed):
#             Bi.indices = Bi.permute_indices(seed)
#             lp_batched = interface.batched_log_prob(
#                 position=pos, batch_indices=Bi, model_state=model.state
#             )
#             return lp_batched

#         grad = jax.grad(log_prob, argnums=0)({"m": 1.0}, key(1))
#         assert not jnp.isnan(grad["m"])

#     def test_val_and_grad(self):
#         m = lsl.Var.new_param(0.0, name="m")
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
#         interface = OptimLieselInterface(
#             model, n_train=Bi.n, n_validation=Bi.n, batch_size=Bi.batch_size
#         )
#         pos = {"m": 1.0}
#         model_state = model.state
#         obs_batch = Bi.extract_batched_position(interface, model_state)
#         val_and_grad = jax.value_and_grad(interface.scaled_neg_log_prob, argnums=0)
#         val, grad_val = val_and_grad(pos, model_state, obs_batch)

#         def value_and_grad(pos, model_state, obs_batch):
#             val_and_grad = jax.value_and_grad(interface.scaled_neg_log_prob, argnums=0)
#             val, grad_tree = val_and_grad(pos, model_state, obs_batch)
#             return val, grad_tree

#         value_and_grad(pos, model_state, obs_batch)

#     def test_loop(self):
#         """
#         Tests a full jitted loop with batched gradient computations

#         grads = jnp.zeros(max_iter, Bi.n_full_batches)

#         # outer loop (over iterations)
#         for i in range(max_iter):
#             key, subkey = jax.random.split(key)
#             Bi.indices = Bi.permute_indices(subkey)

#             # inner loop (over batches)
#             for j in range(Bi.n_full_batches):
#                 Bi.batch_number = j

#                 grad_ij = log_prob_grad(position, Bi)
#                 grads = grads.at[i, j].set(grad_ij)
#         """
#         m = lsl.Var.new_param(0.0, name="m")
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         Bi = BatchIndices(["x"], n=30, batch_size=4, shuffle=True)
#         interface = OptimLieselInterface(
#             model, n_train=Bi.n, n_validation=Bi.n, batch_size=Bi.batch_size
#         )

#         splitter = TrainValidationTestSplit(
#             position_keys=["x"], n=x.value.size, share_validation=0.1, share_test=0.0
#         )

#         split = splitter.split_state(interface, model.state)
#         model.state = split.train_state

#         def value_and_grad(pos, model_state, obs_batch):
#             val_and_grad = jax.value_and_grad(interface.scaled_neg_log_prob, argnums=0)
#             val, grad_tree = val_and_grad(pos, model_state, obs_batch)
#             return val, grad_tree

#         def grad(pos, model_state, obs_batch):
#             grad_ = jax.grad(interface.scaled_neg_log_prob, argnums=0)
#             grad_tree = grad_(pos, model_state, obs_batch)
#             return grad_tree

#         max_iter = 5

#         def inner_loop_over_batches(j, val):
#             grads, loss, loss_val, i, pos, Bi = val

#             # update the batch number
#             Bi.batch_number = j
#             obs_batch = Bi.extract_batched_position(interface, model.state)

#             # grad_ij = log_prob_grad(pos, Bi)
#             # grad_ij = BatchedLogProb(interface, model.state, Bi).grad(pos)
#             # grad_ij = nlp_grad(pos, model.state, obs_batched)

#             grad_ij = grad(pos, model.state, obs_batch)
#             grad_ij_arr = jax.flatten_util.ravel_pytree(grad_ij)[0].squeeze()
#             # loss_ij, grad_ij = value_and_grad(pos, model.state, obs_batch)

#             grads = grads.at[i, j].set(grad_ij_arr)
#             return grads, loss, loss_val, i, pos, Bi

#         def outer_loop_over_iterations(i, val):
#             grads, loss, loss_val, Bi, key = val

#             # permuting the batch indices for each outer iteration
#             key, subkey = jax.random.split(key)
#             Bi.indices = Bi.permute_indices(subkey)

#             # just for testing, using the same position throughout
#             # helps discover problems
#             pos = {"m": 1.0}

#             # run all full batches once
#             grads, loss, loss_val, _, pos, Bi = jax.lax.fori_loop(
#                 lower=0,
#                 upper=Bi.n_full_batches,
#                 body_fun=inner_loop_over_batches,
#                 init_val=(grads, loss, loss_val, i, pos, Bi),
#             )

#             loss_i = interface.scaled_neg_log_prob(pos, model.state)
#             loss_val_i = interface.scaled_neg_log_lik(
#                 pos, model.state, obs_validation=split.validation_position
#             )

#             loss = loss.at[i].set(loss_i)
#             loss_val = loss_val.at[i].set(loss_val_i)

#             return grads, loss, loss_val, Bi, key

#         grads_init = jnp.zeros((max_iter, Bi.n_full_batches))
#         loss_init = jnp.zeros((max_iter,))
#         loss_val_init = jnp.zeros((max_iter,))

#         grads, loss, loss_val, _, _ = jax.lax.fori_loop(
#             lower=0,
#             upper=max_iter,
#             body_fun=outer_loop_over_iterations,
#             init_val=(grads_init, loss_init, loss_val_init, Bi, key(0)),
#         )

#         assert not jnp.any(jnp.isnan(grads))

#         for i in range(max_iter - 1):
#             assert not jnp.allclose(grads[i], grads[i + 1])

#         for i in range(Bi.n_full_batches - 1):
#             assert not jnp.allclose(grads[:, i], grads[:, i + 1])


# class TestTrainValidationTestSplit:
#     def test_no_split(self):
#         m = lsl.Var.new_param(0.0, name="m")
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         split = TrainValidationTestSplit(
#             position_keys=["x"],
#             n=x.value.size,
#             share_validation=0.0,
#             share_test=0.0,
#         )

#         assert split.indices_train.size == x.value.size
#         assert split.indices_test.size == 0
#         assert split.indices_validation.size == 0

#         pos = model.extract_position(["x"])
#         split_pos = split.split_position(pos)
#         assert split_pos.train["x"].size == x.value.size
#         assert split_pos.validation["x"].size == 0
#         assert split_pos.test["x"].size == 0

#     def test_split(self):
#         m = lsl.Var.new_param(0.0, name="m")
#         x = lsl.Var.new_obs(
#             jnp.arange(100),
#             distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         split = TrainValidationTestSplit(
#             position_keys=["x"],
#             n=x.value.size,
#             share_validation=0.2,
#             share_test=0.2,
#         )

#         assert split.indices_train.size == 60
#         assert split.indices_test.size == 20
#         assert split.indices_validation.size == 20

#         pos = model.extract_position(["x"])
#         split_pos = split.split_position(pos)
#         assert split_pos.train["x"].size == 60
#         assert split_pos.validation["x"].size == 20
#         assert split_pos.test["x"].size == 20

#     def test_split_incoherent(self):
#         with pytest.raises(ValueError):
#             TrainValidationTestSplit(
#                 position_keys=["x"],
#                 n=100,
#                 share_validation=0.9,
#                 share_test=0.2,
#             )

#         with pytest.raises(ValueError):
#             TrainValidationTestSplit(
#                 position_keys=["x"],
#                 n=100,
#                 share_validation=1.1,
#                 share_test=0.0,
#             )

#         with pytest.raises(ValueError):
#             TrainValidationTestSplit(
#                 position_keys=["x"],
#                 n=100,
#                 share_validation=0.0,
#                 share_test=1.1,
#             )

#         with pytest.raises(ValueError):
#             TrainValidationTestSplit(
#                 position_keys=["x"],
#                 n=100,
#                 share_validation=-0.3,
#                 share_test=0.5,
#             )


# class TestOptimEngine:
#     def test_optim(self):
#         m = lsl.Var.new_param(0.0, name="m")
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
#             name="x",
#         )
#         model = lsl.Model([x])

#         split = TrainValidationTestSplit(
#             position_keys=["x"],
#             n=x.value.size,
#             share_validation=0.0,
#             share_test=0.0,
#         )

#         bi = BatchIndices(["x"], n=split.n_train, batch_size=5)
#         interface = OptimLieselInterface(
#             model,
#             n_train=split.n_train,
#             n_validation=split.n_validation,
#             batch_size=bi.batch_size,
#         )
#         data = split.split_state(interface, model.state)

#         optim = Optimizer(["m"], optimizer=optax.adam(learning_rate=1e-2))

#         engine = OptimEngine(
#             interface=interface,
#             batching_indices=bi,
#             data=data,
#             optimizers=[optim],
#             restore_best_position=True,
#             prune_history=True,
#             show_progress=True,
#             track_keys=None,
#             save_position_history=True,
#         )

#         stopper = Stopper(max_iter=100, patience=5)
#         carry = engine._fit(stopper, jax.random.key(1))
#         result = engine.fit(stopper, jax.random.key(1))
#         result.history.loss_train
#         result.history.loss_validation


# class TestElbo:
#     def test_runs(self):
#         m = lsl.Var.new_param(0.0, name="m")
#         x = lsl.Var.new_obs(
#             jnp.arange(30),
#             distribution=lsl.Dist(tfd.Normal, loc=m, scale=1.0),
#             name="x",
#         )
#         p = lsl.Model([x])

#         mq_loc = lsl.Var.new_param(0.0, name="loc")
#         mq_scale = lsl.Var.new_param(1.0, name="scale")
#         mq = lsl.Var.new_obs(
#             0.0, distribution=lsl.Dist(tfd.Normal, loc=mq_loc, scale=mq_scale), name="m"
#         )

#         q = lsl.Model([mq])

#         from liesel.experimental.batching import Elbo

#         pos = q.extract_position(["loc", "scale"])

#         elbo = Elbo(p, q)
#         elbo.evaluate(pos, key=jax.random.key(1))
#         jax.jit(elbo.evaluate)(pos, jax.random.key(1), {}, p.state, q.state)

#         pos = q.extract_position(["loc", "scale"])
#         pos["loc"] = 1.0
#         pos["scale"] = 0.1
#         jax.jit(elbo.evaluate)(pos, jax.random.key(1), {}, p.state, q.state)
#         jax.grad(elbo.evaluate, argnums=0)(pos, jax.random.key(1), {}, p.state, q.state)
#         jax.jit(jax.grad(elbo.evaluate, argnums=0))(
#             pos, jax.random.key(1), {}, p.state, q.state
#         )
