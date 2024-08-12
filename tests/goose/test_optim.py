from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
from liesel.goose.optim import (
    Stopper,
    _generate_batch_indices,
    _validate_log_prob_decomposition,
    history_to_df,
    optim_flat,
)
from liesel.goose.types import Array

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
target_params = 0.5

# Generate some data.
xs = jax.random.normal(key, (100, 2))
ys = jnp.sum(xs * target_params, axis=-1) + jax.random.normal(subkey, (100,))

ols_coef = np.linalg.inv(xs.T @ xs) @ xs.T @ ys


def setup_model(ys: Array, xs: Array) -> lsl.Model:
    x = lsl.obs(xs, name="x")
    coef = lsl.param(jnp.zeros(2), name="coef")
    mu = lsl.Var(lsl.Calc(jnp.dot, x, coef), name="mu")
    log_sigma = lsl.Var(2.0, name="log_sigma")
    sigma = lsl.Var(lsl.Calc(jnp.exp, log_sigma), name="sigma")

    ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
    y = lsl.obs(ys, ydist, name="y")

    gb = lsl.GraphBuilder().add(y)
    model = gb.build_model()
    return model


def train_test_split(model_fn, test_share: float = 0.20, **model_args):
    train = dict()
    test = dict()
    for name, value in model_args.items():
        train_idx = int(len(value) * (1 - test_share)) + 1
        test_idx = int(len(value) * test_share)

        train[name] = value[:train_idx]
        test[name] = value[-test_idx:]

    return model_fn(**train), model_fn(**test)


@pytest.fixture
def models() -> Iterator[tuple[lsl.Model, lsl.Model]]:
    yield train_test_split(setup_model, xs=xs, ys=ys)


class TestOptim:
    def test_optim_flat_jointly(self, models):
        model, _ = models

        xs = model.vars["x"].value
        ys = model.vars["y"].value
        ols_coef = np.linalg.inv(xs.T @ xs) @ xs.T @ ys
        ols_sigma = jnp.log((ys - (xs @ ols_coef)).std())

        stopper = Stopper(max_iter=1_000, patience=30)
        result = optim_flat(
            model, ["coef", "log_sigma"], batch_size=None, stopper=stopper
        )
        assert jnp.allclose(result.position["coef"], ols_coef, atol=1e-2)
        assert jnp.allclose(result.position["log_sigma"], ols_sigma, atol=1e-2)

    def test_optim_no_early_stop(self, models):
        model, _ = models

        stopper = Stopper(max_iter=1_000, patience=1_000)
        result = optim_flat(
            model, ["coef", "log_sigma"], batch_size=None, stopper=stopper
        )

        assert result.iteration == 1_000
        assert result.iteration_best < result.iteration

    def test_optim_flat_batched(self, models):
        model, _ = models

        stopper = Stopper(max_iter=1000, patience=30)
        result_batched_small = optim_flat(
            model,
            ["coef", "log_sigma"],
            batch_size=10,
            batch_seed=1,
            stopper=stopper,
            model_validation=model,
        )
        result_batched_big = optim_flat(
            model,
            ["coef", "log_sigma"],
            batch_size=40,
            batch_seed=1,
            stopper=stopper,
            model_validation=model,
        )
        result_nonbatched = optim_flat(
            model, ["coef", "log_sigma"], stopper=stopper, model_validation=model
        )

        assert result_batched_small.iteration != result_nonbatched.iteration
        assert result_batched_small.iteration != result_batched_big.iteration
        assert result_nonbatched.iteration != result_batched_big.iteration

    def test_optim_flat_train_validation(self, models):
        model, model_validation = models
        stopper = Stopper(max_iter=1_000, patience=30)

        result_train = optim_flat(
            model, ["coef", "log_sigma"], batch_size=None, stopper=stopper
        )

        result_train_validation = optim_flat(
            model,
            ["coef", "log_sigma"],
            batch_size=None,
            stopper=stopper,
            model_validation=model_validation,
        )

        assert result_train.n_train == result_train.n_validation
        assert result_train_validation.n_train != result_train_validation.n_validation

        assert not jnp.allclose(
            result_train_validation.history["loss_train"],
            result_train_validation.history["loss_validation"],
        )
        assert jnp.allclose(
            result_train.history["loss_train"], result_train.history["loss_validation"]
        )


class TestLogProbDecompositionValidation:
    def test_const_priors(self):
        x = lsl.obs(xs, name="x")
        coef = lsl.param(jnp.zeros(2), name="coef")
        mu = lsl.Var(lsl.Calc(jnp.dot, x, coef), name="mu")
        log_sigma = lsl.Var(2.0, name="log_sigma")
        sigma = lsl.Var(lsl.Calc(jnp.exp, log_sigma), name="sigma")

        ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
        y = lsl.obs(ys, ydist, name="y")

        gb = lsl.GraphBuilder().add(y)
        model = gb.build_model()

        interface = gs.LieselInterface(model)
        position = interface.extract_position(["coef", "log_sigma"], model.state)

        assert _validate_log_prob_decomposition(interface, position, model.state)

    def test_prior(self):
        x = lsl.obs(xs, name="x")
        coef = lsl.param(
            jnp.zeros(2),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=10.0),
            name="coef",
        )
        mu = lsl.Var(lsl.Calc(jnp.dot, x, coef), name="mu")
        log_sigma = lsl.Var(2.0, name="log_sigma")
        sigma = lsl.Var(lsl.Calc(jnp.exp, log_sigma), name="sigma")

        ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
        y = lsl.obs(ys, ydist, name="y")

        gb = lsl.GraphBuilder().add(y)
        model = gb.build_model()

        interface = gs.LieselInterface(model)
        position = interface.extract_position(["coef", "log_sigma"], model.state)

        assert _validate_log_prob_decomposition(interface, position, model.state)

    def test_prior_but_not_param(self):
        x = lsl.obs(xs, name="x")
        coef = lsl.Var(
            jnp.zeros(2),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=10.0),
            name="coef",
        )
        mu = lsl.Var(lsl.Calc(jnp.dot, x, coef), name="mu")
        log_sigma = lsl.Var(2.0, name="log_sigma")
        sigma = lsl.Var(lsl.Calc(jnp.exp, log_sigma), name="sigma")

        ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
        y = lsl.obs(ys, ydist, name="y")

        gb = lsl.GraphBuilder().add(y)
        model = gb.build_model()

        interface = gs.LieselInterface(model)
        position = interface.extract_position(["coef", "log_sigma"], model.state)

        with pytest.raises(ValueError, match="cannot correctly be decomposed"):
            _validate_log_prob_decomposition(interface, position, model.state)

    def test_not_obs(self):
        x = lsl.obs(xs, name="x")
        coef = lsl.param(
            jnp.zeros(2),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=10.0),
            name="coef",
        )
        mu = lsl.Var(lsl.Calc(jnp.dot, x, coef), name="mu")
        log_sigma = lsl.Var(2.0, name="log_sigma")
        sigma = lsl.Var(lsl.Calc(jnp.exp, log_sigma), name="sigma")

        ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
        y = lsl.Var(ys, ydist, name="y")

        gb = lsl.GraphBuilder().add(y)
        model = gb.build_model()

        interface = gs.LieselInterface(model)
        position = interface.extract_position(["coef", "log_sigma"], model.state)

        with pytest.raises(ValueError, match="cannot correctly be decomposed"):
            _validate_log_prob_decomposition(interface, position, model.state)


def test_position_history_to_df(models):
    model, _ = models
    stopper = Stopper(max_iter=1_000, patience=10)
    result = optim_flat(
        model,
        ["coef", "log_sigma"],
        batch_size=20,
        stopper=stopper,
        batch_seed=1,
        prune_history=False,
    )
    df = history_to_df(result.history["position"])

    assert df.shape == (1000, 4)


def test_full_history_to_df(models):
    model, _ = models
    stopper = Stopper(max_iter=1_000, patience=10)
    result = optim_flat(
        model,
        ["coef", "log_sigma"],
        batch_size=20,
        stopper=stopper,
        batch_seed=1,
        prune_history=False,
    )
    df = history_to_df(result.history)

    assert df.shape == (1000, 6)


def test_history_to_df_pruned(models):
    model, _ = models
    stopper = Stopper(max_iter=1_000, patience=10)
    result = optim_flat(
        model,
        ["coef", "log_sigma"],
        batch_size=20,
        stopper=stopper,
        batch_seed=1,
        prune_history=True,
        model_validation=model,
    )
    df = history_to_df(result.history["position"])

    assert df.shape == (66, 4)


def test_generate_batches():
    key = jax.random.PRNGKey(42)
    jit_batches = jax.jit(_generate_batch_indices, static_argnames=["n", "batch_size"])
    batches = jit_batches(key, n=30, batch_size=9)
    assert len(batches) == 3

    for batch in batches:
        assert len(batch) == 9


class TestStopper:
    def test_stopper_does_not_stop(self):
        stopper = Stopper(patience=5, max_iter=100)

        key = jax.random.PRNGKey(42)
        loss_history = jax.random.uniform(key, shape=(15,))
        loss_history = loss_history.at[6].set(-0.1)

        stop = stopper.stop_early(i=6, loss_history=loss_history)

        assert not stop

    def test_stopper_stops(self):
        stopper = Stopper(patience=5, max_iter=100, atol=0.1, rtol=0.1)

        key = jax.random.PRNGKey(42)
        loss_history = 2.0 + jax.random.uniform(
            key, shape=(15,), minval=0.0, maxval=0.1
        )

        stop = stopper.stop_early(i=6, loss_history=loss_history)
        stop_at = stopper.which_best_in_recent_history(i=6, loss_history=loss_history)

        assert stop
        assert stop_at == jnp.argmin(loss_history)

    def test_stopper_jitted(self):
        stopper = Stopper(patience=5, max_iter=100, atol=0.1, rtol=0.1)
        stop_jit = jax.jit(stopper.stop_early)

        key = jax.random.PRNGKey(42)
        loss_history = jax.random.uniform(key, shape=(15,))
        stop = stop_jit(i=6, loss_history=loss_history)
        assert not stop

    def test_stop_at_jitted(self):
        stopper = Stopper(patience=5, max_iter=100, atol=0.1, rtol=0.1)
        stop_at_jit = jax.jit(stopper.which_best_in_recent_history)

        key = jax.random.PRNGKey(42)
        loss_history = 2.0 + jax.random.uniform(
            key, shape=(15,), minval=0.0, maxval=0.1
        )

        stop = stopper.stop_early(i=6, loss_history=loss_history)
        stop_at = stop_at_jit(i=6, loss_history=loss_history)

        assert stop
        assert loss_history[stop_at] == pytest.approx(jnp.min(loss_history))

    def test_stop_at_start(self):
        stopper = Stopper(max_iter=100, patience=5, atol=0.1, rtol=0.1)
        loss_history = jnp.zeros(shape=(10,))

        stop = stopper.stop_now(i=0, loss_history=loss_history)

        assert not stop

    def test_patience(self):
        stopper = Stopper(max_iter=100, patience=100)
        loss_history = jnp.zeros(shape=(100,))

        for i in range(100):
            assert not stopper.stop_early(i, loss_history)
