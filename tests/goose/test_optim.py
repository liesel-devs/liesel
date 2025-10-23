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
    _find_observed,
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
    x = lsl.Var.new_obs(xs, name="x")
    coef = lsl.Var.new_param(jnp.zeros(2), name="coef")
    mu = lsl.Var.new_calc(jnp.dot, x, coef, name="mu")
    log_sigma = lsl.Var(2.0, name="log_sigma")
    sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")

    ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
    y = lsl.Var.new_obs(ys, ydist, name="y")

    model = lsl.Model([y])
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
        n = ys.shape[-1]
        ols_log_sigma = jnp.log(jnp.sqrt(jnp.square(ys - (xs @ ols_coef)).sum() / n))

        stopper = Stopper(max_iter=1_000, patience=30)
        result = optim_flat(
            model, ["coef", "log_sigma"], batch_size=None, stopper=stopper
        )
        assert jnp.allclose(result.position["coef"], ols_coef, atol=1e-3)
        assert jnp.allclose(result.position["log_sigma"], ols_log_sigma, atol=1e-3)

    def test_optim_no_early_stop(self, models):
        model, _ = models

        stopper = Stopper(max_iter=1_000, patience=1_000)
        result = optim_flat(
            model, ["coef", "log_sigma"], batch_size=None, stopper=stopper
        )

        assert result.iteration == 999
        assert result.iteration_best < result.iteration
        assert result.history["loss_train"].shape == (1000,)
        assert result.history["loss_validation"].shape == (1000,)

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

    def test_track_keys(self, models):
        model, _ = models
        stopper = Stopper(max_iter=10, patience=1)

        result = optim_flat(
            model,
            params=["coef", "log_sigma"],
            batch_size=None,
            stopper=stopper,
            track_keys=["sigma"],
        )

        assert "sigma" in result.history["tracked"]
        assert jnp.allclose(
            result.history["tracked"]["sigma"],
            jnp.exp(result.history["position"]["log_sigma"]),
        )


class TestLogProbDecompositionValidation:
    def test_const_priors(self):
        x = lsl.Var.new_obs(xs, name="x")
        coef = lsl.Var.new_param(jnp.zeros(2), name="coef")
        mu = lsl.Var.new_calc(jnp.dot, x, coef, name="mu")
        log_sigma = lsl.Var(2.0, name="log_sigma")
        sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")

        ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
        y = lsl.Var.new_obs(ys, ydist, name="y")

        model = lsl.Model([y])

        interface = gs.LieselInterface(model)
        position = interface.extract_position(["coef", "log_sigma"], model.state)

        assert _validate_log_prob_decomposition(interface, position, model.state)

    def test_prior(self):
        x = lsl.Var.new_obs(xs, name="x")
        coef = lsl.Var.new_param(
            jnp.zeros(2),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=10.0),
            name="coef",
        )
        mu = lsl.Var.new_calc(jnp.dot, x, coef, name="mu")
        log_sigma = lsl.Var(2.0, name="log_sigma")
        sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")

        ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
        y = lsl.Var.new_obs(ys, ydist, name="y")

        model = lsl.Model([y])

        interface = gs.LieselInterface(model)
        position = interface.extract_position(["coef", "log_sigma"], model.state)

        assert _validate_log_prob_decomposition(interface, position, model.state)

    def test_prior_but_not_param(self):
        x = lsl.Var.new_obs(xs, name="x")
        coef = lsl.Var(
            jnp.zeros(2),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=10.0),
            name="coef",
        )
        mu = lsl.Var.new_calc(jnp.dot, x, coef, name="mu")
        log_sigma = lsl.Var(2.0, name="log_sigma")
        sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")

        ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
        y = lsl.Var.new_obs(ys, ydist, name="y")

        model = lsl.Model([y])

        interface = gs.LieselInterface(model)
        position = interface.extract_position(["coef", "log_sigma"], model.state)

        with pytest.raises(ValueError, match="cannot correctly be decomposed"):
            _validate_log_prob_decomposition(interface, position, model.state)

    def test_not_obs(self):
        x = lsl.Var.new_obs(xs, name="x")
        coef = lsl.Var.new_param(
            jnp.zeros(2),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=10.0),
            name="coef",
        )
        mu = lsl.Var.new_calc(jnp.dot, x, coef, name="mu")
        log_sigma = lsl.Var(2.0, name="log_sigma")
        sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")

        ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
        y = lsl.Var(ys, ydist, name="y")

        model = lsl.Model([y])

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
        track_keys=["sigma"],
    )
    df = history_to_df(result.history)

    assert df.shape == (1000, 7)


def test_history_to_df_pruned(models):
    model, _ = models
    stopper = Stopper(max_iter=1_000, patience=10)
    with jax.disable_jit(disable=False):
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

    assert df.shape[0] < 100


def test_generate_batches():
    key = jax.random.PRNGKey(42)
    jit_batches = jax.jit(_generate_batch_indices, static_argnames=["n", "batch_size"])
    batches = jit_batches(key, n=30, batch_size=9)
    assert len(batches) == 3

    for batch in batches:
        assert len(batch) == 9


def test_find_observed():
    n = 10
    a = lsl.Var.new_obs(jnp.arange(n), name="a")
    model = lsl.Model([a])
    observed = _find_observed(model)

    assert "a" in observed


def test_find_observed_weak():
    n = 10
    a = lsl.Var.new_obs(jnp.arange(n), name="a")
    b_calc = lsl.Var.new_calc(lambda x: x + 1, a)
    b = lsl.Var.new_obs(b_calc, name="b")
    model = lsl.Model([b])
    observed = _find_observed(model)

    assert "a" in observed
    assert "b" not in observed


class TestStopper:
    def test_stopper_does_not_stop(self):
        stopper = Stopper(patience=5, max_iter=100)

        # case 1: continous decrease
        loss_history = np.linspace(100, 0, 100)
        stop = stopper.stop_early(i=6, loss_history=loss_history)
        assert not stop

        # case 2: current loss is not the best
        loss_history = np.zeros((100,))
        loss_history[5] = -1.0
        stop = stopper.stop_early(i=6, loss_history=loss_history)
        assert not stop

    def test_stopper_stops_atol(self):
        stopper = Stopper(patience=5, max_iter=100, atol=0.1)

        loss_history = np.zeros((100,))
        loss_history[2:7] = np.array([0.0, 1.0, -0.05, -0.05, -0.01])
        stop = stopper.stop_early(i=6, loss_history=loss_history)
        assert stop

    def test_stopper_stops_rtol(self):
        stopper = Stopper(patience=5, max_iter=100, atol=0.0, rtol=0.05)

        # case one: 0.0 gets compared to -0.05
        # relative difference is 1
        # so no stop
        loss_history = np.zeros((100,))
        loss_history[2:7] = np.array([0.0, 1.0, -0.05, -0.05, -0.01])
        stop = stopper.stop_early(i=6, loss_history=loss_history)
        assert not stop

        # case one: -0.049 gets compared to -0.05
        # relative difference is ~0.02
        # so stop
        loss_history[2:7] = np.array([-0.049, 1.0, -0.05, -0.05, -0.01])
        stop = stopper.stop_early(i=6, loss_history=loss_history)
        assert stop

    def test_stopper_jitted(self):
        stopper = Stopper(patience=5, max_iter=100, atol=0.1, rtol=0.1)
        stop_jit = jax.jit(stopper.stop_early)

        loss_history = jnp.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        stop = stop_jit(i=6, loss_history=loss_history)
        assert stop

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
