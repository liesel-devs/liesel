from __future__ import annotations

from functools import partial

import jax
import numpy as np
import pytest

import liesel.goose as gs

try:
    import pymc as pm

    from liesel.experimental.pymc import PyMCInterface
except ImportError:
    pytestmark = pytest.mark.skip(reason="need pymc to run")

roughly_close = partial(np.allclose, rtol=0.1, atol=0.01)


@pytest.fixture
def basic_lm():
    RANDOM_SEED = 123
    rng = np.random.RandomState(RANDOM_SEED)

    # set parameter values
    num_obs = 100
    sigma = 1.0
    beta = [1.0, 1.0, 2.0]

    # simulate covariates
    x1 = rng.randn(num_obs)
    x2 = 0.5 * rng.randn(num_obs)

    # Simulate outcome variable
    y = beta[0] + beta[1] * x1 + beta[2] * x2 + sigma * rng.normal(size=num_obs)

    x1 = np.asarray(x1, np.float32)
    x2 = np.asarray(x2, np.float32)
    y = np.asarray(y, np.float32)

    basic_model = pm.Model()
    with basic_model:
        # priors
        beta = pm.Normal("beta", mu=0.0, sigma=10.0, shape=3)
        sigma = pm.HalfNormal(
            "sigma", sigma=1.0
        )  # automatically transformed to real via log

        # predicted value
        mu = beta[0] + beta[1] * x1 + beta[2] * x2

        # track the predicted value of the first obs
        pm.Deterministic("mu[0]", mu[0])

        # distribution of response (likelihood)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    return basic_model


def test_simple():
    model = pm.Model()

    with model:
        _ = pm.Normal("mu", mu=0.0, sigma=1.0)

    interface = PyMCInterface(model=model)
    state = interface.get_initial_state()
    assert state["mu"] == 0.0
    assert roughly_close(interface.log_prob(state), -0.91893853)

    state = interface.update_state({"mu": np.array(1.0)}, state)
    assert state["mu"] == 1.0
    assert roughly_close(interface.log_prob(state), -1.41893853)


def test_default_dtype():
    model = pm.Model()
    with model:
        _ = pm.Normal("mu", mu=0.0, sigma=1.0)

    interface = PyMCInterface(model=model)
    state = interface.get_initial_state()

    assert not jax.config.read("jax_enable_x64")
    assert interface.log_prob(state).dtype == np.float32


@pytest.mark.filterwarnings(
    "ignore:Explicitly requested dtype float64 requested in astype is not available"
    ".*:UserWarning"
)
def test_simple2():
    model = pm.Model()

    with model:
        sigma = pm.HalfNormal(
            "sigma",
            sigma=1.0,
        )
        _ = pm.Normal("mu", mu=0.0, sigma=sigma)

    interface = PyMCInterface(model=model, additional_vars=["sigma"])
    state = interface.get_initial_state()

    assert state["mu"] == 0.0
    assert state["sigma_log__"] == 0.0
    assert roughly_close(interface.log_prob(state), -0.91893853 + -0.72579135)
    assert interface.extract_position(["sigma"], state)["sigma"] == 1.0

    state = interface.update_state(
        {"mu": np.array(1.0), "sigma_log__": np.array(-1.0)}, state
    )
    assert state["mu"] == 1.0
    assert state["sigma_log__"] == -1.0
    assert roughly_close(interface.log_prob(state), -3.613467 + -1.29345899)


@pytest.mark.mcmc
def test_mcmc(basic_lm: pm.Model):  # type: ignore
    interface = PyMCInterface(basic_lm, additional_vars=["sigma", "mu[0]"])
    state = interface.get_initial_state()
    builder = gs.EngineBuilder(1, 2)
    builder.set_initial_values(state)
    builder.set_model(interface)
    builder.set_duration(1000, 2000)

    builder.add_kernel(gs.NUTSKernel(["beta", "sigma_log__"]))

    builder.positions_included = ["sigma", "mu[0]"]

    engine = builder.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    sum = gs.Summary(results)

    assert roughly_close(
        sum.quantities["mean"]["beta"], [0.9068794, 0.94820887, 2.0448447]
    )
    assert roughly_close(sum.quantities["mean"]["sigma_log__"], -0.01403057)
    assert roughly_close(sum.quantities["mean"]["sigma"], 0.98853235)
    assert roughly_close(sum.quantities["mean"]["mu[0]"], 0.5339259)


@pytest.mark.mcmc
def test_mcmc_two_kernels(basic_lm: pm.Model):  # type: ignore
    interface = PyMCInterface(basic_lm, additional_vars=["sigma", "mu[0]"])
    state = interface.get_initial_state()
    builder = gs.EngineBuilder(1, 2)
    builder.set_initial_values(state)
    builder.set_model(interface)
    builder.set_duration(1000, 2000)

    builder.add_kernel(gs.NUTSKernel(["beta"]))
    builder.add_kernel(gs.NUTSKernel(["sigma_log__"]))

    builder.positions_included = ["sigma", "mu[0]"]

    engine = builder.build()

    engine.sample_all_epochs()
    results = engine.get_results()
    sum = gs.Summary(results)

    assert roughly_close(
        sum.quantities["mean"]["beta"], [0.90512399, 0.94801362, 2.0486103]
    )
    assert roughly_close(sum.quantities["mean"]["sigma_log__"], -0.01222878)
    assert roughly_close(sum.quantities["mean"]["sigma"], 0.99029397)
    assert roughly_close(sum.quantities["mean"]["mu[0]"], 0.53359132)
