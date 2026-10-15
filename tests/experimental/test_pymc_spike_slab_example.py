"""Check the tutorial's Gibbs update against its actual PyMC target density."""

import re
from itertools import product
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

pm = pytest.importorskip("pymc")

from liesel.experimental.pymc import PyMCInterface  # noqa: E402

# PyMC's Bernoulli logp casts to float64 even in Liesel's default float32 mode.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Explicitly requested dtype float64 requested in astype is not available"
    ".*:UserWarning"
)


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).parents[2] / "docs/source/tutorials/md/06-pymc.md"
    cells = re.findall(
        r"```\{code-cell\} ipython3\n(.*?)```", path.read_text(), re.DOTALL
    )
    namespace: dict[str, Any] = {
        "jax": jax,
        "jnp": jnp,
        "pm": pm,
        "tfd": tfd,
        "p": 4,
        "X": np.zeros((3, 4), dtype=np.float32),
        "y": np.zeros(3, dtype=np.float32),
    }
    # Load the teaching implementation so this check cannot drift from the page.
    for prefix in ("nu =", "def inclusion_probability"):
        code = next(cell for cell in cells if cell.startswith(prefix))
        exec(compile(code, str(path), "exec"), namespace)  # noqa: S102 - trusted docs
    return namespace, PyMCInterface(namespace["spike_and_slab_model"])


@pytest.mark.parametrize("theta,tau,sigma2", [(0.2, 0.8, 1.7), (0.8, 2.5, 0.6)])
def test_conditional_matches_enumerated_target(example, theta, tau, sigma2):
    namespace, interface = example
    state = {
        name: jnp.asarray(value)
        for name, value in interface.get_initial_state().items()
    }
    state.update(
        beta=jnp.array([0.05, 0.2, 0.4, -0.1]),
        theta_logodds__=jnp.array(np.log(theta / (1 - theta))),
        tau_log__=jnp.array(np.log(tau)),
        sigma2_log__=jnp.array(np.log(sigma2)),
    )
    indicators = jnp.asarray(
        list(product([0, 1], repeat=4)), dtype=state["delta"].dtype
    )
    log_density = jnp.stack(
        [interface.log_prob({**state, "delta": delta}) for delta in indicators]
    )
    exact_joint = jax.nn.softmax(log_density)

    probability = namespace["inclusion_probability"](state)
    gibbs_joint = jnp.prod(
        jnp.where(indicators == 1, probability, 1 - probability),
        axis=1,
    )
    np.testing.assert_allclose(gibbs_joint, exact_joint, rtol=2e-5, atol=2e-6)


def test_jitted_draw_uses_current_state_and_preserves_dtype(example):
    namespace, interface = example
    state = interface.get_initial_state()
    state = {name: jnp.asarray(value) for name, value in state.items()}
    state["beta"] = jnp.zeros(4)
    draw = jax.jit(namespace["draw_indicators"])
    key = jax.random.key(31)

    low = draw(key, {**state, "theta_logodds__": jnp.array(-30.0)})
    high = draw(key, {**state, "theta_logodds__": jnp.array(30.0)})
    assert set(low) == set(high) == {"delta"}
    assert low["delta"].shape == high["delta"].shape == state["delta"].shape
    assert low["delta"].dtype == high["delta"].dtype == state["delta"].dtype
    np.testing.assert_array_equal(low["delta"], 0)
    np.testing.assert_array_equal(high["delta"], 1)
