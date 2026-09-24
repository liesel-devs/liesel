"""Keep the custom distribution shown in the model guide executable."""

import re
from pathlib import Path
from textwrap import dedent

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd


def test_custom_distribution_guide():
    guide = Path(__file__).resolve().parents[2] / "docs/source/model-distributions.rst"
    blocks = re.findall(
        r"\.\. code-block:: python\n\n((?:(?:   [^\n]*)?\n)+)", guide.read_text()
    )
    namespace = {}
    for block in blocks:
        exec(dedent(block), namespace)  # noqa: S102 - Execute trusted, checked-in docs.
        if "custom_model" in namespace:
            break

    laplace = namespace["Laplace"]
    values = jnp.array([-1.2, 0.4, 1.7])
    actual = laplace(loc=jnp.array([0.0, 1.0]), scale=2.0)
    expected = tfd.Laplace(loc=jnp.array([0.0, 1.0]), scale=2.0)
    np.testing.assert_allclose(
        actual.copy().log_prob(values[:, None]),
        expected.log_prob(values[:, None]),
        rtol=1e-6,
    )
    actual_grad = jax.jit(
        jax.grad(lambda loc, scale: laplace(loc, scale).log_prob(values).sum(), (0, 1))
    )
    expected_grad = jax.grad(
        lambda loc, scale: tfd.Laplace(loc, scale).log_prob(values).sum(), (0, 1)
    )
    np.testing.assert_allclose(actual_grad(0.3, 2.0), expected_grad(0.3, 2.0))
    assert actual.sample((5,), seed=jax.random.key(1)).shape == (5, 2)
    draws = namespace["custom_model"].sample(
        (6,), seed=jax.random.key(2), fixed=["mu", "scale"]
    )
    assert draws["response"].shape == (6, 3)
    assert bool(jnp.isfinite(draws["response"]).all())
