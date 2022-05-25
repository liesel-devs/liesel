import jax.numpy as jnp
import numpy as np
from pytest import approx

import liesel.tfp.jax.distributions as jd
import liesel.tfp.numpy.distributions as nd


def _test_scalar_batch(distribution, np):
    assert distribution.event_shape.as_list() == [3]
    assert list(distribution.event_shape_tensor()) == [3]

    assert distribution.batch_shape.as_list() == []
    assert list(distribution.batch_shape_tensor()) == []

    x1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([1.0, 1.0, 1.0])

    p1 = 0.0
    p2 = -1.5

    assert distribution.log_prob(x1) == approx(p1)
    assert distribution.log_prob(x2) == approx(p2)

    x3 = np.row_stack([x1, x2])
    p3 = np.array([p1, p2])

    assert distribution.log_prob(x3) == approx(p3)


def _test_vector_batch(distribution, np):
    assert distribution.event_shape.as_list() == [3]
    assert list(distribution.event_shape_tensor()) == [3]

    assert distribution.batch_shape.as_list() == [2]
    assert list(distribution.batch_shape_tensor()) == [2]

    x = np.array([1.0, 1.0, 1.0])
    p = np.array([-1.5, -1.5 - 3.0 / np.e])

    assert distribution.log_prob(x) == approx(p)


def test_distribution():
    _test_scalar_batch(jd.SmoothPrior(tau2=1.0, K=jnp.identity(3)), jnp)
    _test_scalar_batch(nd.SmoothPrior(tau2=1.0, K=np.identity(3)), np)

    tau2 = jnp.array([1.0, jnp.e])

    K1 = jnp.identity(3)
    K2 = 2.0 * jnp.identity(3)
    K = jnp.stack([K1, K2])

    _test_vector_batch(jd.SmoothPrior(tau2=tau2, K=K), jnp)

    tau2 = np.array([1.0, np.e])

    K1 = np.identity(3)
    K2 = 2.0 * np.identity(3)
    K = np.stack([K1, K2])

    _test_vector_batch(nd.SmoothPrior(tau2=tau2, K=K), np)
