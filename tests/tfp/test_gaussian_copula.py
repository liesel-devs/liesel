import jax.numpy as jnp
import numpy as np
from pytest import approx

import liesel.tfp.jax.distributions as jd
import liesel.tfp.numpy.distributions as nd


def _test_scalar_batch(distribution, np):
    assert distribution.event_shape.as_list() == [2]
    assert list(distribution.event_shape_tensor()) == [2]

    assert distribution.batch_shape.as_list() == []
    assert list(distribution.batch_shape_tensor()) == []

    x1 = np.array([0.2, 0.194])
    x2 = np.array([0.311, 0.756])
    x3 = np.array([0.76, 0.245])

    p1 = 0.3118741
    p2 = -0.1548549
    p3 = -0.2560579

    assert distribution.log_prob(x1) == approx(p1)
    assert distribution.log_prob(x2) == approx(p2)
    assert distribution.log_prob(x3) == approx(p3)

    x4 = np.row_stack([x1, x2, x3])
    p4 = np.array([p1, p2, p3])

    assert distribution.log_prob(x4) == approx(p4)


def _test_matrix_batch(distribution, np):
    assert distribution.event_shape.as_list() == [2]
    assert list(distribution.event_shape_tensor()) == [2]

    assert distribution.batch_shape.as_list() == [3, 3]
    assert list(distribution.batch_shape_tensor()) == [3, 3]

    x = np.array([0.2, 0.194])
    p = np.broadcast_to(0.3118741, [3, 3])

    assert distribution.log_prob(x) == approx(p)


def test_distribution():
    _test_scalar_batch(jd.GaussianCopula(0.42), jnp)
    _test_scalar_batch(nd.GaussianCopula(0.42), np)

    _test_matrix_batch(jd.GaussianCopula(jnp.broadcast_to(0.42, [3, 3])), jnp)
    _test_matrix_batch(nd.GaussianCopula(np.broadcast_to(0.42, [3, 3])), np)
