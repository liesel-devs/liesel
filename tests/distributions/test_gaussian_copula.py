import jax
import jax.numpy as jnp
from pytest import approx, mark

import liesel.distributions as lsld


def test_default_dependence():
    distribution = lsld.GaussianCopula()

    assert distribution.batch_shape.as_list() == []
    assert distribution.log_prob(jnp.array([0.5, 0.5])) == approx(0.0)


def test_scalar_batch():
    distribution = lsld.GaussianCopula(0.42)

    assert distribution.event_shape.as_list() == [2]
    assert list(distribution.event_shape_tensor()) == [2]

    assert distribution.batch_shape.as_list() == []
    assert list(distribution.batch_shape_tensor()) == []

    x1 = jnp.array([0.2, 0.194])
    x2 = jnp.array([0.311, 0.756])
    x3 = jnp.array([0.76, 0.245])

    p1 = 0.3118741
    p2 = -0.1548549
    p3 = -0.2560579

    assert distribution.log_prob(x1) == approx(p1)
    assert distribution.log_prob(x2) == approx(p2)
    assert distribution.log_prob(x3) == approx(p3)

    x4 = jnp.vstack([x1, x2, x3])
    p4 = jnp.array([p1, p2, p3])

    assert distribution.log_prob(x4) == approx(p4)


def test_matrix_batch():
    distribution = lsld.GaussianCopula(jnp.broadcast_to(0.42, [3, 3]))

    assert distribution.event_shape.as_list() == [2]
    assert list(distribution.event_shape_tensor()) == [2]

    assert distribution.batch_shape.as_list() == [3, 3]
    assert list(distribution.batch_shape_tensor()) == [3, 3]

    x = jnp.array([0.2, 0.194])
    p = jnp.broadcast_to(0.3118741, [3, 3])

    assert distribution.log_prob(x) == approx(p)


@mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_explicit_dependence_dtype_with_x64_enabled(dtype):
    with jax.enable_x64(True):
        rho = jnp.array(0.42, dtype=dtype)
        values = jnp.array([0.2, 0.194], dtype=dtype)
        distribution = lsld.GaussianCopula(rho)
        assert distribution.dtype == dtype
        assert distribution.log_prob(values) == approx(0.3118741)
        assert jnp.isfinite(
            jax.grad(lambda r: lsld.GaussianCopula(r).log_prob(values))(rho)
        )
