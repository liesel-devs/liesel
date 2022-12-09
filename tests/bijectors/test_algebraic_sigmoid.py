import jax.numpy as jnp
from pytest import approx

import liesel.bijectors as lslb


def test_bijector():
    bijector = lslb.AlgebraicSigmoid()

    assert bijector.forward(0.0) == approx(0.0)
    assert bijector.forward(-1.0) == approx(-1.0 / jnp.sqrt(2.0))
    assert bijector.forward(1.0) == approx(1.0 / jnp.sqrt(2.0))

    assert bijector.forward(-9999.0) == approx(-1.0)
    assert bijector.forward(9999.0) == approx(1.0)

    assert bijector.inverse(bijector.forward(jnp.pi)) == approx(jnp.pi)
