import jax.numpy as jnp
import numpy as np
from pytest import approx

import liesel.tfp.jax.bijectors as jb
import liesel.tfp.numpy.bijectors as nb


def _test_bijector(bijector, np):
    assert bijector.forward(0.0) == approx(0.0)
    assert bijector.forward(-1.0) == approx(-1.0 / np.sqrt(2.0))
    assert bijector.forward(1.0) == approx(1.0 / np.sqrt(2.0))

    assert bijector.forward(-9999.0) == approx(-1.0)
    assert bijector.forward(9999.0) == approx(1.0)

    assert bijector.inverse(bijector.forward(np.pi)) == approx(np.pi)


def test_bijector():
    _test_bijector(jb.AlgebraicSigmoid(), jnp)
    _test_bijector(nb.AlgebraicSigmoid(), np)
