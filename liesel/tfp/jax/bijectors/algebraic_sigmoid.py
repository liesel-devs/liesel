# GENERATED FILE, DO NOT EDIT!

"""
# An algebraic sigmoid bijector
"""

import jax.numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

__docformat__ = "google"


class AlgebraicSigmoid(tfb.Bijector):
    """
    The algebraic sigmoid bijector `f(x) = x / sqrt(1 + x^2)`.
    """

    def __init__(self, validate_args=False, name="algebraic_sigmoid"):
        parameters = dict(locals())

        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    def _forward(self, x):
        return x / np.sqrt(1.0 + x**2)

    def _inverse(self, y):
        return y / np.sqrt(1.0 - y**2)

    def _inverse_log_det_jacobian(self, y):
        return -1.5 * np.log(1.0 - y**2)

    def _forward_log_det_jacobian(self, x):
        return -1.5 * np.log(1.0 + x**2)

    @classmethod
    def _is_increasing(cls):
        return True
