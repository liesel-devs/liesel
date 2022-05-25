# GENERATED FILE, DO NOT EDIT!

"""
# Bivariate copulas
"""

import jax.numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.internal import parameter_properties

from ..bijectors import AlgebraicSigmoid

__docformat__ = "google"


class GaussianCopula(tfd.TransformedDistribution):
    """A bivariate Gaussian copula."""

    def __init__(
        self,
        dependence=None,
        validate_args=False,
        allow_nan_stats=True,
        name="GaussianCopula",
    ):
        """
        Constructs the distribution.

        Args:
            dependence: The correlation parameter.
        """

        parameters = dict(locals())

        batch_shape = np.shape(dependence)
        loc = np.zeros(batch_shape + (2,))

        if dependence is None:
            scale_tril = None
        else:
            if validate_args:
                assert np.all(dependence >= 0.0)
                assert np.all(dependence <= 1.0)

            tril11 = np.broadcast_to(1.0, batch_shape)
            tril12 = np.broadcast_to(0.0, batch_shape)
            tril22 = np.sqrt(1.0 - dependence**2)

            tril1 = np.stack([tril11, tril12], axis=-1)
            tril2 = np.stack([dependence, tril22], axis=-1)
            scale_tril = np.stack([tril1, tril2], axis=-2)

        distribution = tfd.MultivariateNormalTriL(
            loc=loc,
            scale_tril=scale_tril,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
        )

        bijector = tfb.NormalCDF(validate_args=validate_args)

        super().__init__(
            distribution=distribution,
            bijector=bijector,
            validate_args=validate_args,
            name=name,
        )

        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return {
            "dependence": parameter_properties.ParameterProperties(
                shape_fn=lambda sample_shape: sample_shape[:-1],
                default_constraining_bijector_fn=lambda: AlgebraicSigmoid(),
            )
        }
