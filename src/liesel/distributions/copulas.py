"""
The bivariate Gaussian copula.
"""

import jax.numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.internal import parameter_properties

from ..bijectors import AlgebraicSigmoid


class GaussianCopula(tfd.TransformedDistribution):
    """
    The bivariate Gaussian copula.

    Parameters
    ----------
    dependence
        The correlation parameter.
    validate_args
        Python ``bool``, default ``False``. When ``True``, distribution parameters \
        are checked for validity despite possibly degrading runtime performance. \
        When ``False``, invalid inputs may silently render incorrect outputs.
    allow_nan_stats
        Python ``bool``, default ``True``. When ``True``, statistics (e.g., mean, \
        mode, variance) use the value ``NaN`` to indicate the result is undefined. \
        When ``False``, an exception is raised if one or more of the statistic's \
        batch members are undefined.
    name
        Python ``str``, name prefixed to ``Ops`` created by this class.

    Examples
    --------

    This is an example for how to use this class with Liesel.

    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.bijectors as tfb
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.distributions import GaussianCopula

    >>> correlation = lsl.Var.new_param(0.2, bijector=tfb.Tanh(), name="rho")
    >>> correlation.bijected_var
    Var(name="h(rho)")

    >>> dist1 = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
    >>> x1 = lsl.Var.new_obs(jnp.array([-0.5, -0.1, -0.25]), dist1).update()
    >>> dist2 = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
    >>> x2 = lsl.Var.new_obs(jnp.array([0.5, 0.1, 0.25]), dist2).update()

    >>> x1_pit = lsl.PIT(x1, name="PIT(x1)").update()
    >>> x2_pit = lsl.PIT(x2, name="PIT(x2)").update()
    >>> x_dependence = lsl.Var.new_calc(
    ...     lambda *inputs: jnp.stack(inputs, axis=1),
    ...     x1_pit,
    ...     x2_pit,
    ...     distribution=lsl.Dist(GaussianCopula, dependence=correlation),
    ...     name="x_dependence",
    ... ).update()

    >>> (x1.log_prob + x2.log_prob + x_dependence.log_prob).sum().round(1)
    Array(-5.9, dtype=float32)

    Comparing to an ordinary multivariate normal:

    >>> dist = tfd.MultivariateNormalFullCovariance(
    ...     loc=jnp.zeros(2), covariance_matrix=jnp.array([[1.0, 0.2], [0.2, 1.0]])
    ... )
    >>> dist.log_prob(jnp.stack((x1.value, x2.value), axis=-1)).sum().round(1)
    Array(-5.9, dtype=float32)
    """

    def __init__(
        self,
        dependence=None,
        validate_args=False,
        allow_nan_stats=True,
        name="GaussianCopula",
    ):
        parameters = dict(locals())

        batch_shape = np.shape(dependence)
        loc = np.zeros(batch_shape + (2,))

        if dependence is None:
            scale_tril = None
        else:
            if validate_args:
                assert np.all(dependence >= -1.0)
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
