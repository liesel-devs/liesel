# GENERATED FILE, DO NOT EDIT!

"""
# A normal smooth prior for semi-parametric regression
"""

import jax.numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf

__docformat__ = "google"


class SmoothPrior(tfd.Distribution):
    """
    A (potentially rank-deficient) normal smooth prior for semi-parametric regression.

    See Ludwig Fahrmeir et al., [Regression (2013), Section 8.1.3](
    https://link.springer.com/book/10.1007/978-3-642-34333-9) for details.
    """

    def __init__(
        self,
        tau2,
        K,
        rank=None,
        validate_args=False,
        allow_nan_stats=True,
        name="SmoothPrior",
    ):
        """
        Constructs the distribution.

        Args:
            tau2: The smoothing parameter.
            K: The (potentially rank-deficient) penalty matrix.
            rank: The rank of the penalty matrix. Computed from `K` if not provided.
        """

        parameters = dict(locals())

        if rank is None:
            event_shape = np.shape(K)[-1]
            signature = f"({event_shape},{event_shape})->()"
            rank_fn = np.vectorize(np.linalg.matrix_rank, signature=signature)
            rank = rank_fn(K)

        self._rank = rank
        self._tau2 = tau2
        self._K = K

        super().__init__(
            dtype=K.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    def _log_prob(self, x):
        x = np.expand_dims(x, axis=-2)
        x_T = np.swapaxes(x, -2, -1)

        p1 = -0.5 * self._rank * np.log(self._tau2)
        p2 = -0.5 / self._tau2 * np.squeeze(x @ self._K @ x_T, axis=(-2, -1))
        return p1 + p2

    def _event_shape(self):
        return tf.TensorShape((np.shape(self._K)[-1],))

    def _event_shape_tensor(self):
        return np.array((np.shape(self._K)[-1],), dtype=np.int32)

    def _batch_shape(self):
        return tf.TensorShape(np.shape(self._tau2))

    def _batch_shape_tensor(self):
        return np.array(np.shape(self._tau2), dtype=np.int32)
