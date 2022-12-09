"""
The degenerate, i.e. rank-deficient, multivariate normal distribution.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf

Array = Any


def _rank_and_log_pdet(
    prec: Array,
    rank: Array | int | None = None,
    log_pdet: Array | float | None = None,
    tol: float = 1e-6,
) -> tuple[Array | float, Array | int]:
    """
    Computes the rank and the log-pseudo-determinant of the positive semi-definite
    precision matrix ``prec``.

    Can handle batches.

    If both the rank and the determinant are provided, the function does nothing and
    just returns the provided arguments. If the rank is provided, it is used to select
    the non-zero eigenvalues. If the rank is not provided, it is computed by counting
    the non-zero eigenvalues. An eigenvalue is deemed to be non-zero if it is greater
    than the numerical tolerance ``tol``.
    """

    if log_pdet is not None and rank is not None:
        return rank, log_pdet

    eigenvals = jnp.linalg.eigvalsh(prec)

    if rank is None:
        mask = eigenvals > tol
        rank = jnp.sum(mask, axis=-1)
    else:
        max_index = eigenvals.shape[-1] - rank

        def fn(i, x):
            return x.at[..., i].set(i >= max_index)

        mask = jax.lax.fori_loop(0, eigenvals.shape[-1], fn, eigenvals)

    if log_pdet is None:
        selected = jnp.where(mask, eigenvals, 1.0)
        log_pdet = jnp.sum(jnp.log(selected), axis=-1)

    return rank, log_pdet


class MultivariateNormalDegenerate(tfd.Distribution):
    """
    A potentially degenerate multivariate normal distribution.

    Provides the alternative constructor :meth:`.from_penalty`.

    Parameters
    ----------
    loc
        The location (= mean) vector.
    prec
        The precision matrix (= a pseudo-inverse of the variance-covariance matrix).
    rank
        The rank of the precision matrix. Optional.
    log_pdet
        The log-pseudo-determinant of the precision matrix. Optional.
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

    Notes
    -----
    If they are not provided as arguments, the constructor computes ``rank`` and
    ``log_pdet`` based on the eigenvalues of the precision matrix ``prec``. This
    is an expensive operation and can be avoided by specifying the corresponding
    arguments.
    """

    def __init__(
        self,
        loc: Array,
        prec: Array,
        rank: Array | int | None = None,
        log_pdet: Array | float | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "MultivariateNormalDegenerate",
    ):
        parameters = dict(locals())

        self._prec = prec
        # necessary for correct broadcasting over event size
        self._loc = jnp.atleast_1d(loc)

        if not self._prec.shape[-2] == self._prec.shape[-1]:
            raise ValueError(
                "`prec` must be square (the last two dimensions must be equal)."
            )

        try:
            jnp.broadcast_shapes(self._prec.shape[-1], self._loc.shape[-1])
        except ValueError:
            raise ValueError(
                f"The event sizes of `prec` ({self._prec.shape[-1]}) and `loc` "
                f"({self._loc.shape[-1]}) cannot be broadcast together. If you "
                "are trying to use batches for `loc`, you may need to add a "
                "dimension for the event size."
            )

        self._broadcast_batch_shape = jnp.broadcast_shapes(
            jnp.shape(self._prec)[:-2], jnp.shape(self._loc)[:-1]
        )

        self._rank, self._log_pdet = _rank_and_log_pdet(
            self._prec, rank, log_pdet, tol=1e-6
        )

        super().__init__(
            dtype=prec.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @classmethod
    def from_penalty(
        cls,
        loc: Array,
        var: Array,
        pen: Array,
        rank: Array | int | None = None,
        log_pdet: Array | float | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "MultivariateNormalDegenerate",
    ) -> MultivariateNormalDegenerate:
        """
        Alternative constructor based on a penalty matrix and an inverse smoothing
        parameter.

        Sometimes, the precision matrix of a degenerate multivariate normal
        distribution is decomposed into a penalty matrix ``pen`` and an inverse
        smoothing parameter ``var``. Using this constructor, a degenerate multivariate
        normal distribution can be initialized from such a decomposition.

        Parameters
        ----------
        loc
            The location (= mean) vector.
        var
            The variance (= inverse smoothing) parameter.
        pen
            The (potentially rank-deficient) penalty matrix.
        rank
            The rank of the penalty matrix. Optional.
        log_pdet
            The log-pseudo-determinant of the penalty matrix. Optional.
        validate_args
            Python ``bool``, default ``False``. When ``True`` distribution parameters
            are checked for validity despite possibly degrading runtime performance.
            When ``False`` invalid inputs may silently render incorrect outputs.
        allow_nan_stats
            Python ``bool``, default ``True``. When ``True``, statistics (e.g., mean,
            mode, variance) use the value ``NaN`` to indicate the result is undefined.
            When ``False``, an exception is raised if one or more of the statistic's
            batch members are undefined.
        name
            Python ``str`` name prefixed to ``Ops`` created by this class.

        Warnings
        --------
        If the log-pseudo-determinant is provided as an argument, it must be
        of the penalty matrix ``pen``, **not** of the precision matrix.

        Notes
        -----
        If they are not provided as arguments, the constructor computes ``rank`` and
        ``log_pdet`` based on the eigenvalues of the penalty matrix ``pen``. This is
        an expensive operation and can be avoided by specifying the corresponding
        arguments.
        """

        prec = pen / jnp.expand_dims(var, axis=(-2, -1))
        rank, log_pdet = _rank_and_log_pdet(pen, rank, log_pdet, tol=1e-6)
        log_pdet_prec = log_pdet - rank * jnp.log(var)

        mvnd = cls(
            loc=loc,
            prec=prec,
            rank=rank,
            log_pdet=log_pdet_prec,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        return mvnd

    def _log_prob(self, x: Array) -> Array | float:
        x = x - self._loc
        # necessary for correct broadcasting in the quadratic form
        x = jnp.expand_dims(x, axis=-2)
        x_T = jnp.swapaxes(x, -2, -1)

        prob1 = -jnp.squeeze(x @ self._prec @ x_T, axis=(-2, -1))
        prob2 = self._rank * jnp.log(2 * jnp.pi) - self._log_pdet
        return 0.5 * (prob1 - prob2)

    def _event_shape(self):
        return tf.TensorShape((jnp.shape(self._prec)[-1],))

    def _event_shape_tensor(self):
        return jnp.array((jnp.shape(self._prec)[-1],), dtype=jnp.int32)

    def _batch_shape(self):
        return tf.TensorShape(self._broadcast_batch_shape)

    def _batch_shape_tensor(self):
        return jnp.array(self._broadcast_batch_shape, dtype=jnp.int32)
