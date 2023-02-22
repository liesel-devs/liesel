"""
The degenerate, i.e. rank-deficient, multivariate normal distribution.
"""

from __future__ import annotations

from functools import cached_property
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


def _rank(eigenvalues: Array, tol: float = 1e-6) -> Array | float:
    """
    Computes the rank of a matrix based on the provided eigenvalues. The rank is taken
    to be the number of non-zero eigenvalues.

    Can handle batches.
    """
    mask = eigenvalues > tol
    rank = jnp.sum(mask, axis=-1)
    return rank


def _log_pdet(
    eigenvalues: Array, rank: Array | float | None = None, tol: float = 1e-6
) -> Array | float:
    """
    Computes the log of the pseudo-determinant of a matrix based on the provided
    eigenvalues. If the rank is provided, it is used to select the non-zero eigenvalues.
    If the rank is not provided, it is computed by counting the non-zero eigenvalues. An
    eigenvalue is deemed to be non-zero if it is greater than the numerical tolerance
    ``tol``.

    Can handle batches.
    """
    if rank is None:
        mask = eigenvalues > tol
    else:
        max_index = eigenvalues.shape[-1] - rank

        def fn(i, x):
            return x.at[..., i].set(i >= max_index)

        mask = jax.lax.fori_loop(0, eigenvalues.shape[-1], fn, eigenvalues)

    selected = jnp.where(mask, eigenvalues, 1.0)
    log_pdet = jnp.sum(jnp.log(selected), axis=-1)
    return log_pdet


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

        self._rank = rank
        self._log_pdet = log_pdet

        # necessary for correct broadcasting over event size
        loc = jnp.atleast_1d(loc)

        if not prec.shape[-2] == prec.shape[-1]:
            raise ValueError(
                "`prec` must be square (the last two dimensions must be equal)."
            )

        try:
            jnp.broadcast_shapes(prec.shape[-1], loc.shape[-1])
        except ValueError:
            raise ValueError(
                f"The event sizes of `prec` ({prec.shape[-1]}) and `loc` "
                f"({loc.shape[-1]}) cannot be broadcast together. If you "
                "are trying to use batches for `loc`, you may need to add a "
                "dimension for the event size."
            )

        prec_batches = jnp.shape(prec)[:-2]
        loc_batches = jnp.shape(loc)[:-1]
        self._broadcast_batch_shape = jnp.broadcast_shapes(prec_batches, loc_batches)
        nbatch = len(self.batch_shape)

        self._prec = jnp.expand_dims(prec, jnp.arange(nbatch - len(prec_batches)))
        self._loc = jnp.expand_dims(loc, jnp.arange(nbatch - len(loc_batches)))

        super().__init__(
            dtype=prec.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @cached_property
    def eig(self) -> tuple[Array, Array]:
        """Eigenvalues and eigenvectors of the precision."""
        return jax.scipy.linalg.eigh(self._prec)

    @cached_property
    def _sample_transformer(self) -> Array:
        eigenvalues, evecs = self.eig

        numerically_zero = eigenvalues < 1e-6
        sqrt_eval = jnp.sqrt(1 / eigenvalues).at[numerically_zero].set(0)

        event_shape = sqrt_eval.shape[-1]
        shape = sqrt_eval.shape + (event_shape,)

        r = jnp.arange(event_shape)
        diags = jnp.zeros(shape).at[..., r, r].set(sqrt_eval)
        return evecs @ diags

    @cached_property
    def rank(self) -> Array | float:
        if self._rank is not None:
            return self._rank
        evals, _ = self.eig
        return _rank(evals)

    @cached_property
    def log_pdet(self) -> Array | float:
        """Log pseudo-determinant."""
        if self._log_pdet is not None:
            return self._log_pdet
        evals, _ = self.eig
        return _log_pdet(evals, self.rank)

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

        evals = jax.numpy.linalg.eigvalsh(pen)
        rank = _rank(evals) if rank is None else rank
        log_pdet = _log_pdet(evals, rank=rank) if log_pdet is None else log_pdet
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

    def _sample_n(self, n, seed=None) -> Array:
        shape = [n] + self.batch_shape + self.event_shape

        # The added dimension at the end here makes sure that matrix multiplication
        # with the "transformer" matrices works out correctly.
        z = jax.random.normal(key=seed, shape=shape + [1])

        # Add a dimension at 0 for the sample size.
        transformer = jnp.expand_dims(self._sample_transformer, 0)
        centered_samples = jnp.reshape(transformer @ z, shape)

        # Add a dimension at 0 for the sample size.
        loc = jnp.expand_dims(self._loc, 0)

        return centered_samples + loc

    def _log_prob(self, x: Array) -> Array | float:
        x = x - self._loc
        # necessary for correct broadcasting in the quadratic form
        x = jnp.expand_dims(x, axis=-2)
        x_T = jnp.swapaxes(x, -2, -1)

        prob1 = -jnp.squeeze(x @ self._prec @ x_T, axis=(-2, -1))
        prob2 = self.rank * jnp.log(2 * jnp.pi) - self.log_pdet
        return 0.5 * (prob1 - prob2)

    def _event_shape(self):
        return tf.TensorShape((jnp.shape(self._prec)[-1],))

    def _event_shape_tensor(self):
        return jnp.array((jnp.shape(self._prec)[-1],), dtype=jnp.int32)

    def _batch_shape(self):
        return tf.TensorShape(self._broadcast_batch_shape)

    def _batch_shape_tensor(self):
        return jnp.array(self._broadcast_batch_shape, dtype=jnp.int32)
