"""
The degenerate, i.e. rank-deficient, multivariate normal distribution.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates.jax import tf2jax as tf

Array = Any


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

    Provides the alternative constructor :meth:`.from_penalty` and sampling
    via :meth:`.sample`.

    This is a simplified code-based illustration of how the log-probability for an array
    ``x`` is evaluated::

        xc = x - loc
        log_prob = -0.5 * (rank * np.log(2*np.pi) - log_pdet) -0.5 * (xc.T @ prec @ xc)


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
    tol
        Numerical tolerance for determining which eigenvalues of the distribution's \
        precision matrices should be treated as zeros. Used in :attr:`.rank` and \
        :attr:`.log_pdet`, if they are computed by the class. Also used in \
        :meth:`.sample`.

    Notes
    -----
    * If they are not provided as arguments, ``rank`` and ``log_pdet`` are computed
      based on the eigenvalues of the precision matrix ``prec``. This is an expensive
      operation and can be avoided by specifying the corresponding arguments.
    * When you draw samples from the distribution via :meth:`.sample`, it is always
      necessary to compute the eigendecomposition of the distribution's precision
      matrices once and cache it, because sampling requires both the eigenvalues and
      eigenvectors.

    **Details on sampling**

    To draw samples from a denegerate multivariate normal distribution, we 1) draw
    standard normal samples with mean zero and variance one, 2) transform these samples
    to have the desired covariance structure, and 3) add the desired mean.

    The main problem is to find out how we have to transform the standard normal samples
    in step 2. Say that we have a singular :math:`(m \\times m)`  precision matrix
    :math:`P`. We can view it as
    the generalized inverse of a variance-covariance matrix :math:`\\Sigma`. We can
    obtain :math:`\\Sigma` by finding the eigenvalue decomposition of the precision
    matrix, i.e.

    .. math::
        P = QA^+Q^T,

    where :math:`Q` is the orthogonal matrix of eigenvectors and :math:`A^+` is the
    diagonal matrix of eigenvalues. Note that, if the precision matrix is singular, then
    :math:`\\text{diag}(A^+)` contains zeroes. Now we take the inverse of the non-zero
    entries of :math:`\\text{diag}(A^+)`, while the zero entries remain at zero,
    resulting in a matrix :math:`A`. We can now write

    .. math::
        \\Sigma = Q A Q^T.

    Now we can go through the three steps in detail. We first draw a vector of the
    desired length :math:`z \\sim N(0, I)` from a standard normal distribution.
    :math:`I` is the identity matrix of appropriate dimension. Next, we transform the
    sample by applying :math:`x = Q A^{1/2}z`, such that :math:`\\text{Cov}(x) =
    \\Sigma`:

    .. math::
        \\text{Cov}(x)  & = Q A^{1/2} I (A^{1/2})^T Q^T \\

                        & = Q A Q^T \\

                        & = \\Sigma.

    In the last step, we add the desired mean :math:`\\mu` to :math:`x`.
    Note that the distribution is not a proper distribution on :math:`\\mathbb{R}^m`,
    where :math:`m` refers to the number of columns and rows of :math:`P`.
    Any vector in the null space of :math:`P` can be added to any
    :math:`x \\in \\mathbb{R}^m` without changing the density. The samples generated
    using the procedure described above are orthogonal to the null space of :math:`P`.
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
        tol: float = 1e-6,
    ):
        parameters = dict(locals())

        self._tol = tol
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

        self._prec = jnp.expand_dims(prec, tuple(range(nbatch - len(prec_batches))))
        self._loc = jnp.expand_dims(loc, tuple(range(nbatch - len(loc_batches))))

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

        if rank is None or log_pdet is None:
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

    @classmethod
    def from_penalty_smooth(
        cls,
        loc: Array,
        smooth: Array,
        pen: Array,
        rank: Array | int | None = None,
        log_pdet: Array | float | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "MultivariateNormalDegenerate",
    ) -> MultivariateNormalDegenerate:
        """
        Alternative constructor based on a penalty matrix and a smoothing
        parameter.

        Sometimes, the precision matrix of a degenerate multivariate normal
        distribution is decomposed into a penalty matrix ``pen`` and an inverse
        smoothing parameter ``var``. Using this constructor, a degenerate multivariate
        normal distribution can be initialized from such a decomposition.


        Parameters
        ----------
        loc
            The location (= mean) vector.
        smooth
            The smoothing (= inverse variance) parameter.
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

        prec = pen * jnp.expand_dims(smooth, axis=(-2, -1))

        if rank is None or log_pdet is None:
            evals = jax.numpy.linalg.eigvalsh(pen)
            rank = _rank(evals) if rank is None else rank
            log_pdet = _log_pdet(evals, rank=rank) if log_pdet is None else log_pdet

        log_pdet_prec = log_pdet + rank * jnp.log(smooth)

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

    @cached_property
    def eig(self) -> tuple[Array, Array]:
        """Eigenvalues and eigenvectors of the distribution's precision matrices."""
        return jnpla.eigh(self._prec)

    @cached_property
    def _sqrt_pcov(self) -> Array:
        """
        Square roots of the distribution's pseudo-covariance matrices.

        Let ``prec = Q @ A @ Q.T`` be the eigendecomposition of the precision matrix.
        In essence, this property returns ``Q @ jnp.sqrt(1/A)``.
        """
        eigenvalues, evecs = self.eig

        sqrt_eval = jnp.sqrt(1 / eigenvalues)
        sqrt_eval = jnp.where(eigenvalues < self._tol, 0.0, sqrt_eval)

        event_shape = sqrt_eval.shape[-1]
        shape = sqrt_eval.shape + (event_shape,)

        r = tuple(range(event_shape))
        diags = jnp.zeros(shape).at[..., r, r].set(sqrt_eval)
        return evecs @ diags

    @cached_property
    def rank(self) -> Array | float:
        """Ranks of the distribution's precision matrices."""
        if self._rank is not None:
            return self._rank
        evals, _ = self.eig
        return _rank(evals, tol=self._tol)

    @cached_property
    def log_pdet(self) -> Array | float:
        """Log-pseudo-determinants of the distribution's precision matrices."""
        if self._log_pdet is not None:
            return self._log_pdet
        evals, _ = self.eig
        return _log_pdet(evals, self.rank, tol=self._tol)

    @property
    def prec(self) -> Array:
        """Precision matrices."""
        return self._prec

    @property
    def loc(self) -> Array:
        """Locations."""
        return self._loc

    def _sample_n(self, n, seed=None) -> Array:
        shape = [n] + self.batch_shape + self.event_shape

        # The added dimension at the end here makes sure that matrix multiplication
        # with the "sqrt pcov" matrices works out correctly.
        z = jax.random.normal(key=seed, shape=shape + [1])

        # Add a dimension at 0 for the sample size.
        sqrt_pcov = jnp.expand_dims(self._sqrt_pcov, 0)
        centered_samples = jnp.reshape(sqrt_pcov @ z, shape)

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
