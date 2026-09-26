"""Gaussian posterior approximations shared by optimizer losses."""

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .types import Position


@dataclass
class LaplaceApproximation:
    """Joint Gaussian approximation of the fitted parameters.

    The lower triangular ``precision_cholesky`` factors the joint precision.
    ``mean`` contains the fitted parameter values. Their flattened order is given
    by ``names`` and ``shapes``: sorted optimized names, followed by sorted latent
    names when present. Construct instances through
    :meth:`~liesel.optim.NegLogProbLoss.approximate_joint_posterior` or
    :meth:`~liesel.optim.LaplaceLoss.approximate_joint_posterior`.

    ``valid=False`` denotes a diagnostic object with no precision factor.
    ``diagnostics['reason']`` explains failure; any evaluated raw curvature and
    gradients remain available there. Invalid objects reject sampling and
    covariance construction. A dense covariance is only allocated on request.
    """

    mean: Position
    """Fitted parameter values at the Gaussian mean."""
    precision_cholesky: jax.Array | None
    """
    Lower Cholesky factor of the joint precision, or ``None`` for an invalid
    approximation.
    """
    names: tuple[str, ...]
    """Parameter names in flattened order: optimized names, then latent names."""
    shapes: tuple[tuple[int, ...], ...]
    """
    Original parameter shapes in the order of
    :attr:`~liesel.optim.LaplaceApproximation.names`.
    """
    valid: bool
    """Whether this approximation supports sampling and covariance construction."""
    diagnostics: dict[str, Any]
    """Curvature diagnostics and, on failure, its reason."""

    def _failed(self, reason, raise_on_failure):
        self.diagnostics["reason"] = reason
        if raise_on_failure:
            raise RuntimeError(reason)
        return self

    def covariance(self) -> jax.Array:
        """Construct the dense joint covariance on request."""
        if not self.valid or self.precision_cholesky is None:
            raise RuntimeError("Cannot use an invalid Laplace approximation.")
        factor = self.precision_cholesky
        return jsp.linalg.cho_solve(
            (factor, True), jnp.eye(factor.shape[0], dtype=factor.dtype)
        )

    def _block_slices(self, position_keys):
        if isinstance(position_keys, str):
            raise ValueError("Pass coordinate names as a sequence, not a string.")  # noqa: TRY004
        slices = {}
        offset = 0
        for name, shape in zip(self.names, self.shapes, strict=True):
            width = math.prod(shape)
            slices[name] = slice(offset, offset + width)
            offset += width
        keys = self.names if position_keys is None else position_keys
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate coordinate names.")
        for name in keys:
            if name not in slices:
                raise ValueError(f"Unknown coordinate {name!r}.")
        return {name: slices[name] for name in keys}

    def marginal_covariance_blocks(
        self, position_keys: Sequence[str] | None = None
    ) -> Position:
        """Return named diagonal blocks of the joint covariance.

        Each block describes a parameter's marginal uncertainty, retaining the
        effect of correlations with all other parameters. Blocks are flattened
        two-dimensional matrices, including (1, 1) for scalars. None selects all
        names; an explicit selection preserves its order. Only selected blocks
        are constructed, without allocating the full covariance. Unknown or
        duplicate names, and a string instead of a sequence, raise ValueError.
        """
        if not self.valid or self.precision_cholesky is None:
            raise RuntimeError("Cannot use an invalid Laplace approximation.")
        factor = self.precision_cholesky
        blocks = Position({})
        for name, section in self._block_slices(position_keys).items():
            selected = jax.nn.one_hot(
                jnp.arange(section.start, section.stop),
                factor.shape[0],
                dtype=factor.dtype,
            ).T
            solved = jsp.linalg.solve_triangular(factor, selected, lower=True)
            blocks[name] = solved.T @ solved
        return blocks

    def marginal_precision_cholesky_blocks(
        self, position_keys: Sequence[str] | None = None
    ) -> Position:
        """Factor the inverse of each selected marginal covariance block.

        Each lower triangular factor L satisfies L @ L.T = inverse(Sigma_ii).
        These are not diagonal blocks of the joint precision Cholesky factor.
        Selection and flattened shapes follow
        :meth:`~liesel.optim.LaplaceApproximation.marginal_covariance_blocks`.
        """
        blocks = Position({})
        for name, covariance in self.marginal_covariance_blocks(position_keys).items():
            precision = jsp.linalg.cho_solve(
                (jnp.linalg.cholesky(covariance), True),
                jnp.eye(covariance.shape[0], dtype=covariance.dtype),
            )
            blocks[name] = jnp.linalg.cholesky(precision)
        return blocks

    def conditional_precision_blocks(
        self, position_keys: Sequence[str] | None = None
    ) -> Position:
        """Return named diagonal blocks of the joint precision.

        Each block is the precision of that parameter conditional on all other
        parameters. Its inverse is a conditional covariance, generally different
        from the marginal covariance. Selection and flattened shapes follow
        :meth:`~liesel.optim.LaplaceApproximation.marginal_covariance_blocks`. The full
        precision is not constructed.
        """
        if not self.valid or self.precision_cholesky is None:
            raise RuntimeError("Cannot use an invalid Laplace approximation.")
        blocks = Position({})
        for name, section in self._block_slices(position_keys).items():
            rows = self.precision_cholesky[section, :]
            blocks[name] = rows @ rows.T
        return blocks

    def sample(
        self, sample_shape: int | Sequence[int] = (), *, seed: jax.Array
    ) -> Position:
        """Draw parameter dictionaries with common leading sample axes.

        The default returns one draw in the original parameter shapes. Pass
        ``draws``, ``(draws,)``, or ``(chains, draws)`` for leading axes accepted
        by :meth:`liesel.model.Model.predict <liesel.model.Model.predict>`. The JAX
        random key ``seed`` is
        required and keyword-only, for example ``sample(1000, seed=key)``.
        """
        if not self.valid or self.precision_cholesky is None:
            raise RuntimeError("Cannot sample an invalid Laplace approximation.")
        factor = self.precision_cholesky
        shape_arg: Any = sample_shape
        try:
            sample_shape = (operator.index(shape_arg),)
        except TypeError:
            sample_shape = tuple(operator.index(dim) for dim in shape_arg)
        size = factor.shape[0]
        noise = jax.random.normal(seed, sample_shape + (size,), dtype=factor.dtype)
        centered = jsp.linalg.solve_triangular(
            factor, noise.reshape((-1, size)).T, lower=True, trans="T"
        ).T.reshape(sample_shape + (size,))
        samples = Position({})
        offset = 0
        for name, shape in zip(self.names, self.shapes, strict=True):
            width = math.prod(shape)
            samples[name] = self.mean[name] + centered[
                ..., offset : offset + width
            ].reshape(sample_shape + shape)
            offset += width
        return samples


def _positive_definite(precision, factor):
    # Roundoff can leave a finite Cholesky factor even for singular curvature.
    return jnp.isfinite(factor).all() & (jnp.linalg.eigvalsh(precision) > 0).all()


def _prepare_approximation(result, at, raise_on_failure, stationarity_tol):
    """Validate common options and select a finite fitted position."""
    if at not in ("min_monitor", "final"):
        raise ValueError("at must be 'min_monitor' or 'final'.")
    if not isinstance(raise_on_failure, bool):
        raise ValueError("raise_on_failure must be True or False.")  # noqa: TRY004
    if (
        isinstance(stationarity_tol, bool)
        or not isinstance(stationarity_tol, Real)
        or not math.isfinite(stationarity_tol)
        or stationarity_tol <= 0
    ):
        raise ValueError("stationarity_tol must be a positive finite real number.")
    approximation = LaplaceApproximation(
        mean=Position({}),
        precision_cholesky=None,
        names=(),
        shapes=(),
        valid=False,
        diagnostics={"reason": None},
    )
    try:
        position = (
            result.position_min_monitor
            if at == "min_monitor"
            else result.position_final
        )
    except RuntimeError as error:
        return approximation._failed(
            f"The selected position is unavailable: {error}", raise_on_failure
        )
    approximation.mean = Position(dict(position))
    approximation.names = tuple(sorted(position))
    approximation.shapes = tuple(
        jnp.shape(position[name]) for name in approximation.names
    )
    return approximation
