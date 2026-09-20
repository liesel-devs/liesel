from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import InitVar, dataclass, field
from typing import Literal, cast, overload

import jax
import jax.numpy as jnp
import numpy as np

from ..model import Model
from . import _alias
from ._log_lik import scaled_common_log_lik as _scaled_common_log_lik
from ._log_lik import scaled_liesel_log_lik as _scaled_liesel_log_lik
from ._model_utils import position_key_groups_from_model
from .split import (
    PositionSplit,
    PositionSplitManager,
    _count_likelihood_contributions,
    _has_custom_model_log_lik,
    _observed_dist_infos,
)
from .types import Array, ModelInterface, ModelState, Position

_MISSING = object()


def _sampling_categories(labels: Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Object conversion preserves label identity and exposes mixed string/numeric
    # sequences before NumPy can silently coerce them to strings or floats.
    values = np.asarray(labels, dtype=object)
    if values.ndim != 1 or not values.size:
        raise ValueError("labels must be a nonempty one-dimensional sequence.")
    strings = all(isinstance(value, str) for value in values)
    numeric = all(
        isinstance(value, (int, np.integer, bool, np.bool_))
        or isinstance(value, (float, np.floating))
        and np.isfinite(value)
        for value in values
    )
    if not (strings or numeric):
        raise ValueError(
            "labels must contain only strings or finite numeric categories, "
            "without missing values or mixed string/numeric labels."
        )
    return np.unique(values, return_inverse=True, return_counts=True)


def _resolve_batch_size(
    batch_size: int | None | object,
    batch_axis_size: int | None | object,
) -> int | None:
    if batch_size is _MISSING and batch_axis_size is _MISSING:
        raise TypeError("missing required argument: 'batch_size'")

    if batch_size is not _MISSING and batch_axis_size is not _MISSING:
        raise TypeError("Pass either batch_size or batch_axis_size, not both.")

    return cast(int | None, batch_axis_size if batch_size is _MISSING else batch_size)


def _sampling_weights_for_groups(
    sampling_weights: Array | Mapping[str, Array] | None,
    groups: Sequence[Sequence[str]],
) -> list[Array | None]:
    """Route one weight vector to each group, without broadcasting across groups."""
    if sampling_weights is None:
        return [None] * len(groups)
    if not isinstance(sampling_weights, Mapping):
        if len(groups) != 1:
            raise ValueError(
                "Multiple batch groups require sampling_weights as a mapping "
                "from one position key per group to its weight vector."
            )
        return [sampling_weights]

    unknown = sampling_weights.keys() - {key for group in groups for key in group}
    if unknown:
        raise ValueError(f"Unknown sampling_weights position keys: {list(unknown)}.")
    weights = []
    for group in groups:
        keys = [key for key in group if key in sampling_weights]
        if len(keys) > 1:
            raise ValueError(
                "Provide sampling_weights for only one position key per group; "
                f"got {keys} in group {list(group)}."
            )
        if keys and sampling_weights[keys[0]] is None:
            raise ValueError("sampling_weights entries must be weight vectors.")
        weights.append(sampling_weights[keys[0]] if keys else None)
    return weights


def _normalize_positive_size(size: float | None, name: str) -> float | None:
    if size is None:
        return None

    size_float = float(size)
    if not math.isfinite(size_float) or size_float <= 0.0:
        raise ValueError(f"{name} must be finite and positive, but got {size!r}.")

    return size_float


def _infer_sample_size_from_state(
    model: Model,
    model_state: ModelState,
    position_keys: Sequence[str],
    axis_size: int,
) -> float:
    if _has_custom_model_log_lik(model):
        raise ValueError(
            "Cannot infer sample sizes for a model with a custom log_lik_node. "
            "Provide sample_size and batch_sample_size manually."
        )

    infos = _observed_dist_infos(model, position_keys)
    sizes: dict[str, int] = {}

    for var_name, node_name, per_obs in infos:
        if not per_obs:
            raise ValueError(
                "Cannot infer sample sizes because "
                f"{var_name!r} has Var.dist_node.per_obs=False. Provide "
                "sample_size and batch_sample_size manually."
            )

        sizes[var_name] = _count_likelihood_contributions(model_state[node_name].value)

    if not sizes:
        return float(axis_size)

    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(
            "Cannot infer a scalar sample size because observed variables in one "
            f"Batches object imply incompatible pointwise sample sizes: {sizes}. "
            "Use BatchManager or provide sample sizes manually."
        )

    return float(unique_sizes.pop())


def _axis_size_for_empty_position_keys(
    model: Model,
    batch_axes: dict[str, int] | None,
    default_batch_axis: int,
) -> int:
    _, groups = position_key_groups_from_model(
        model,
        list(model.observed),
        batch_axes,
        default_batch_axis,
    )

    if not groups:
        raise ValueError(
            "Cannot infer axis_size for empty position_keys from a model without "
            "observed variables. Provide axis_size manually."
        )

    if len(groups) != 1:
        raise ValueError(
            "Cannot infer a single axis_size for empty position_keys because "
            "observed variables have different axis sizes. Provide axis_size manually."
        )

    return int(groups[0][0])


@dataclass(init=False)
class Batches:
    """
    Defines mini-batches for observed entries in an optimizer position.

    ``Batches`` owns the observation indices for the current epoch and reshapes complete
    parts of shuffled passes into batches. The observed position entries named in
    ``position_keys`` are sliced with these indices. By default, every entry is
    sliced along axis ``0``; use ``default_batch_axis`` or ``batch_axes`` for
    arrays where observations live on another axis.

    Parameters
    ----------
    position_keys
        Names of the position entries that should be batched. An empty sequence is
        allowed only with ``batch_size=None``; in that case the object is a
        full-data adapter that does not replace any observed entries.
    axis_size
        Number of observations along each batched axis.
    batch_size
        Number of observations per batch. If ``None``, batching is disabled by using
        a single batch with all ``axis_size`` observations.
    shuffle
        Whether :meth:`permute_indices` should return a random permutation of the
        indices. If ``False``, :meth:`permute_indices` returns the indices unchanged.
    batch_axes
        Optional mapping from position key to batching axis. Keys missing from this
        mapping use ``default_batch_axis``.
    default_batch_axis
        Batching axis for all position keys not listed in ``batch_axes``.
    sample_with_replacement
        Whether every assembled batch draws observations independently with
        replacement. This is useful for explicit replacement sampling and is enabled
        automatically for an oversized child when a common batch size exceeds its
        observation count.
    sample_size
        Optional effective likelihood sample size represented by the full data. If
        omitted, likelihood scaling falls back to ``axis_size``. Use
        :meth:`from_model` to infer this value from pointwise observed
        log-probability arrays.
    batch_sample_size
        Optional effective likelihood sample size represented by one batch. If
        ``sample_size`` is supplied and ``batch_sample_size`` is omitted, it is
        derived as ``sample_size * batch_size / axis_size``.
    batch_axis_size
        Backwards-compatible keyword-only alias for ``batch_size``. Pass only one
        of ``batch_size`` and ``batch_axis_size``.
    sampling_weights
        Optional finite, strictly positive relative weights of length ``axis_size``,
        aligned with the training data in its current order. Requires
        ``sample_with_replacement=True``. Normalized probabilities remain fixed
        during a run; saved probabilities and alias tables take precedence on
        checkpoint recovery. Probabilities and corrections use at least float32.
        Values whose probabilities or correction factors cannot be represented in
        the working dtype are rejected. No probability floor or clipping is applied.
    likelihood_axes
        Optional mapping from observed-variable name to its pointwise log-likelihood
        axis. Weighted correction normally infers this axis using trailing event
        reduction and leading broadcasting. Declare it explicitly for custom
        reductions or transpositions. The declared axis must enumerate the sampled
        observations; matching dimension sizes alone cannot establish this.

    Attributes
    ----------
    indices
        Current ordering of the observations. Initialized as
        ``jnp.arange(axis_size)`` and used by :attr:`batch_indices`. Assign the
        result of :meth:`permute_indices` to this attribute to use a fresh order.

    Notes
    -----
    If ``axis_size`` is not divisible by ``batch_size``, only full batches
    are used and the final incomplete tail is dropped independently from each
    shuffled pass. When more batches are requested, another independent shuffled pass
    supplies fresh rows; prior batches are never copied wholesale.

    With replacement sampling, an epoch retains the same number of batches but
    does not guarantee coverage. Weighted groups apply an extra per-index factor
    ``1 / (axis_size * p_i)`` through :meth:`scaled_log_lik`, preserving the
    original objective in expectation. :class:`.NegLogProbLoss` applies this
    automatically. Other losses must use that
    method or apply :meth:`correction_factors` before summing likelihood values.

    A no-key full-data adapter can be useful when an optimizer workflow expects a
    :class:`Batches` object but the model should always be evaluated on the full
    observed data. Use ``position_keys=[]`` and ``batch_size=None`` for this case.

    Examples
    --------
    Create two batches of size four from ten observations:

    >>> from liesel.optim import Batches
    >>> batches = Batches(["y"], axis_size=10, batch_size=4, shuffle=False)
    >>> batches.batch_indices.tolist()
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> batches.n_full_batches
    2

    With ``batch_size=None``, the object represents one full-data batch:

    >>> full_data = Batches(["y"], axis_size=5, batch_size=None)
    >>> full_data.batch_size
    5
    >>> full_data.batch_indices.tolist()
    [[0, 1, 2, 3, 4]]

    An empty-key full-data adapter leaves observed entries untouched:

    >>> no_key = Batches([], axis_size=5, batch_size=None)
    >>> no_key.position_keys, no_key.is_full_data
    ([], True)

    ``batch_axes`` can batch different entries along different batch_axes:

    >>> import jax.numpy as jnp
    >>> batches = Batches(
    ...     ["x", "y"],
    ...     axis_size=5,
    ...     batch_size=2,
    ...     batch_axes={"x": 1},
    ...     shuffle=False,
    ... )
    >>> position = {
    ...     "x": jnp.arange(15).reshape(3, 5),
    ...     "y": jnp.arange(20).reshape(5, 4),
    ... }
    >>> batched = batches.get_batched_position(position, batch_index=0)
    >>> batched["x"].shape, batched["y"].shape
    ((3, 2), (2, 4))
    """

    position_keys: Sequence[str]
    axis_size: int
    batch_size: int | None
    shuffle: bool = True
    batch_axes: dict[str, int] | None = None
    default_batch_axis: int = 0
    sample_with_replacement: bool = False
    sample_size: int | float | None = None
    batch_sample_size: int | float | None = None
    likelihood_axes: dict[str, int]
    _sampling_probabilities: jax.Array | None
    _alias_table: _alias.AliasTable | None

    def __init__(
        self,
        position_keys: Sequence[str],
        axis_size: int,
        batch_size: int | None | object = _MISSING,
        shuffle: bool = True,
        batch_axes: dict[str, int] | None = None,
        default_batch_axis: int = 0,
        sample_with_replacement: bool = False,
        sample_size: float | None = None,
        batch_sample_size: float | None = None,
        *,
        batch_axis_size: int | None | object = _MISSING,
        sampling_weights: Array | None = None,
        likelihood_axes: dict[str, int] | None = None,
    ) -> None:
        self.position_keys = position_keys
        self.axis_size = axis_size
        self.batch_size = _resolve_batch_size(batch_size, batch_axis_size)
        if sample_with_replacement and self.batch_size is None:
            raise ValueError(
                "sample_with_replacement=True requires an explicit batch_size."
            )
        if sample_with_replacement and not shuffle:
            raise ValueError("sample_with_replacement=True requires shuffle=True.")
        self.shuffle = shuffle
        self.batch_axes = batch_axes
        self.default_batch_axis = default_batch_axis
        self.sample_with_replacement = sample_with_replacement
        self.sample_size = sample_size
        self.batch_sample_size = batch_sample_size
        self.likelihood_axes = dict(likelihood_axes or {})
        if any(not isinstance(axis, int) for axis in self.likelihood_axes.values()):
            raise ValueError("likelihood_axes values must be integers.")
        self.__post_init__()
        self._set_sampling_weights(sampling_weights)

    def _set_sampling_weights(self, sampling_weights: Array | None) -> None:
        self._sampling_probabilities = None
        self._alias_table = None
        if sampling_weights is not None:
            if not self.sample_with_replacement:
                raise ValueError(
                    "sampling_weights requires sample_with_replacement=True."
                )
            weights = np.asarray(sampling_weights)
            if jnp.issubdtype(weights.dtype, jnp.integer):
                weights = weights.astype(float)
            if (
                not jnp.issubdtype(weights.dtype, jnp.floating)
                or weights.shape != (self.axis_size,)
                or not np.all(np.isfinite(weights) & (weights > 0))
            ):
                raise ValueError(
                    "sampling_weights must be a finite, positive vector "
                    "of length axis_size."
                )
            # Normalize on the host: XLA can replace division by a reciprocal,
            # which underflows for large, otherwise valid relative weights.
            relative = np.asarray(weights, dtype=np.float64)
            relative = relative / relative.max()
            dtype = jax.dtypes.canonicalize_dtype(
                jnp.promote_types(weights.dtype, jnp.float32)
            )
            normalized = relative / relative.sum()
            probabilities = jnp.asarray(normalized, dtype=dtype)
            corrections = 1.0 / (self.axis_size * probabilities)
            if not bool(
                jnp.all(
                    jnp.isfinite(probabilities)
                    & (probabilities > 0)
                    & jnp.isfinite(corrections)
                    & (corrections > 0)
                )
            ):
                raise ValueError(
                    "sampling_weights probabilities or corrections "
                    "are not representable."
                )
            self._sampling_probabilities = probabilities
            self._alias_table = _alias.build_table(normalized)

    @property
    def sampling_probabilities(self) -> jax.Array | None:
        """Normalized sampling probabilities, or None for uniform sampling."""
        return self._sampling_probabilities

    @staticmethod
    def weights_balanced(labels: Array, *, strength: float = 1.0) -> np.ndarray:
        """Construct sampling weights from category frequencies.

        Each observation in a category of size ``n`` receives weight
        ``n ** (-strength)``. At strength one, every observed category has equal
        total sampling probability; zero gives uniform observation sampling.

        Parameters
        ----------
        labels
            Nonempty one-dimensional labels in training-row order. Accepts strings,
            booleans, integers, or finite floating labels interpreted as exact
            categories. Missing and mixed string/numeric labels are rejected.
        strength
            Finite scalar between zero and one; defaults to full balancing.

        Returns
        -------
        numpy.ndarray
            Float64 relative weights in input order. This helper runs on the host,
            outside JIT. Pass its output to ``Batches(..., sampling_weights=...)``
            with ``sample_with_replacement=True``. The existing importance
            correction preserves the objective; balancing sampling does not
            reweight the likelihood objective.

        Examples
        --------
        >>> Batches.weights_balanced(["a", "b", "a"]).tolist()
        [0.5, 1.0, 0.5]
        >>> Batches.weights_balanced(["a", "b", "a"], strength=0).tolist()
        [1.0, 1.0, 1.0]
        """
        strength_array = np.asarray(strength)
        if (
            strength_array.ndim != 0
            or strength_array.dtype.kind not in "biuf"
            or not np.isfinite(strength_array)
            or not 0 <= strength_array <= 1
        ):
            raise ValueError("strength must be a finite scalar between 0 and 1.")
        _, inverse, counts = _sampling_categories(labels)
        return counts[inverse].astype(np.float64) ** (-float(strength_array))

    @staticmethod
    def weights_for_shares(
        labels: Array,
        shares: Mapping[str | int | float, float],
        *,
        check_sum: bool = True,
    ) -> np.ndarray:
        """Divide each category's target sampling share among its observations.

        Parameters
        ----------
        labels
            One-dimensional category labels in training-row order, with the same
            requirements as :meth:`weights_balanced`.
        shares
            Mapping from every observed category to its finite, strictly positive
            sampling share. Missing or extra categories are rejected. A category
            with share ``s`` and count ``n`` gives each observation weight ``s / n``.
        check_sum
            Require shares to sum to one within absolute tolerance ``1e-6`` and
            zero relative tolerance. Set to ``False`` to supply relative category
            masses. Only this sum check is skipped; all other validation remains.

        Returns
        -------
        numpy.ndarray
            Float64 weights in input order, computed on the host outside JIT.
            Weights are not normalized here; :class:`Batches` normalizes them into
            sampling probabilities. Unrepresentable zero weights are rejected.
            Shares describe expected frequencies, not fixed quotas per minibatch.

        Examples
        --------
        >>> Batches.weights_for_shares(["a", "b", "a"], {"a": 0.6, "b": 0.4}).tolist()
        [0.3, 0.4, 0.3]
        >>> Batches.weights_for_shares(
        ...     ["a", "b", "a"], {"a": 60, "b": 40}, check_sum=False
        ... ).tolist()
        [30.0, 40.0, 30.0]
        """
        categories, inverse, counts = _sampling_categories(labels)
        if not isinstance(shares, Mapping) or set(shares) != set(categories):
            raise ValueError("shares must contain exactly the observed categories.")
        masses = np.asarray([shares[label] for label in categories])
        if masses.shape != counts.shape or masses.dtype.kind not in "biuf":
            raise ValueError(
                "shares must be finite, strictly positive numeric scalars."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            masses = masses.astype(np.float64)
        if not np.all(np.isfinite(masses) & (masses > 0)):
            raise ValueError("shares must be finite and strictly positive.")
        if check_sum:
            with np.errstate(over="ignore"):
                total = masses.sum()
            if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "shares must sum to one within absolute tolerance 1e-6."
                )
        with np.errstate(under="ignore"):
            weights = masses / counts
        if not np.all(np.isfinite(weights) & (weights > 0)):
            raise ValueError("shares produce weights that are not representable.")
        return weights[inverse]

    @staticmethod
    def weights_binned(
        values: Array,
        *,
        bins: int | Sequence[float] | np.ndarray,
        strength: float = 0.5,
    ) -> np.ndarray:
        """Construct sampling weights by balancing one-dimensional bin counts.

        Parameters
        ----------
        values
            Nonempty, finite one-dimensional numeric values in training-row order.
        bins
            Required positive integer count or strictly increasing numeric edges.
            Integer counts give equal-width bins across the observed range.
            Explicit edges must cover all observations; only the outer endpoints
            may be infinite. Intervals include their left endpoint and exclude
            their right, except that the final right endpoint is included.
        strength
            Finite scalar between zero and one, as in :meth:`weights_balanced`.
            Defaults to partial balancing with strength ``0.5``.

        Returns
        -------
        numpy.ndarray
            Float64 weights in input order. Each observation receives its bin's
            count raised to ``-strength``. Empty bins get no sampling mass;
            constant data get uniform weights. Unequal-width bins are balanced
            by counts without a width adjustment: this is not density estimation.
            Values are processed as float64 on the host, outside JIT. Degenerate
            generated edges are rejected; supply explicit edges in that case.

        Examples
        --------
        >>> Batches.weights_binned([0, 0, 0, 0, 10], bins=2).tolist()
        [0.5, 0.5, 0.5, 0.5, 1.0]
        >>> Batches.weights_binned([0, 1, 4], bins=[0, 1, 4], strength=1).tolist()
        [1.0, 0.5, 0.5]
        """
        values = np.asarray(values)
        if (
            values.ndim != 1
            or not values.size
            or values.dtype.kind not in "biuf"
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("values must be a nonempty, finite numeric vector.")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be representable as finite float64 values.")
        if isinstance(bins, (int, np.integer)) and not isinstance(
            bins, (bool, np.bool_)
        ):
            if bins < 1:
                raise ValueError("bins must be a positive integer.")
            if np.all(values == values[0]):
                return Batches.weights_balanced(
                    np.zeros(values.size), strength=strength
                )
            with np.errstate(over="ignore", invalid="ignore"):
                edges = np.histogram_bin_edges(values, bins=bins)
            if not np.all(np.isfinite(edges)):
                raise ValueError("bins cannot be represented; supply explicit edges.")
        else:
            edges = np.asarray(bins)
            if edges.ndim != 1 or edges.dtype.kind not in "iuf":
                raise ValueError("bins must be a positive integer or numeric edges.")
        if (
            edges.size < 2
            or not np.all(edges[1:] > edges[:-1])
            or not np.all(np.isfinite(edges[1:-1]))
        ):
            raise ValueError(
                "bins must have strictly increasing edges and finite interior edges."
            )
        if values.min() < edges[0] or values.max() > edges[-1]:
            raise ValueError("bins must cover every observation in values.")
        categories = np.searchsorted(edges, values, side="right") - 1
        # Only the final right edge is inclusive; outside values were rejected above.
        categories = np.minimum(categories, len(edges) - 2)
        return Batches.weights_balanced(categories, strength=strength)

    def __post_init__(self):
        if self.axis_size < 1:
            raise ValueError(f"{self.axis_size=} is < 1, which is not allowed.")

        if len(self.position_keys) == 0 and self.batch_size is not None:
            raise ValueError("position_keys may be empty only when batch_size=None.")

        if self.batch_size is None:
            self.batch_size = self.axis_size

        if self.batch_size < 1:
            raise ValueError(f"{self.batch_size=} is < 1, which is not allowed.")

        if self.axis_size < self.batch_size and not self.sample_with_replacement:
            raise ValueError(
                f"{self.axis_size=} is < {self.batch_size=}. This is only "
                "allowed with sample_with_replacement=True."
            )

        if len(set(self.position_keys)) != len(self.position_keys):
            raise ValueError(
                f"Duplicate position_keys are not allowed: {list(self.position_keys)}"
            )

        if self.batch_axes is None:
            self.batch_axes = {}

        self.sample_size = _normalize_positive_size(self.sample_size, "sample_size")
        self.batch_sample_size = _normalize_positive_size(
            self.batch_sample_size, "batch_sample_size"
        )

        if self.sample_size is not None and self.batch_sample_size is None:
            assert self.batch_size is not None
            self.batch_sample_size = self.sample_size * self.batch_size / self.axis_size

        self.indices = self._default_indices()

    @property
    def _uses_replacement(self) -> bool:
        return self.sample_with_replacement

    def _default_indices(self) -> jax.Array:
        return self._assemble_indices(None, None)

    @classmethod
    def from_split(
        cls,
        split: PositionSplit | PositionSplitManager,
        batch_size: int | None,
        shuffle: bool = True,
        batch_axes: dict[str, int] | None = None,
        default_batch_axis: int = 0,
        epoch_size: Literal["strict", "min", "max"] | int = "max",
    ) -> Batches | BatchManager:
        """Build training batches directly from a completed position split."""
        if isinstance(split, PositionSplitManager):
            children = [
                cls(
                    position_keys=child.split_position_keys,
                    axis_size=child.train_axis_size,
                    batch_size=batch_size,
                    shuffle=False if batch_size is None else shuffle,
                    batch_axes=batch_axes,
                    default_batch_axis=default_batch_axis,
                    sample_size=child.train_sample_size,
                    sample_with_replacement=(
                        batch_size is not None and batch_size > child.train_axis_size
                    ),
                )
                for child in split.splits
            ]
            return BatchManager(children, epoch_size=epoch_size)

        return cls(
            position_keys=split.split_position_keys,
            axis_size=split.train_axis_size,
            batch_size=batch_size,
            shuffle=False if batch_size is None else shuffle,
            batch_axes=batch_axes,
            default_batch_axis=default_batch_axis,
            sample_size=split.train_sample_size,
        )

    @classmethod
    @overload
    def from_model(
        cls,
        model: Model,
        batch_size: int | None | object = _MISSING,
        position_keys: Sequence[str] | Sequence[Sequence[str]] | None = None,
        axis_size: int | None = None,
        shuffle: bool = True,
        batch_axes: dict[str, int] | None = None,
        default_batch_axis: int = 0,
        multi_size: Literal["error"] = "error",
        epoch_size: Literal["strict", "min", "max"] | int = "max",
        sample_size: float | None = None,
        batch_sample_size: float | None = None,
        infer_sample_size: bool = True,
        sample_with_replacement: bool = False,
        *,
        batch_axis_size: int | None | object = _MISSING,
        sampling_weights: Array | Mapping[str, Array] | None = None,
    ) -> Batches: ...

    @classmethod
    @overload
    def from_model(
        cls,
        model: Model,
        batch_size: int | None | object = _MISSING,
        position_keys: Sequence[str] | Sequence[Sequence[str]] | None = None,
        axis_size: int | None = None,
        shuffle: bool = True,
        batch_axes: dict[str, int] | None = None,
        default_batch_axis: int = 0,
        multi_size: Literal["manager"] = "manager",
        epoch_size: Literal["strict", "min", "max"] | int = "max",
        sample_size: float | None = None,
        batch_sample_size: float | None = None,
        infer_sample_size: bool = True,
        sample_with_replacement: bool = False,
        *,
        batch_axis_size: int | None | object = _MISSING,
        sampling_weights: Array | Mapping[str, Array] | None = None,
    ) -> Batches | BatchManager: ...

    @classmethod
    def from_model(
        cls,
        model: Model,
        batch_size: int | None | object = _MISSING,
        position_keys: Sequence[str] | Sequence[Sequence[str]] | None = None,
        axis_size: int | None = None,
        shuffle: bool = True,
        batch_axes: dict[str, int] | None = None,
        default_batch_axis: int = 0,
        multi_size: Literal["error", "manager"] = "error",
        epoch_size: Literal["strict", "min", "max"] | int = "max",
        sample_size: float | None = None,
        batch_sample_size: float | None = None,
        infer_sample_size: bool = True,
        sample_with_replacement: bool = False,
        *,
        batch_axis_size: int | None | object = _MISSING,
        sampling_weights: Array | Mapping[str, Array] | None = None,
    ) -> Batches | BatchManager:
        """
        Builds a :class:`Batches` object from a Liesel model.

        Parameters
        ----------
        model
            Model containing the observed variables to batch.
        batch_size
            Number of observations per batch. If ``None``, batching is disabled and
            the returned object uses one full-data batch.
        position_keys
            Names of the observed position entries to batch. If ``None``, all observed
            variables in ``model`` are used. Flat keys are grouped by axis length;
            nested keys specify exact groups, including equal-sized groups. Keys
            in a group must have matching lengths along their batching axes.
            Pass an empty sequence only with
            ``batch_size=None`` to build a full-data adapter that does not replace
            observed entries.
        axis_size
            Number of observations. If ``None``, the number is guessed from the model's
            observed variables along ``default_batch_axis``.
        shuffle
            Whether epoch-wise calls to :meth:`permute_indices` should shuffle the
            indices. This is forced to ``False`` when ``batch_size`` is ``None``.
        batch_axes
            Optional mapping from position key to batching axis.
        default_batch_axis
            Axis used for guessing ``axis_size`` and for position keys missing
            from ``batch_axes``.
        multi_size
            How to handle multiple inferred or explicit observation groups.
            The default ``"error"`` keeps :class:`Batches` scalar and raises a
            helpful error. Use ``"manager"`` to return a :class:`BatchManager` when
            multiple groups are detected, even with equal axis sizes. One group
            still returns a scalar :class:`Batches` object.
        epoch_size
            Batch manager epoch size used only when ``multi_size="manager"``.
        sample_size
            Optional effective likelihood sample size represented by the full
            observed data. If omitted and ``infer_sample_size=True``, it is inferred
            from pointwise observed log-probability values when possible. Inference
            counts log-probability scalars, not observed value elements; for
            multivariate observation distributions, one observed event may have
            several value dimensions but one pointwise log-probability scalar.
        batch_sample_size
            Optional effective likelihood sample size represented by one batch. If
            omitted and ``infer_sample_size=True``, it is inferred from the first
            batch when possible.
        infer_sample_size
            Whether to infer missing effective sample sizes from observed
            log-probability values.
        sample_with_replacement
            Whether every assembled batch draws observations independently with
            replacement. With ``multi_size="manager"``, this applies to every
            inferred child; automatic construction also enables it for an oversized
            child when the common batch size exceeds its observation count.
        batch_axis_size
            Backwards-compatible keyword-only alias for ``batch_size``. Pass only
            one of ``batch_size`` and ``batch_axis_size``.
        sampling_weights
            Positive relative sampling weights in current data order. Supply a vector
            for one group, or a mapping from one selected position key per group to
            its vector. The vector applies to all aligned entries in that group.
            Unknown keys and multiple entries for one group are rejected; omitted
            groups use uniform sampling. Weighted groups require replacement
            sampling. See :class:`Batches` for validation and likelihood correction.

        Returns
        -------
        Batches or BatchManager
            Batch configuration for the model's observed data. A
            :class:`BatchManager` is returned only when ``multi_size="manager"`` and
            multiple groups are detected, even with equal axis sizes. One group
            still returns a scalar :class:`Batches` object.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.optim import Batches

        >>> y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        >>> model = lsl.Model([y])
        >>> batches = Batches.from_model(model, batch_size=2, position_keys=["y"])
        >>> batches.axis_size, batches.batch_size, batches.position_keys
        (6, 2, ['y'])

        Passing ``batch_size=None`` disables shuffling and creates one
        full-data batch:

        >>> full_data = Batches.from_model(model, batch_size=None, position_keys=["y"])
        >>> full_data.shuffle, full_data.batch_indices.tolist()
        (False, [[0, 1, 2, 3, 4, 5]])

        An empty ``position_keys`` sequence creates a no-key full-data adapter:

        >>> no_key = Batches.from_model(model, batch_size=None, position_keys=[])
        >>> no_key.position_keys, no_key.is_full_data
        ([], True)

        Multi-size observed data must opt into the manager API:

        >>> x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        >>> z = lsl.Var.new_obs(jnp.arange(5.0), name="z")
        >>> model = lsl.Model([x, z])
        >>> manager = Batches.from_model(
        ...     model,
        ...     batch_size=2,
        ...     position_keys=["x", "z"],
        ...     multi_size="manager",
        ... )
        >>> type(manager).__name__, manager.axis_size, manager.n_full_batches
        ('BatchManager', (8, 5), 4)
        """
        batch_size = _resolve_batch_size(batch_size, batch_axis_size)
        if multi_size not in ("error", "manager"):
            raise ValueError("multi_size must be 'error' or 'manager'.")

        pos_keys = (
            list(position_keys) if position_keys is not None else list(model.observed)
        )
        if not pos_keys and batch_size is not None:
            raise ValueError("position_keys may be empty only when batch_size=None.")

        pos_keys, groups = position_key_groups_from_model(
            model, pos_keys, batch_axes, default_batch_axis
        )

        if len(groups) > 1:
            if multi_size == "manager":
                if (
                    axis_size is not None
                    or sample_size is not None
                    or batch_sample_size is not None
                ):
                    raise ValueError(
                        "Single axis or sample-size values cannot configure multiple "
                        "batch groups. Omit axis_size, sample_size, and "
                        "batch_sample_size when using multi_size='manager'."
                    )

                return BatchManager.from_model(
                    model,
                    batch_size=batch_size,
                    position_keys=position_keys,
                    shuffle=shuffle,
                    batch_axes=batch_axes,
                    default_batch_axis=default_batch_axis,
                    epoch_size=epoch_size,
                    infer_sample_size=infer_sample_size,
                    sample_with_replacement=sample_with_replacement,
                    sampling_weights=sampling_weights,
                )

            raise ValueError(
                "Batches.from_model() found multiple observation groups "
                f"with axis sizes {[size for size, _ in groups]}. Use "
                "Batches.from_model(..., multi_size='manager') or "
                "BatchManager.from_model(...)."
            )

        if axis_size is None:
            axis_size = (
                groups[0][0]
                if groups
                else _axis_size_for_empty_position_keys(
                    model, batch_axes, default_batch_axis
                )
            )

        if batch_size is None:
            shuffle = False

        (weights,) = _sampling_weights_for_groups(sampling_weights, [pos_keys])
        batches = cls(
            pos_keys,
            batch_size=batch_size,
            axis_size=axis_size,
            shuffle=shuffle,
            batch_axes=batch_axes,
            default_batch_axis=default_batch_axis,
            sample_size=sample_size,
            batch_sample_size=batch_sample_size,
            sample_with_replacement=sample_with_replacement,
            sampling_weights=weights,
        )

        if infer_sample_size and sample_size is None:
            batches.sample_size = _infer_sample_size_from_state(
                model,
                model.state,
                pos_keys,
                axis_size,
            )

            if batch_sample_size is None:
                obs = model.extract_position(pos_keys)
                batch = batches.get_batched_position(obs, 0)
                batch_state = model.update_state(batch, model.state)
                assert batches.batch_size is not None
                batches.batch_sample_size = _infer_sample_size_from_state(
                    model,
                    batch_state,
                    pos_keys,
                    batches.batch_size,
                )

        return batches

    @property
    def batch_sample_scales(self) -> tuple[float]:
        """
        Batch likelihood scaling factors.

        Returns
        -------
        tuple[float]
            A one-element tuple containing :attr:`batch_sample_scale`.

        Examples
        --------
        >>> from liesel.optim import Batches
        >>> Batches(["y"], axis_size=10, batch_size=4).batch_sample_scales
        (2.5,)
        """
        return (self.batch_sample_scale,)

    @property
    def n_full_batches(self) -> int:
        """
        Number of complete batches.

        Returns
        -------
        int
            The integer quotient ``axis_size // batch_size``. An oversized batch
            configured for replacement has one complete batch.

        Examples
        --------
        >>> from liesel.optim import Batches
        >>> Batches(["y"], axis_size=10, batch_size=4).n_full_batches
        2
        """
        assert self.batch_size is not None
        if self.axis_size < self.batch_size:
            return 1

        return int(self.axis_size // self.batch_size)

    @property
    def batch_sample_scale(self) -> float:
        """
        Mini-batch likelihood scaling factor.

        Returns
        -------
        float
            The ratio ``sample_size / batch_sample_size``. Directly constructed
            batches without explicit sample sizes fall back to
            ``axis_size / batch_size``.

        Examples
        --------
        >>> from liesel.optim import Batches
        >>> Batches(["y"], axis_size=10, batch_size=4).batch_sample_scale
        2.5
        """
        assert self.batch_size is not None
        sample_size = (
            float(self.axis_size) if self.sample_size is None else self.sample_size
        )
        batch_sample_size = (
            float(self.batch_size)
            if self.batch_sample_size is None
            else self.batch_sample_size
        )

        return float(sample_size / batch_sample_size)

    @property
    def is_full_data(self) -> bool:
        """
        Whether the object represents one full-data batch.

        Examples
        --------
        >>> from liesel.optim import Batches
        >>> Batches(["y"], axis_size=5, batch_size=None).is_full_data
        True
        """
        return self.axis_size == self.batch_size and not self.sample_with_replacement

    def permute_indices(self, key: jax.Array) -> jax.Array:
        """
        Returns epoch indices, optionally shuffled.

        This method does not mutate :attr:`indices`. Assign the return value to
        ``indices`` if the object should use the new order.

        Parameters
        ----------
        key
            JAX pseudo-random key used when ``shuffle=True``.

        Returns
        -------
        jax.Array
            For ordinary batching, a vector of indices from ``0`` to
            ``axis_size - 1``. With replacement, ``n_full_batches * batch_size``
            independent in-range draws. The order is random if ``shuffle=True`` and
            unchanged otherwise.

        Examples
        --------
        >>> import jax
        >>> from liesel.optim import Batches

        >>> batches = Batches(["y"], axis_size=6, batch_size=3, shuffle=False)
        >>> batches.permute_indices(jax.random.key(0)).tolist()
        [0, 1, 2, 3, 4, 5]

        >>> shuffled = Batches(["y"], axis_size=6, batch_size=3, shuffle=True)
        >>> shuffled.indices = shuffled.permute_indices(jax.random.key(0))
        >>> sorted(shuffled.batch_indices.ravel().tolist())
        [0, 1, 2, 3, 4, 5]
        """
        if self._uses_replacement:
            assert self.batch_size is not None
            n_indices = self.n_full_batches * self.batch_size
            if self.shuffle:
                return self._draw_indices(key, n_indices)

            return self._default_indices()

        indices = jnp.arange(self.axis_size)
        all_indices = jax.random.permutation(key, indices) if self.shuffle else indices

        return all_indices

    def _validate_n_batches(self, n_batches: int | None) -> int:
        if n_batches is None:
            return self.n_full_batches
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError("n_batches must be a positive integer or None.")
        if n_batches <= 0:
            raise ValueError("n_batches must be a positive integer or None.")
        return n_batches

    def _assemble_indices(
        self, key: jax.Array | None, n_batches: int | None
    ) -> jax.Array:
        assert self.batch_size is not None
        if key is None and n_batches is None:
            if self._uses_replacement:
                return (
                    jnp.arange(self.n_full_batches * self.batch_size) % self.axis_size
                )
            return jnp.arange(self.axis_size)

        if n_batches is None and not self._uses_replacement:
            assert key is not None
            return (
                jax.random.permutation(key, self.axis_size)
                if self.shuffle
                else jnp.arange(self.axis_size)
            )

        n_batches = self._validate_n_batches(n_batches)
        n_indices = n_batches * self.batch_size
        if self._uses_replacement:
            if key is None:
                return jnp.arange(n_indices) % self.axis_size
            return self._draw_indices(key, n_indices)

        if not self.shuffle:
            if n_batches > self.n_full_batches:
                raise ValueError(
                    "shuffle=False cannot assemble more than the natural number "
                    "of complete batches."
                )
            return jnp.arange(n_indices)

        n_per_pass = self.n_full_batches * self.batch_size
        if key is None:
            return jnp.tile(jnp.arange(n_per_pass), math.ceil(n_indices / n_per_pass))[
                :n_indices
            ]
        keys = jax.random.split(key, math.ceil(n_batches / self.n_full_batches))
        passes = [
            jax.random.permutation(pass_key, self.axis_size)[:n_per_pass]
            for pass_key in keys
        ]
        return jnp.concatenate(passes)[:n_indices]

    def _draw_indices(self, key: jax.Array, size: int) -> jax.Array:
        if self.sampling_probabilities is None:
            return jax.random.randint(key, (size,), 0, self.axis_size)
        assert self._alias_table is not None
        # Match the default index dtype used by the epoch's initial carry.
        return _alias.sample(key, size, self._alias_table).astype(int)

    def start_epoch(self, key: jax.Array, n_batches: int | None = None) -> Batches:
        """
        Starts a new epoch by updating the observation order.

        Parameters
        ----------
        key
            JAX pseudo-random key used for epoch assembly.
        n_batches
            ``None`` assembles the natural epoch. For ordinary batching, the
            unused incomplete tail remains in :attr:`indices`. A positive integer
            requests exactly that many complete batch rows.

        Returns
        -------
        Batches
            This object with freshly assigned :attr:`indices`.

        Examples
        --------
        >>> import jax
        >>> from liesel.optim import Batches
        >>> batches = Batches(["y"], axis_size=5, batch_size=2, shuffle=False)
        >>> batches.start_epoch(jax.random.key(0)).indices.tolist()
        [0, 1, 2, 3, 4]
        """
        self.indices = self._assemble_indices(key, n_batches)
        return self

    def _replace_indices_for_manager(self, n_batches: int) -> Batches:
        self.indices = self._assemble_indices(None, n_batches)
        return self

    @property
    def batch_indices(self) -> jax.Array:
        """
        Batch index matrix.

        Returns
        -------
        jax.Array
            Integer array with shape ``(n_assembled_batches, batch_size)``. Each
            row gives one complete batch; the first dimension can exceed natural
            ``n_full_batches`` when a manager requests overflow rows.

        Examples
        --------
        >>> from liesel.optim import Batches
        >>> Batches(
        ...     ["y"], axis_size=7, batch_size=3, shuffle=False
        ... ).batch_indices.tolist()
        [[0, 1, 2], [3, 4, 5]]
        """
        assert self.batch_size is not None
        n_indices = (self.indices.size // self.batch_size) * self.batch_size
        batch_indices = jnp.reshape(self.indices[:n_indices], (-1, self.batch_size))
        return batch_indices

    def get_batched_position(
        self, position: Position, batch_index: int | jax.Array
    ) -> Position:
        """
        Slices observed position entries for one batch.

        Parameters
        ----------
        position
            Mapping from position key to array. Every key listed in
            ``position_keys`` must be present and have length ``axis_size`` along
            its batching axis.
        batch_index
            Row number in :attr:`batch_indices`.

        Returns
        -------
        Position
            Position containing only the batched entries named in ``position_keys``.

        Raises
        ------
        ValueError
            If an entry's length along its batching axis is not equal to ``axis_size``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import Batches

        >>> batches = Batches(["y"], axis_size=6, batch_size=2, shuffle=False)
        >>> position = {"y": jnp.arange(6)}
        >>> batches.get_batched_position(position, batch_index=1)["y"].tolist()
        [2, 3]

        Batch along a non-leading axis:

        >>> batches = Batches(
        ...     ["x"],
        ...     axis_size=4,
        ...     batch_size=2,
        ...     default_batch_axis=1,
        ...     shuffle=False,
        ... )
        >>> position = {"x": jnp.arange(12).reshape(3, 4)}
        >>> batches.get_batched_position(position, batch_index=0)["x"].tolist()
        [[0, 1], [4, 5], [8, 9]]
        """
        idx = self.batch_indices[batch_index]
        batched_position = {}
        assert isinstance(self.batch_axes, dict)
        for key in self.position_keys:
            axis = self.batch_axes.get(key, self.default_batch_axis)

            n_this_key = jnp.shape(position[key])[axis]
            if not jnp.shape(position[key])[axis] == self.axis_size:
                raise ValueError(
                    f"{key} has axis_size={n_this_key}, which is incompatible with the "
                    f"given axis_size={self.axis_size}."
                )

            batched = jnp.take(position[key], idx, axis=axis)
            batched_position[key] = batched

        return Position(batched_position)

    def extract_batched_position(
        self,
        interface: ModelInterface | Model,
        model_state: ModelState,
        batch_number: int,
    ) -> Position:
        """
        Extracts observed data from a model state and returns one batch.

        Parameters
        ----------
        interface
            Model or model interface used to extract the observed position entries.
        model_state
            State from which ``position_keys`` are extracted.
        batch_number
            Row number in :attr:`batch_indices`.

        Returns
        -------
        Position
            Batched position extracted from ``model_state``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.optim import Batches

        >>> y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        >>> model = lsl.Model([y])
        >>> batches = Batches(["y"], axis_size=6, batch_size=2, shuffle=False)
        >>> batches.extract_batched_position(model, model.state, 2)["y"].tolist()
        [4.0, 5.0]
        """
        obs = interface.extract_position(self.position_keys, model_state)
        return self.get_batched_position(obs, batch_number)

    def correction_factors(self, batch_index: int | jax.Array) -> jax.Array:
        """Return extra factors ``1 / (axis_size * p_i)`` for the selected batch.

        Uniform sampling returns ones. These factors do not include
        :attr:`batch_sample_scale`.
        """
        indices = self.batch_indices[batch_index]
        if self.sampling_probabilities is None:
            return jnp.ones(indices.shape)
        return 1.0 / (self.axis_size * self.sampling_probabilities[indices])

    def _likelihood_corrections(
        self,
        model: Model | ModelInterface,
        model_state: ModelState,
        batch_index: int | jax.Array | None,
    ) -> dict[str, tuple[jax.Array, int]]:
        if self.sampling_probabilities is None:
            return {}
        if batch_index is None:
            raise ValueError("Weighted likelihood correction requires batch_index.")
        if not isinstance(model, Model):
            raise TypeError(
                "Weighted likelihood correction requires a liesel.model.Model."
            )
        if _has_custom_model_log_lik(model):
            raise ValueError(
                "Weighted correction cannot decompose a custom log_lik_node. "
                "Apply correction_factors in a custom loss instead."
            )
        observed_names = {
            var.name
            for var in model.observed.values()
            if var.name in self.position_keys
            or var.value_node.name in self.position_keys
        }
        unknown = self.likelihood_axes.keys() - observed_names
        if unknown:
            raise ValueError(
                f"likelihood_axes names are not in this batch group: {sorted(unknown)}."
            )
        factors = self.correction_factors(batch_index)
        corrections = {}
        assert isinstance(self.batch_axes, dict)
        for var in model.observed.values():
            if var.dist_node is None:
                continue
            key = next(
                (
                    key
                    for key in self.position_keys
                    if key in (var.name, var.value_node.name)
                ),
                None,
            )
            if key is None:
                continue
            value = jnp.asarray(model_state[var.dist_node.name].value)
            if not value.ndim:
                raise ValueError(f"{var.name!r} needs a pointwise likelihood axis.")
            if var.name in self.likelihood_axes:
                axis = self.likelihood_axes[var.name]
            else:
                data_ndim = model_state[var.value_node.name].value.ndim
                event_ndim = len(var.dist_node.init_dist().event_shape)
                axis = self.batch_axes.get(key, self.default_batch_axis)
                if not -data_ndim <= axis < data_ndim:
                    raise ValueError(f"{var.name!r} has an invalid data batching axis.")
                axis %= data_ndim
                # Standard log_prob removes trailing event dimensions and may
                # add leading broadcast dimensions; it preserves the rest.
                shift = value.ndim - (data_ndim - event_ndim)
                if axis >= data_ndim - event_ndim or shift < 0:
                    raise ValueError(
                        f"Cannot infer a pointwise likelihood axis for {var.name!r}. "
                        "Set likelihood_axes explicitly if the likelihood still "
                        "enumerates the sampled observations."
                    )
                axis += shift
            if not -value.ndim <= axis < value.ndim:
                raise ValueError(f"{var.name!r} has an invalid likelihood_axes entry.")
            axis %= value.ndim
            if value.shape[axis] != self.batch_size:
                raise ValueError(
                    f"{var.name!r} likelihood axis must have batch_size entries."
                )
            corrections[var.dist_node.name] = (factors, axis)
        return corrections

    def scaled_log_lik(
        self,
        model: Model | ModelInterface,
        model_state: ModelState,
        *,
        batch_index: int | jax.Array | None = None,
    ) -> jax.Array:
        """
        Returns the log likelihood with this batch group's likelihood scaled.

        For a :class:`.Model`, observed likelihood terms belonging to
        :attr:`position_keys` are multiplied by :attr:`batch_sample_scale`. Other
        observed likelihood terms are left unscaled.

        Parameters
        ----------
        model
            Liesel model or compatible model interface.
        model_state
            Updated model state containing the current log-likelihood values.
        batch_index
            Row of :attr:`batch_indices` used to produce ``model_state``. Required
            for weighted sampling. Applies :meth:`correction_factors` before
            summation in addition to :attr:`batch_sample_scale`. Reduced scalar
            likelihoods and ambiguous axis mappings are rejected.

        Returns
        -------
        jax.Array
            Scaled log likelihood.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import tensorflow_probability.substrates.jax.distributions as tfd
        >>> from liesel.optim import Batches

        >>> y = lsl.Var.new_obs(
        ...     jnp.arange(6.0),
        ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        ...     name="y",
        ... )
        >>> model = lsl.Model([y])
        >>> batches = Batches(["y"], axis_size=6, batch_size=2, shuffle=False)
        >>> batched = batches.get_batched_position(model.extract_position(["y"]), 0)
        >>> state = model.update_state(batched, model.state)
        >>> bool(
        ...     jnp.allclose(
        ...         batches.scaled_log_lik(model, state),
        ...         batches.batch_sample_scale * state["_model_log_lik"].value,
        ...     )
        ... )
        True
        """
        corrections = self._likelihood_corrections(model, model_state, batch_index)
        if isinstance(model, Model):
            return _scaled_liesel_log_lik(
                model,
                model_state,
                [(self.position_keys, self.batch_sample_scale)],
                corrections,
            )

        return _scaled_common_log_lik(model_state, self.batch_sample_scale)

    def _tree_flatten(self):
        # Unweighted checkpoints from before alias sampling remain compatible.
        table = getattr(self, "_alias_table", None)
        if self.sampling_probabilities is not None and table is None:
            raise ValueError(
                "Weighted checkpoint predates alias sampling and cannot be resumed. "
                "Start a new run with an unused checkpoint path."
            )
        children = (self.indices, self.sampling_probabilities, table)
        aux_data = {
            "position_keys": self.position_keys,
            "axis_size": self.axis_size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "batch_axes": self.batch_axes,
            "default_batch_axis": self.default_batch_axis,
            "sample_with_replacement": self.sample_with_replacement,
            "sample_size": self.sample_size,
            "batch_sample_size": self.batch_sample_size,
            "likelihood_axes": self.likelihood_axes,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        bi = object.__new__(cls)
        for name, value in aux_data.items():
            setattr(bi, name, value)
        bi.indices, bi._sampling_probabilities, bi._alias_table = children
        return bi

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(axis_size={self.axis_size}, "
            f"batch_size={self.batch_size}, "
            f"default_batch_axis={self.default_batch_axis})"
        )
        return out


@dataclass
class BatchManager:
    """
    Coordinates multiple :class:`Batches` objects as one batching interface.

    A ``BatchManager`` is useful when a model contains observed branches with
    different observation sizes. Each contained :class:`Batches` object owns the
    slicing rules for one branch. The manager combines them into one joint batched
    position for every optimizer step.

    Parameters
    ----------
    batches
        Non-empty sequence of :class:`Batches` objects. Their ``position_keys`` must
        not overlap.
    epoch_size
        Epoch length policy: ``"strict"``, ``"min"``, ``"max"``, or a positive
        integer.
    sampling_weights
        Optional keyword-only weights: a vector for a single child, or a mapping
        from one child position key per group to its vector. Each vector applies
        to the entire group and requires that child's
        ``sample_with_replacement=True``. Unknown keys and multiple entries for
        one group are rejected. Supplied weights override existing child weights
        on a copy; omitted groups retain their existing sampling configuration.

    Attributes
    ----------
    batches
        Tuple of contained :class:`Batches` objects.

    Raises
    ------
    ValueError
        If ``batches`` is empty, if any ``position_keys`` are claimed by more than
        one child, if ``epoch_size`` is invalid, or if strict sizing is used with
        unequal child :attr:`Batches.n_full_batches`.

    Notes
    -----
    The properties :attr:`axis_size`, :attr:`batch_size`, and
    :attr:`batch_sample_scales` return tuples in child-batch order. The scalar
    aliases are available only when all children have the same likelihood scale.
    With unequal scales, use :meth:`scaled_log_lik` so each branch is scaled by
    its own sample-size ratio.

    Use manual ``BatchManager([Batches(...)])`` construction when child groups need
    custom per-branch ``sample_size`` or ``batch_sample_size`` values.

    Like :class:`Batches`, :meth:`start_epoch` mutates and returns ``self``.

    Examples
    --------
    Combine two equally long batch sequences:

    >>> import jax.numpy as jnp
    >>> from liesel.optim import BatchManager, Batches

    >>> manager = BatchManager(
    ...     [
    ...         Batches(["x"], axis_size=6, batch_size=2, shuffle=False),
    ...         Batches(["y"], axis_size=9, batch_size=3, shuffle=False),
    ...     ]
    ... )
    >>> manager.n_full_batches
    3
    >>> position = {"x": jnp.arange(6), "y": jnp.arange(9)}
    >>> batched = manager.get_batched_position(position, 1)
    >>> batched["x"].tolist(), batched["y"].tolist()
    ([2, 3], [3, 4, 5])

    With ``epoch_size="max"``, shorter branches assemble additional shuffled passes:

    >>> import jax
    >>> manager = BatchManager(
    ...     [
    ...         Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
    ...         Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
    ...     ],
    ...     epoch_size="max",
    ... ).start_epoch(jax.random.key(0))
    >>> manager.n_full_batches
    3

    Per-branch scaling agrees with a manual scaled log-likelihood calculation:

    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> y1 = lsl.Var.new_obs(
    ...     jnp.arange(6.0),
    ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    ...     name="y1",
    ... )
    >>> y2 = lsl.Var.new_obs(
    ...     jnp.arange(8.0),
    ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    ...     name="y2",
    ... )
    >>> model = lsl.Model([y1, y2])
    >>> manager = BatchManager(
    ...     [
    ...         Batches(["y1"], axis_size=6, batch_size=2, shuffle=True),
    ...         Batches(["y2"], axis_size=8, batch_size=4, shuffle=True),
    ...     ],
    ...     epoch_size="max",
    ... )
    >>> batch = manager.get_batched_position(model.extract_position(["y1", "y2"]), 0)
    >>> state = model.update_state(batch, model.state)
    >>> manual = (
    ...     3.0 * state["y1_log_prob"].value.sum()
    ...     + 2.0 * state["y2_log_prob"].value.sum()
    ... )
    >>> bool(jnp.allclose(manager.scaled_log_lik(model, state), manual))
    True
    """

    batches: Sequence[Batches]
    epoch_size: Literal["strict", "min", "max"] | int = "strict"
    sampling_weights: InitVar[Array | Mapping[str, Array] | None] = field(
        default=None, kw_only=True
    )

    def __post_init__(self, sampling_weights):
        self.batches = tuple(self.batches)

        if len(self.batches) == 0:
            raise ValueError("BatchManager requires at least one Batches object.")

        if isinstance(self.epoch_size, bool) or (
            not isinstance(self.epoch_size, int)
            and self.epoch_size not in ("strict", "max", "min")
        ):
            raise ValueError(
                "epoch_size must be 'strict', 'min', 'max', or a positive integer."
            )

        if isinstance(self.epoch_size, int) and self.epoch_size < 1:
            raise ValueError("Manual epoch_size must be a positive integer.")

        self._validate_position_keys()
        self._validate_batch_counts()
        weights = _sampling_weights_for_groups(
            sampling_weights, [batch.position_keys for batch in self.batches]
        )
        batches = []
        for batch, weight in zip(self.batches, weights, strict=True):
            if weight is not None:
                batch = copy(batch)
                batch._set_sampling_weights(weight)
            batches.append(batch)
        self.batches = tuple(batches)
        count = self.n_full_batches
        self.batches = tuple(
            batch._replace_indices_for_manager(count) for batch in self.batches
        )

    @classmethod
    def from_model(
        cls,
        model: Model,
        batch_size: int | None | object = _MISSING,
        position_keys: Sequence[str] | Sequence[Sequence[str]] | None = None,
        shuffle: bool = True,
        batch_axes: dict[str, int] | None = None,
        default_batch_axis: int = 0,
        epoch_size: Literal["strict", "min", "max"] | int = "max",
        infer_sample_size: bool = True,
        sample_with_replacement: bool = False,
        *,
        batch_axis_size: int | None | object = _MISSING,
        sampling_weights: Array | Mapping[str, Array] | None = None,
    ) -> BatchManager:
        """
        Builds a :class:`BatchManager` from inferred or explicit groups.

        Flat keys are grouped by inferred length along their batching axes; nested
        keys specify exact groups. One child :class:`Batches` is created per group using
        the same ``batch_size``. With the default ``epoch_size="max"``, shorter
        branches assemble additional shuffled passes for the joint steps.

        Use manual ``BatchManager([Batches(...)])`` construction when child groups
        need custom per-branch ``sample_size`` or ``batch_sample_size`` values.

        Parameters
        ----------
        model
            Model containing the observed variables to batch.
        batch_size
            Common batch size for every child group. If ``None``, each child uses one
            full-data batch and shuffling is disabled.
        position_keys
            Names of observed position entries to batch. If ``None``, all observed
            variables in ``model`` are used. Flat keys are grouped by axis length;
            nested keys preserve exact groups in the supplied order. Each group
            must have matching lengths along its configured batching axes.
        shuffle
            Whether each child should shuffle observation indices at epoch start.
        batch_axes
            Optional mapping from position key to batching axis.
        default_batch_axis
            Batching axis for all position keys not listed in ``batch_axes``.
        epoch_size
            Epoch length policy: ``"strict"``, ``"min"``, ``"max"``, or a positive
            integer.
        infer_sample_size
            Whether child batches should infer missing effective sample sizes from
            observed log-probability values. Inference counts log-probability
            scalars, not observed value elements; for multivariate observation
            distributions, one observed event may have several value dimensions but
            one pointwise log-probability scalar.
        sample_with_replacement
            Whether every assembled batch draws observations independently with
            replacement. This applies to every inferred child; automatic construction
            also enables it for an oversized child when the common batch size exceeds
            its observation count.
        sampling_weights
            A vector for a single group, or a mapping from one selected position
            key per group to its weight vector. Vectors apply to all aligned entries
            in the group, in current data order. Omitted groups use uniform sampling.
            Unknown keys and multiple entries for one group are rejected. Weighted
            groups must use replacement sampling; normally set
            ``sample_with_replacement=True``. See :class:`Batches` for validation
            and likelihood correction.

        Returns
        -------
        BatchManager
            Batch manager with one child :class:`Batches` object per observation
            group, including separate groups of equal size.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.optim import BatchManager
        >>> x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        >>> y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        >>> model = lsl.Model([x, y])
        >>> manager = BatchManager.from_model(
        ...     model,
        ...     batch_size=2,
        ...     position_keys=["x", "y"],
        ... )
        >>> manager.axis_size, manager.batch_size, manager.n_full_batches
        ((8, 5), (2, 2), 4)
        >>> started = manager.start_epoch(jax.random.key(1))
        >>> started.batch_indices[1].shape
        (4, 2)

        Passing ``batch_size=None`` creates one full-data child batch per group:

        >>> full_data = BatchManager.from_model(
        ...     model,
        ...     batch_size=None,
        ...     position_keys=["x", "y"],
        ... )
        >>> full_data.is_full_data, full_data.n_full_batches
        (True, 1)
        """
        batch_size = _resolve_batch_size(batch_size, batch_axis_size)
        pos_keys = (
            list(position_keys) if position_keys is not None else list(model.observed)
        )
        pos_keys, groups = position_key_groups_from_model(
            model, pos_keys, batch_axes, default_batch_axis
        )
        shuffle = False if batch_size is None else shuffle

        batches = []
        for axis_size, keys in groups:
            batch = Batches.from_model(
                model,
                batch_size=batch_size,
                position_keys=keys,
                axis_size=axis_size,
                shuffle=shuffle,
                batch_axes=batch_axes,
                default_batch_axis=default_batch_axis,
                infer_sample_size=infer_sample_size,
                sample_with_replacement=(
                    sample_with_replacement
                    or (batch_size is not None and batch_size > axis_size)
                ),
            )
            assert isinstance(batch, Batches)
            batches.append(batch)

        return cls(
            batches=batches,
            epoch_size=epoch_size,
            sampling_weights=sampling_weights,
        )

    def _validate_position_keys(self) -> None:
        counts: dict[str, int] = {}

        for batch in self.batches:
            for key in batch.position_keys:
                counts[key] = counts.get(key, 0) + 1

        duplicates = [key for key, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Position keys claimed by multiple batches: {duplicates}")

    def _validate_batch_counts(self) -> None:
        counts = [batch.n_full_batches for batch in self.batches]
        if self.epoch_size == "strict" and len(set(counts)) != 1:
            raise ValueError(
                "epoch_size='strict' requires all contained Batches objects to have "
                f"same n_full_batches, but got {counts}."
            )
        count = self.n_full_batches
        if any(
            not batch.shuffle and count > batch.n_full_batches for batch in self.batches
        ):
            raise ValueError(
                "epoch_size requires additional batches from a shuffle=False child."
            )

    @property
    def position_keys(self) -> list[str]:
        """
        Position keys claimed by all contained batch objects.

        Returns
        -------
        list[str]
            Concatenated ``position_keys`` in child-batch order.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> manager = BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y", "z"], axis_size=9, batch_size=3),
        ...     ]
        ... )
        >>> manager.position_keys
        ['x', 'y', 'z']
        """
        keys: list[str] = []
        for batch in self.batches:
            keys.extend(batch.position_keys)
        return keys

    @property
    def axis_size(self) -> tuple[int, ...]:
        """
        Number of observations for each contained batch object.

        Returns
        -------
        tuple[int, ...]
            One axis size per child :class:`Batches` object.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=9, batch_size=3),
        ...     ]
        ... ).axis_size
        (6, 9)
        """
        return tuple(batch.axis_size for batch in self.batches)

    @property
    def sample_sizes(self) -> tuple[float | None, ...]:
        """Full-data sample sizes for each contained batch object."""
        return tuple(batch.sample_size for batch in self.batches)

    @property
    def batch_sample_sizes(self) -> tuple[float | None, ...]:
        """Batch sample sizes for each contained batch object."""
        return tuple(batch.batch_sample_size for batch in self.batches)

    @property
    def batch_size(self) -> tuple[int, ...]:
        """
        Batch size for each contained batch object.

        Returns
        -------
        tuple[int, ...]
            One batch size per child :class:`Batches` object.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=9, batch_size=3),
        ...     ]
        ... ).batch_size
        (2, 3)
        """
        sizes: list[int] = []
        for batch in self.batches:
            assert batch.batch_size is not None
            sizes.append(batch.batch_size)

        return tuple(sizes)

    @property
    def batch_sample_scales(self) -> tuple[float, ...]:
        """
        Likelihood scaling factors for each contained batch object.

        Returns
        -------
        tuple[float, ...]
            The child-specific ratios ``sample_size / batch_sample_size``. Directly
            constructed batches without explicit sample sizes fall back to
            ``axis_size / batch_size``.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     epoch_size="min",
        ... ).batch_sample_scales
        (3.0, 2.0)
        """
        return tuple(batch.batch_sample_scale for batch in self.batches)

    @property
    def batch_sample_scale(self) -> float:
        """
        Common likelihood scaling factor.

        Returns
        -------
        float
            The common child ratio ``sample_size / batch_sample_size``. Directly
            constructed batches without explicit sample sizes fall back to
            ``axis_size / batch_size``.

        Raises
        ------
        ValueError
            If the contained batch objects have unequal values in
            :attr:`batch_sample_scales`.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> manager = BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=9, batch_size=3),
        ...     ]
        ... )
        >>> manager.batch_sample_scale
        3.0

        With unequal child scales, use :meth:`scaled_log_lik` instead:

        >>> unequal = BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     epoch_size="min",
        ... )
        >>> try:
        ...     unequal.batch_sample_scale
        ... except ValueError as error:
        ...     print("scaled_log_lik" in str(error))
        True
        """
        if not self._has_common_batch_sample_scale:
            raise ValueError(
                "BatchManager.batch_sample_scale is only available when all contained "
                "Batches objects have the same sample-size ratio. Use "
                "per-branch scaling via BatchManager.scaled_log_lik() instead."
            )

        return self.batch_sample_scales[0]

    @property
    def _has_common_batch_sample_scale(self) -> bool:
        first = self.batch_sample_scales[0]
        return all(abs(scale - first) <= 1e-12 for scale in self.batch_sample_scales)

    @property
    def n_full_batches(self) -> int:
        """
        Number of joint batch steps in one epoch.

        With ``epoch_size="strict"``, this is the common child
        :attr:`Batches.n_full_batches`; otherwise it is determined by
        :attr:`epoch_size`.

        Returns
        -------
        int
            Joint epoch length.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     epoch_size="max",
        ... ).n_full_batches
        3
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     epoch_size="min",
        ... ).n_full_batches
        2
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     epoch_size=5,
        ... ).n_full_batches
        5
        """
        counts = [batch.n_full_batches for batch in self.batches]

        if self.epoch_size == "strict":
            return counts[0]

        if self.epoch_size == "max":
            return max(counts)

        if self.epoch_size == "min":
            return min(counts)

        assert isinstance(self.epoch_size, int)
        return self.epoch_size

    @property
    def is_full_data(self) -> bool:
        """
        Whether every child represents one full-data batch.

        Returns
        -------
        bool
            ``True`` if all contained :class:`Batches` objects have
            :attr:`Batches.is_full_data`.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=None),
        ...         Batches(["y"], axis_size=8, batch_size=None),
        ...     ]
        ... ).is_full_data
        True
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=None),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     epoch_size="min",
        ... ).is_full_data
        False
        """
        return all(batch.is_full_data for batch in self.batches)

    def permute_indices(self, key: jax.Array) -> tuple[jax.Array, ...]:
        """
        Returns fresh epoch indices for every contained batch object.

        This method mirrors :meth:`Batches.permute_indices` for each child. It does
        not mutate the manager or the child ``indices``. Use :meth:`start_epoch` to
        update the manager in place.

        Parameters
        ----------
        key
            JAX pseudo-random key split across children.

        Returns
        -------
        tuple[jax.Array, ...]
            One index vector per contained :class:`Batches` object.

        Examples
        --------
        >>> import jax
        >>> from liesel.optim import BatchManager, Batches
        >>> manager = BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=4, batch_size=2, shuffle=False),
        ...         Batches(["y"], axis_size=6, batch_size=3, shuffle=False),
        ...     ]
        ... )
        >>> tuple(idx.tolist() for idx in manager.permute_indices(jax.random.key(0)))
        ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5])
        """
        keys = jax.random.split(key, len(self.batches))
        return tuple(
            batch.permute_indices(subkey)
            for batch, subkey in zip(self.batches, keys, strict=True)
        )

    def start_epoch(self, key: jax.Array) -> BatchManager:
        """
        Starts a new joint epoch.

        The manager updates every child via :meth:`Batches.start_epoch` and
        updates every child with exactly the joint number of assembled rows.

        Parameters
        ----------
        key
            JAX pseudo-random key used for child permutations and replacement draws.

        Returns
        -------
        BatchManager
            This object with updated child indices.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from liesel.optim import BatchManager, Batches
        >>> manager = BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=4, batch_size=2, shuffle=True),
        ...         Batches(["y"], axis_size=6, batch_size=3, shuffle=True),
        ...     ],
        ...     epoch_size=4,
        ... ).start_epoch(jax.random.key(1))
        """
        keys = jax.random.split(key, len(self.batches))
        self.batches = tuple(
            batch.start_epoch(child_key, n_batches=self.n_full_batches)
            for batch, child_key in zip(self.batches, keys, strict=True)
        )
        return self

    @property
    def batch_indices(self) -> tuple[jax.Array, ...]:
        """
        Batch index matrices selected for the joint epoch.

        Returns
        -------
        tuple[jax.Array, ...]
            One integer array per child. The ``i``-th array has shape
            ``(n_full_batches, child_batch_size)`` and contains the observation
            indices selected for that child at each joint batch step.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> manager = BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2, shuffle=False),
        ...         Batches(["y"], axis_size=9, batch_size=3, shuffle=False),
        ...     ]
        ... )
        >>> tuple(idx.tolist() for idx in manager.batch_indices)
        ([[0, 1], [2, 3], [4, 5]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        """
        return tuple(batch.batch_indices for batch in self.batches)

    def get_batched_position(
        self, position: Position, batch_index: int | jax.Array
    ) -> Position:
        """
        Returns the joint batched position for one optimizer step.

        Each child :class:`Batches` object slices the entries named in its own
        ``position_keys``. The resulting partial positions are merged into a single
        :class:`Position`.

        Parameters
        ----------
        position
            Mapping containing every key in :attr:`position_keys`.
        batch_index
            Joint batch row in ``0, ..., n_full_batches - 1``.

        Returns
        -------
        Position
            Batched entries from all contained batch objects.

        Raises
        ------
        ValueError
            If any child finds an incompatible observation size along its batching
            axis.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import BatchManager, Batches
        >>> manager = BatchManager(
        ...     [
        ...         Batches(
        ...             ["x"],
        ...             axis_size=4,
        ...             batch_size=2,
        ...             default_batch_axis=1,
        ...             shuffle=False,
        ...         ),
        ...         Batches(["y"], axis_size=6, batch_size=3, shuffle=False),
        ...     ]
        ... )
        >>> position = {
        ...     "x": jnp.arange(8).reshape(2, 4),
        ...     "y": jnp.arange(6),
        ... }
        >>> batch = manager.get_batched_position(position, batch_index=1)
        >>> batch["x"].tolist(), batch["y"].tolist()
        ([[2, 3], [6, 7]], [3, 4, 5])
        """
        batched_position: dict[str, Array] = {}

        for batch in self.batches:
            batched_position |= batch.get_batched_position(position, batch_index)

        return Position(batched_position)

    def extract_batched_position(
        self,
        interface: ModelInterface | Model,
        model_state: ModelState,
        batch_number: int,
    ) -> Position:
        """
        Extracts observed data from a model state and returns one joint batch.

        Parameters
        ----------
        interface
            Model or model interface used to extract the observed position entries.
        model_state
            State from which :attr:`position_keys` are extracted.
        batch_number
            Joint batch row in ``0, ..., n_full_batches - 1``.

        Returns
        -------
        Position
            Batched observed position entries from ``model_state``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.optim import BatchManager, Batches
        >>> x = lsl.Var.new_obs(jnp.arange(4.0), name="x")
        >>> y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        >>> model = lsl.Model([x, y])
        >>> manager = BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=4, batch_size=2, shuffle=False),
        ...         Batches(["y"], axis_size=6, batch_size=3, shuffle=False),
        ...     ]
        ... )
        >>> batch = manager.extract_batched_position(model, model.state, 1)
        >>> batch["x"].tolist(), batch["y"].tolist()
        ([2.0, 3.0], [3.0, 4.0, 5.0])
        """
        obs = interface.extract_position(self.position_keys, model_state)
        return self.get_batched_position(obs, batch_number)

    def correction_factors(self, batch_index: int | jax.Array) -> tuple[jax.Array, ...]:
        """Return extra per-index correction factors for each child, in order."""
        return tuple(batch.correction_factors(batch_index) for batch in self.batches)

    def scaled_log_lik(
        self,
        model: Model | ModelInterface,
        model_state: ModelState,
        *,
        batch_index: int | jax.Array | None = None,
    ) -> jax.Array:
        """
        Returns a log likelihood with per-child batch scaling.

        For a :class:`.Model`, each child group scales the observed likelihood terms
        belonging to its ``position_keys`` by that child's
        :attr:`Batches.batch_sample_scale`. Observed likelihood terms not covered
        by any child are left unscaled.

        For a generic :class:`.ModelInterface`, per-branch decomposition is not
        available. In that case, this method can only use the old scalar path and
        therefore requires a common :attr:`batch_sample_scale`.

        Parameters
        ----------
        model
            Liesel model or compatible model interface.
        model_state
            Updated model state containing the current log-likelihood values.
        batch_index
            Current batch row, required when any child uses weighted sampling.
            Each weighted child corrects its own pointwise likelihood terms before
            summation. Priors and unbatched terms receive no per-index correction.

        Returns
        -------
        jax.Array
            Scaled log likelihood.

        Raises
        ------
        ValueError
            If ``model`` is a generic interface and the child batch shares differ.
        TypeError
            If ``model`` is a generic interface and ``model_state`` does not expose
            ``"_model_log_lik"``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import tensorflow_probability.substrates.jax.distributions as tfd
        >>> from liesel.optim import BatchManager, Batches
        >>> x = lsl.Var.new_obs(
        ...     jnp.arange(6.0),
        ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        ...     name="x",
        ... )
        >>> y = lsl.Var.new_obs(
        ...     jnp.arange(8.0),
        ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        ...     name="y",
        ... )
        >>> model = lsl.Model([x, y])
        >>> manager = BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2, shuffle=True),
        ...         Batches(["y"], axis_size=8, batch_size=4, shuffle=True),
        ...     ],
        ...     epoch_size="min",
        ... )
        >>> batch = manager.get_batched_position(model.extract_position(["x", "y"]), 0)
        >>> state = model.update_state(batch, model.state)
        >>> manual = (
        ...     manager.batches[0].batch_sample_scale * state["x_log_prob"].value.sum()
        ...     + manager.batches[1].batch_sample_scale
        ...     * state["y_log_prob"].value.sum()
        ... )
        >>> bool(jnp.allclose(manager.scaled_log_lik(model, state), manual))
        True
        """
        corrections = {}
        for batch in self.batches:
            corrections.update(
                batch._likelihood_corrections(model, model_state, batch_index)
            )
        if isinstance(model, Model):
            groups = [
                (batch.position_keys, batch.batch_sample_scale)
                for batch in self.batches
            ]
            return _scaled_liesel_log_lik(model, model_state, groups, corrections)

        return _scaled_common_log_lik(model_state, self.batch_sample_scale)

    def _tree_flatten(self):
        children = (tuple(self.batches),)
        aux_data = {"epoch_size": self.epoch_size}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        (batches,) = children
        bm = object.__new__(cls)
        bm.batches = tuple(batches)
        bm.epoch_size = aux_data["epoch_size"]
        return bm

    def __repr__(self) -> str:
        name = type(self).__name__
        return (
            f"{name}(axis_size={self.axis_size}, "
            f"batch_size={self.batch_size}, "
            f"epoch_size={self.epoch_size!r}, n_full_batches={self.n_full_batches})"
        )


jax.tree_util.register_pytree_node(
    Batches, Batches._tree_flatten, Batches._tree_unflatten
)

jax.tree_util.register_pytree_node(
    BatchManager, BatchManager._tree_flatten, BatchManager._tree_unflatten
)
