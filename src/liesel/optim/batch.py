from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from ..model import Model
from ._log_lik import scaled_common_log_lik as _scaled_common_log_lik
from ._log_lik import scaled_liesel_log_lik as _scaled_liesel_log_lik
from ._model_utils import position_key_groups_from_model
from .split import (
    _count_likelihood_contributions,
    _has_custom_model_log_lik,
    _observed_dist_infos,
)
from .types import Array, ModelInterface, ModelState, Position
from .util import guess_n

_MISSING = object()


def _resolve_batch_size(
    batch_size: int | None | object,
    batch_axis_size: int | None | object,
) -> int | None:
    if batch_size is _MISSING and batch_axis_size is _MISSING:
        raise TypeError("missing required argument: 'batch_size'")

    if batch_size is not _MISSING and batch_axis_size is not _MISSING:
        raise TypeError("Pass either batch_size or batch_axis_size, not both.")

    return batch_axis_size if batch_size is _MISSING else batch_size  # type: ignore[return-value]


def _normalize_positive_size(size: int | float | None, name: str) -> float | None:
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


@dataclass(init=False)
class Batches:
    """
    Defines mini-batches for observed entries in an optimizer position.

    ``Batches`` stores an index vector of length ``axis_size`` and reshapes the first
    complete part of that vector into batches. The observed position entries named in
    ``position_keys`` are sliced with these indices. By default, every entry is
    sliced along axis ``0``; use ``default_split_axis`` or ``split_axes`` for
    arrays where observations live on another axis.

    Parameters
    ----------
    position_keys
        Names of the position entries that should be batched.
    axis_size
        Number of observations along each batched axis.
    batch_size
        Number of observations per batch. If ``None``, batching is disabled by using
        a single batch with all ``axis_size`` observations.
    shuffle
        Whether :meth:`permute_indices` should return a random permutation of the
        indices. If ``False``, :meth:`permute_indices` returns the indices unchanged.
    split_axes
        Optional mapping from position key to batching axis. Keys missing from this
        mapping use ``default_split_axis``.
    default_split_axis
        Batching axis for all position keys not listed in ``split_axes``.
    sample_with_replacement
        Whether an oversized batch may be filled by sampling observations with
        replacement. This is mainly used by :meth:`BatchManager.from_model` when
        ``mode="resample"`` and a common ``batch_size`` is larger than a branch's
        observation count.

    Attributes
    ----------
    indices
        Current ordering of the observations. Initialized as
        ``jnp.arange(axis_size)`` and used by :attr:`batch_indices`. Assign the
        result of :meth:`permute_indices` to this attribute to use a fresh order.

    Notes
    -----
    If ``axis_size`` is not divisible by ``batch_size``, only full batches
    are used and the final incomplete batch is dropped.

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

    ``split_axes`` can batch different entries along different split_axes:

    >>> import jax.numpy as jnp
    >>> batches = Batches(
    ...     ["x", "y"],
    ...     axis_size=5,
    ...     batch_size=2,
    ...     split_axes={"x": 1},
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
    split_axes: dict[str, int] | None = None
    default_split_axis: int = 0
    sample_with_replacement: bool = False
    sample_size: int | float | None = None
    batch_sample_size: int | float | None = None

    def __init__(
        self,
        position_keys: Sequence[str],
        axis_size: int,
        batch_size: int | None | object = _MISSING,
        shuffle: bool = True,
        split_axes: dict[str, int] | None = None,
        default_split_axis: int = 0,
        sample_with_replacement: bool = False,
        sample_size: int | float | None = None,
        batch_sample_size: int | float | None = None,
        *,
        batch_axis_size: int | None | object = _MISSING,
    ) -> None:
        self.position_keys = position_keys
        self.axis_size = axis_size
        self.batch_size = _resolve_batch_size(batch_size, batch_axis_size)
        self.shuffle = shuffle
        self.split_axes = split_axes
        self.default_split_axis = default_split_axis
        self.sample_with_replacement = sample_with_replacement
        self.sample_size = sample_size
        self.batch_sample_size = batch_sample_size
        self.__post_init__()

    def __post_init__(self):
        if self.axis_size < 1:
            raise ValueError(f"{self.axis_size=} is < 1, which is not allowed.")

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

        if self.split_axes is None:
            self.split_axes = {}

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
        assert self.batch_size is not None
        return self.sample_with_replacement and self.axis_size < self.batch_size

    def _default_indices(self) -> jax.Array:
        if self._uses_replacement:
            assert self.batch_size is not None
            return jnp.arange(self.n_full_batches * self.batch_size) % self.axis_size

        return jnp.arange(self.axis_size)

    @classmethod
    def from_model(
        cls,
        model: Model,
        batch_size: int | None | object = _MISSING,
        position_keys: Sequence[str] | None = None,
        axis_size: int | None = None,
        shuffle: bool = True,
        split_axes: dict[str, int] | None = None,
        default_split_axis: int = 0,
        multi_size: Literal["error", "manager"] = "error",
        mode: Literal["strict", "resample"] = "resample",
        epoch_size: Literal["max", "min"] | int = "max",
        sample_size: int | float | None = None,
        batch_sample_size: int | float | None = None,
        infer_sample_size: bool = True,
        sample_with_replacement: bool = False,
        *,
        batch_axis_size: int | None | object = _MISSING,
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
            variables in ``model`` are used.
        axis_size
            Number of observations. If ``None``, the number is guessed from the model's
            observed variables along ``default_split_axis``.
        shuffle
            Whether epoch-wise calls to :meth:`permute_indices` should shuffle the
            indices. This is forced to ``False`` when ``batch_size`` is ``None``.
        split_axes
            Optional mapping from position key to batching axis.
        default_split_axis
            Axis used for guessing ``axis_size`` and for position keys missing
            from ``split_axes``.
        multi_size
            How to handle observed variables with different inferred axis sizes.
            The default ``"error"`` keeps :class:`Batches` scalar and raises a
            helpful error. Use ``"manager"`` to return a :class:`BatchManager` when
            multiple axis sizes are detected.
        mode
            Batch manager mode used only when ``multi_size="manager"``. The default
            ``"resample"`` allows branches with fewer complete batches to sample
            batch rows with replacement.
        epoch_size
            Batch manager epoch size used only when ``multi_size="manager"``.

        Returns
        -------
        Batches or BatchManager
            Batch configuration for the model's observed data. A
            :class:`BatchManager` is returned only when ``multi_size="manager"`` and
            multiple axis sizes are detected.

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
        groups = position_key_groups_from_model(
            model, pos_keys, split_axes, default_split_axis
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
                    position_keys=pos_keys,
                    shuffle=shuffle,
                    split_axes=split_axes,
                    default_split_axis=default_split_axis,
                    mode=mode,
                    epoch_size=epoch_size,
                    infer_sample_size=infer_sample_size,
                )

            raise ValueError(
                "Batches.from_model() found observed variables with different "
                f"axis sizes: {groups}. Use "
                "Batches.from_model(..., multi_size='manager') or "
                "BatchManager.from_model(...)."
            )

        if axis_size is None:
            axis_size = (
                next(iter(groups))
                if groups
                else guess_n(model, axis=default_split_axis)
            )

        if batch_size is None:
            shuffle = False

        batches = cls(
            pos_keys,
            batch_size=batch_size,
            axis_size=axis_size,
            shuffle=shuffle,
            split_axes=split_axes,
            default_split_axis=default_split_axis,
            sample_size=sample_size,
            batch_sample_size=batch_sample_size,
            sample_with_replacement=sample_with_replacement,
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
            The integer quotient ``axis_size // batch_size``.

        Examples
        --------
        >>> from liesel.optim import Batches
        >>> Batches(["y"], axis_size=10, batch_size=4).n_full_batches
        2
        """
        assert self.batch_size is not None
        if self._uses_replacement:
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
        return self.axis_size == self.batch_size

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
            A vector of indices from ``0`` to ``axis_size - 1``. The order is random if
            ``shuffle=True`` and unchanged otherwise.

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
                return jax.random.randint(key, (n_indices,), 0, self.axis_size)

            return self._default_indices()

        if self.shuffle:
            all_indices = jax.random.permutation(key, self.indices)
        else:
            all_indices = self.indices

        return all_indices

    def start_epoch(self, key: jax.Array) -> Batches:
        """
        Starts a new epoch by updating the observation order.

        Parameters
        ----------
        key
            JAX pseudo-random key passed to :meth:`permute_indices`.

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
        self.indices = self.permute_indices(key)
        return self

    @property
    def batch_indices(self) -> jax.Array:
        """
        Batch index matrix.

        Returns
        -------
        jax.Array
            Integer array with shape ``(n_full_batches, batch_size)``. Each
            row gives the observation indices for one full batch.

        Examples
        --------
        >>> from liesel.optim import Batches
        >>> Batches(
        ...     ["y"], axis_size=7, batch_size=3, shuffle=False
        ... ).batch_indices.tolist()
        [[0, 1, 2], [3, 4, 5]]
        """
        assert self.batch_size is not None
        idx = self.indices[: self.n_full_batches * self.batch_size]
        batch_indices = jnp.reshape(idx, (self.n_full_batches, self.batch_size))
        return batch_indices

    def get_batched_position(self, position: Position, batch_index: int) -> Position:
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
        ...     default_split_axis=1,
        ...     shuffle=False,
        ... )
        >>> position = {"x": jnp.arange(12).reshape(3, 4)}
        >>> batches.get_batched_position(position, batch_index=0)["x"].tolist()
        [[0, 1], [4, 5], [8, 9]]
        """
        idx = self.batch_indices[batch_index]
        batched_position = {}
        assert isinstance(self.split_axes, dict)
        for key in self.position_keys:
            axis = self.split_axes.get(key, self.default_split_axis)

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

    def scaled_log_lik(
        self, model: Model | ModelInterface, model_state: ModelState
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
        if isinstance(model, Model):
            return _scaled_liesel_log_lik(
                model, model_state, [(self.position_keys, self.batch_sample_scale)]
            )

        return _scaled_common_log_lik(model_state, self.batch_sample_scale)

    def _tree_flatten(self):
        children = (self.indices,)
        aux_data = {
            "position_keys": self.position_keys,
            "axis_size": self.axis_size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "split_axes": self.split_axes,
            "default_split_axis": self.default_split_axis,
            "sample_with_replacement": self.sample_with_replacement,
            "sample_size": self.sample_size,
            "batch_sample_size": self.batch_sample_size,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        bi = cls(**aux_data)
        bi.indices = children[0]
        return bi

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(axis_size={self.axis_size}, "
            f"batch_size={self.batch_size}, "
            f"default_split_axis={self.default_split_axis})"
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
    mode
        If ``"strict"``, all contained batch objects must have the same
        :attr:`Batches.n_full_batches`. If ``"resample"``, unequal numbers of batches
        are allowed and child batch rows are selected for the joint epoch.
    epoch_size
        Epoch length in ``"resample"`` mode. ``"max"`` uses the longest child
        epoch, ``"min"`` uses the shortest child epoch, and a positive integer sets
        the epoch length manually. ``"max"`` and ``"min"`` are accepted in
        ``"strict"`` mode but do not change the strict epoch length.

    Attributes
    ----------
    batches
        Tuple of contained :class:`Batches` objects.
    batch_numbers
        Integer array with shape ``(n_full_batches, len(batches))``. Row ``i`` maps
        the manager's joint batch ``i`` to one batch row in each contained
        :class:`Batches` object.

    Raises
    ------
    ValueError
        If ``batches`` is empty, if any ``position_keys`` are claimed by more than
        one child, if ``mode`` or ``epoch_size`` are invalid, or if ``mode="strict"``
        is used with unequal child :attr:`Batches.n_full_batches`.

    Notes
    -----
    The properties :attr:`axis_size`, :attr:`batch_size`, and
    :attr:`batch_sample_scales` return tuples in child-batch order. The scalar
    aliases are available only when all children have the same likelihood scale.
    With unequal scales, use :meth:`scaled_log_lik` so each branch is scaled by
    its own sample-size ratio.

    Like :class:`Batches`, :meth:`start_epoch` mutates and returns ``self``.

    Examples
    --------
    Combine two equally long batch sequences in strict mode:

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

    In ``"resample"`` mode, branches with fewer batches can be sampled with
    replacement to match a chosen epoch length:

    >>> import jax
    >>> manager = BatchManager(
    ...     [
    ...         Batches(["x"], axis_size=6, batch_size=2, shuffle=False),
    ...         Batches(["y"], axis_size=8, batch_size=4, shuffle=False),
    ...     ],
    ...     mode="resample",
    ...     epoch_size="max",
    ... ).start_epoch(jax.random.key(0))
    >>> manager.n_full_batches
    3
    >>> manager.batch_numbers.shape
    (3, 2)

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
    ...         Batches(["y1"], axis_size=6, batch_size=2, shuffle=False),
    ...         Batches(["y2"], axis_size=8, batch_size=4, shuffle=False),
    ...     ],
    ...     mode="resample",
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
    mode: Literal["strict", "resample"] = "strict"
    epoch_size: Literal["max", "min"] | int = "max"

    def __post_init__(self):
        self.batches = tuple(self.batches)

        if len(self.batches) == 0:
            raise ValueError("BatchManager requires at least one Batches object.")

        if self.mode not in ("strict", "resample"):
            raise ValueError(f"Unrecognized {self.mode=}.")

        if isinstance(self.epoch_size, bool) or (
            not isinstance(self.epoch_size, int)
            and self.epoch_size not in ("max", "min")
        ):
            raise ValueError("epoch_size must be 'max', 'min', or a positive integer.")

        if isinstance(self.epoch_size, int) and self.epoch_size < 1:
            raise ValueError("Manual epoch_size must be a positive integer.")

        if self.mode == "strict" and isinstance(self.epoch_size, int):
            raise ValueError(
                "Manual epoch_size is only supported with mode='resample'."
            )

        self._validate_position_keys()
        self._validate_batch_counts()
        self.batch_numbers = self._default_batch_numbers()

    @classmethod
    def from_model(
        cls,
        model: Model,
        batch_size: int | None | object = _MISSING,
        position_keys: Sequence[str] | None = None,
        shuffle: bool = True,
        split_axes: dict[str, int] | None = None,
        default_split_axis: int = 0,
        mode: Literal["strict", "resample"] = "resample",
        epoch_size: Literal["max", "min"] | int = "max",
        infer_sample_size: bool = True,
        *,
        batch_axis_size: int | None | object = _MISSING,
    ) -> BatchManager:
        """
        Builds a :class:`BatchManager` by grouping observed variables by size.

        Observed variables are grouped by inferred length along their batching axis.
        One child :class:`Batches` object is created for each axis-size group using
        the same ``batch_size``. With the default ``mode="resample"`` and
        ``epoch_size="max"``, branches with fewer complete batches sample batch rows
        with replacement for the additional joint steps.

        Parameters
        ----------
        model
            Model containing the observed variables to batch.
        batch_size
            Common batch size for every child group. If ``None``, each child uses one
            full-data batch and shuffling is disabled.
        position_keys
            Names of observed position entries to batch. If ``None``, all observed
            variables in ``model`` are used.
        shuffle
            Whether each child should shuffle observation indices at epoch start.
        split_axes
            Optional mapping from position key to batching axis.
        default_split_axis
            Batching axis for all position keys not listed in ``split_axes``.
        mode
            Batch manager mode. ``"resample"`` allows unequal numbers of child
            batches; ``"strict"`` requires all child groups to have the same number
            of complete batches.
        epoch_size
            Epoch length in ``"resample"`` mode. ``"max"`` uses the longest child
            epoch, ``"min"`` uses the shortest child epoch, and a positive integer
            sets the epoch length manually.

        Returns
        -------
        BatchManager
            Batch manager with one child :class:`Batches` object per inferred
            axis size.

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
        >>> bool(jnp.all(started.batch_numbers[:, 1] < 2))
        True

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
        groups = position_key_groups_from_model(
            model, pos_keys, split_axes, default_split_axis
        )
        shuffle = False if batch_size is None else shuffle

        if mode == "resample" and batch_size is not None and not shuffle:
            warnings.warn(
                "BatchManager.from_model(..., mode='resample', shuffle=False) "
                "resamples child batch rows but leaves observations within each "
                "child in deterministic order. Set shuffle=True for stochastic "
                "observation-level batches.",
                UserWarning,
                stacklevel=2,
            )

        batches = []
        for axis_size, keys in groups.items():
            batch = Batches.from_model(
                model,
                batch_size=batch_size,
                position_keys=keys,
                axis_size=axis_size,
                shuffle=shuffle,
                split_axes=split_axes,
                default_split_axis=default_split_axis,
                infer_sample_size=infer_sample_size,
                sample_with_replacement=(
                    mode == "resample"
                    and batch_size is not None
                    and batch_size > axis_size
                ),
            )
            assert isinstance(batch, Batches)
            batches.append(batch)

        return cls(batches=batches, mode=mode, epoch_size=epoch_size)

    def _validate_position_keys(self) -> None:
        counts: dict[str, int] = {}

        for batch in self.batches:
            for key in batch.position_keys:
                counts[key] = counts.get(key, 0) + 1

        duplicates = [key for key, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Position keys claimed by multiple batches: {duplicates}")

    def _validate_batch_counts(self) -> None:
        if self.mode == "resample":
            return

        counts = [batch.n_full_batches for batch in self.batches]
        if len(set(counts)) != 1:
            raise ValueError(
                "mode='strict' requires all contained Batches objects to have the "
                f"same n_full_batches, but got {counts}."
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
            The child-specific ratios ``axis_size / batch_size``.

        Examples
        --------
        >>> from liesel.optim import BatchManager, Batches
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     mode="resample",
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
            The common child ratio ``axis_size / batch_size``.

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
        ...     mode="resample",
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

        In ``"strict"`` mode, this is the common child
        :attr:`Batches.n_full_batches`. In ``"resample"`` mode, it is determined by
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
        ...     mode="resample",
        ...     epoch_size="max",
        ... ).n_full_batches
        3
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     mode="resample",
        ...     epoch_size="min",
        ... ).n_full_batches
        2
        >>> BatchManager(
        ...     [
        ...         Batches(["x"], axis_size=6, batch_size=2),
        ...         Batches(["y"], axis_size=8, batch_size=4),
        ...     ],
        ...     mode="resample",
        ...     epoch_size=5,
        ... ).n_full_batches
        5
        """
        counts = [batch.n_full_batches for batch in self.batches]

        if self.mode == "strict":
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
        ...     mode="resample",
        ... ).is_full_data
        False
        """
        return all(batch.is_full_data for batch in self.batches)

    def _default_batch_numbers(self) -> jax.Array:
        rows = []

        for batch in self.batches:
            rows.append(jnp.arange(self.n_full_batches) % batch.n_full_batches)

        return jnp.stack(rows, axis=1)

    def _draw_batch_numbers(self, batch: Batches, key: jax.Array) -> jax.Array:
        n_manager_batches = self.n_full_batches
        n_child_batches = batch.n_full_batches

        if self.mode == "strict":
            return jnp.arange(n_manager_batches)

        key_base, key_extra = jax.random.split(key)
        shuffled = jax.random.permutation(key_base, jnp.arange(n_child_batches))

        if n_manager_batches <= n_child_batches:
            return shuffled[:n_manager_batches]

        extra = jax.random.randint(
            key_extra,
            shape=(n_manager_batches - n_child_batches,),
            minval=0,
            maxval=n_child_batches,
        )
        return jnp.concatenate([shuffled, extra])

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
        recomputes :attr:`batch_numbers`. In ``"strict"`` mode, joint batch ``i``
        uses child batch row ``i`` for every child. In ``"resample"`` mode, each
        child uses shuffled rows without replacement where possible and samples
        additional rows with replacement if the joint epoch is longer than that
        child's own epoch.

        Parameters
        ----------
        key
            JAX pseudo-random key used for child permutations and, in
            ``"resample"`` mode, row selection.

        Returns
        -------
        BatchManager
            This object with updated child ``indices`` and :attr:`batch_numbers`.

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
        ...     mode="resample",
        ...     epoch_size=4,
        ... ).start_epoch(jax.random.key(1))
        >>> manager.batch_numbers.shape
        (4, 2)
        >>> bool(jnp.all(manager.batch_numbers[:, 0] < 2))
        True
        >>> bool(jnp.all(manager.batch_numbers[:, 1] < 2))
        True
        """
        keys = jax.random.split(key, len(self.batches) * 2)
        batches = []
        batch_numbers = []

        for i, batch in enumerate(self.batches):
            index_key = keys[2 * i]
            row_key = keys[2 * i + 1]
            batch = batch.start_epoch(index_key)
            batches.append(batch)
            batch_numbers.append(self._draw_batch_numbers(batch, row_key))

        self.batches = tuple(batches)
        self.batch_numbers = jnp.stack(batch_numbers, axis=1)
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
        return tuple(
            batch.batch_indices[self.batch_numbers[:, i]]
            for i, batch in enumerate(self.batches)
        )

    def get_batched_position(self, position: Position, batch_index: int) -> Position:
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
        ...             default_split_axis=1,
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

        for i, batch in enumerate(self.batches):
            child_batch_index = self.batch_numbers[batch_index, i]
            batched_position |= batch.get_batched_position(position, child_batch_index)

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

    def scaled_log_lik(
        self, model: Model | ModelInterface, model_state: ModelState
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
        ...         Batches(["x"], axis_size=6, batch_size=2, shuffle=False),
        ...         Batches(["y"], axis_size=8, batch_size=4, shuffle=False),
        ...     ],
        ...     mode="resample",
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
        if isinstance(model, Model):
            groups = [
                (batch.position_keys, batch.batch_sample_scale)
                for batch in self.batches
            ]
            return _scaled_liesel_log_lik(model, model_state, groups)

        return _scaled_common_log_lik(model_state, self.batch_sample_scale)

    def _tree_flatten(self):
        children = (tuple(self.batches), self.batch_numbers)
        aux_data = {
            "mode": self.mode,
            "epoch_size": self.epoch_size,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        batches, batch_numbers = children
        bm = cls(batches=batches, **aux_data)
        bm.batch_numbers = batch_numbers
        return bm

    def __repr__(self) -> str:
        name = type(self).__name__
        return (
            f"{name}(axis_size={self.axis_size}, "
            f"batch_size={self.batch_size}, "
            f"mode={self.mode!r}, n_full_batches={self.n_full_batches})"
        )


jax.tree_util.register_pytree_node(
    Batches, Batches._tree_flatten, Batches._tree_unflatten
)

jax.tree_util.register_pytree_node(
    BatchManager, BatchManager._tree_flatten, BatchManager._tree_unflatten
)
