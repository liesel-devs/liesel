"""Train, validation, and test splitting utilities for optimizer positions."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp

from ...model import Model
from ._log_lik import scaled_common_log_lik, scaled_liesel_log_lik
from .types import Array, ModelInterface, ModelState, Position
from .util import guess_n


def _merge_positions(positions: Sequence[Position]) -> Position:
    merged: dict[str, Array] = {}

    for position in positions:
        duplicates = sorted(set(merged) & set(position))
        if duplicates:
            raise ValueError(
                f"Position keys claimed by multiple split groups: {duplicates}"
            )
        merged.update(position)

    return Position(merged)


def _common_value(values: Sequence, name: str, plural_name: str):
    values = tuple(values)
    first = values[0]

    for value in values[1:]:
        if isinstance(first, float) or isinstance(value, float):
            equal = abs(float(value) - float(first)) <= 1e-12
        else:
            equal = value == first

        if not equal:
            raise ValueError(
                f"PositionSplitManager.{name} is only available when all contained "
                f"PositionSplit objects have the same {name}. Use "
                f"PositionSplitManager.{plural_name} for branch-specific values."
            )

    return first


def _validate_unique_position_keys(groups: Sequence[Sequence[str]], owner: str) -> None:
    counts: dict[str, int] = {}

    for group in groups:
        for key in group:
            counts[key] = counts.get(key, 0) + 1

    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Position keys claimed by multiple {owner}: {duplicates}")


def _position_key_groups_from_model(
    model: Model,
    position_keys: Sequence[str],
    axes: dict[str, int] | None,
    default_axis: int,
) -> dict[int, list[str]]:
    axes = axes or {}
    position = model.extract_position(position_keys)
    groups: dict[int, list[str]] = {}

    for key in position_keys:
        axis = axes.get(key, default_axis)
        n_key = int(jnp.shape(position[key])[axis])
        groups.setdefault(n_key, []).append(key)

    return groups


def _child_seeds(
    seed: jax.Array | int | None, n_children: int
) -> tuple[jax.Array | int | None, ...]:
    if seed is None:
        return (None,) * n_children

    key = seed if isinstance(seed, jax.Array) else jax.random.key(seed)
    return tuple(jax.random.split(key, n_children))


@dataclass
class PositionSplit:
    """
    Container for train, validation, and test position dictionaries.

    A ``PositionSplit`` is usually returned by :meth:`Split.split_position` or
    :meth:`PositionSplit.from_model`. It stores the split observed position entries
    together with the corresponding split sizes.

    Parameters
    ----------
    train
        Position entries used for optimization.
    validate
        Position entries used for validation or early stopping.
    test
        Position entries reserved for testing.
    n_train
        Number of training observations.
    n_validate
        Number of validation observations.
    n_test
        Number of test observations.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim import PositionSplit
    >>> from liesel.experimental.optim.types import Position
    >>> split = PositionSplit(
    ...     train=Position({"x": jnp.arange(3)}),
    ...     validate=Position({"x": jnp.arange(3, 5)}),
    ...     test=Position({"x": jnp.arange(5, 6)}),
    ...     n_train=3,
    ...     n_validate=2,
    ...     n_test=1,
    ... )
    >>> split
    PositionSplit(train=3, validate=2, test=1)
    >>> split.validate["x"].tolist()
    [3, 4]
    """

    train: Position
    validate: Position
    test: Position

    n_train: int
    n_validate: int
    n_test: int

    @property
    def position_keys(self) -> list[str]:
        """
        Position keys contained in this split.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> split = PositionSplit(
        ...     Position({"x": jnp.arange(2)}),
        ...     Position({}),
        ...     Position({}),
        ...     n_train=2,
        ...     n_validate=0,
        ...     n_test=0,
        ... )
        >>> split.position_keys
        ['x']
        """
        return list(self.train)

    @property
    def n(self) -> int:
        """
        Total number of observations represented by the split.

        Examples
        --------
        >>> from liesel.experimental.optim import PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> PositionSplit(Position({}), Position({}), Position({}), 7, 2, 1).n
        10
        """
        return self.n_train + self.n_validate + self.n_test

    @property
    def has_validation(self) -> bool:
        """
        Whether this split contains validation observations.

        Examples
        --------
        >>> from liesel.experimental.optim import PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> PositionSplit(
        ...     Position({}), Position({}), Position({}), 7, 2, 1
        ... ).has_validation
        True
        """
        return self.n_validate > 0

    @property
    def has_test(self) -> bool:
        """
        Whether this split contains test observations.

        Examples
        --------
        >>> from liesel.experimental.optim import PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> PositionSplit(Position({}), Position({}), Position({}), 7, 0, 0).has_test
        False
        """
        return self.n_test > 0

    @property
    def share_validate(self) -> float:
        """
        Share of observations assigned to validation.

        Examples
        --------
        >>> from liesel.experimental.optim import PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> PositionSplit(
        ...     Position({}), Position({}), Position({}), 7, 2, 1
        ... ).share_validate
        0.2
        """
        return self.n_validate / self.n

    @property
    def share_test(self) -> float:
        """
        Share of observations assigned to testing.

        Examples
        --------
        >>> from liesel.experimental.optim import PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> PositionSplit(Position({}), Position({}), Position({}), 7, 2, 1).share_test
        0.1
        """
        return self.n_test / self.n

    @property
    def scale_validate(self) -> float:
        """
        Likelihood scale for validation data.

        Returns ``1.0`` when no validation split is present.

        Examples
        --------
        >>> from liesel.experimental.optim import PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> PositionSplit(
        ...     Position({}), Position({}), Position({}), 8, 2, 0
        ... ).scale_validate
        4.0
        """
        return self._scale_for_part("validate")

    def _scale_for_part(self, part: Literal["train", "validate", "test"]) -> float:
        if part == "train":
            return 1.0

        if part == "validate":
            if not self.has_validation:
                return 1.0
            return self.n_train / self.n_validate

        if part == "test":
            if not self.has_test:
                return 1.0
            return self.n_train / self.n_test

        raise ValueError(f"Unrecognized {part=}.")

    def scaled_log_lik(
        self,
        model: Model | ModelInterface,
        model_state: ModelState,
        part: Literal["train", "validate", "test"] = "validate",
    ) -> jax.Array:
        """
        Returns the log likelihood scaled for one split part.

        For a :class:`.Model`, observed likelihood terms belonging to this split's
        :attr:`position_keys` are multiplied by the branch scale for ``part``.
        Other observed likelihood terms are left unscaled. For a generic model
        interface, the scalar ``"_model_log_lik"`` state entry is scaled.

        Parameters
        ----------
        model
            Liesel model or compatible model interface.
        model_state
            Updated model state containing log-likelihood values for the selected
            split part.
        part
            Split part whose scale should be applied. Validation uses
            ``n_train / n_validate``.

        Returns
        -------
        jax.Array
            Scaled log likelihood.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import tensorflow_probability.substrates.jax.distributions as tfd
        >>> from liesel.experimental.optim import Split
        >>> y = lsl.Var.new_obs(
        ...     jnp.arange(10.0),
        ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        ...     name="y",
        ... )
        >>> model = lsl.Model([y])
        >>> split = Split(["y"], n=10, n_validate=2).split_position(
        ...     model.extract_position(["y"])
        ... )
        >>> state = model.update_state(split.validate, model.state)
        >>> bool(
        ...     jnp.allclose(
        ...         split.scaled_log_lik(model, state),
        ...         split.scale_validate * state["_model_log_lik"].value,
        ...     )
        ... )
        True
        """
        scale = self._scale_for_part(part)

        if isinstance(model, Model):
            return scaled_liesel_log_lik(
                model, model_state, [(self.position_keys, scale)]
            )

        return scaled_common_log_lik(model_state, scale)

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validate={self.n_validate}, test={self.n_test})"
        )
        return out

    @staticmethod
    def from_model(
        model: Model,
        position_keys: Sequence[str] | None = None,
        n: int | None = None,
        share_validate: float = 0.0,
        share_test: float = 0.0,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
        multi_size: Literal["error", "manager"] = "error",
    ) -> PositionSplit | PositionSplitManager:
        """
        Builds a :class:`PositionSplit` from the observed variables in a model.

        Parameters
        ----------
        model
            Model containing the observed variables to split.
        position_keys
            Names of observed position entries to split. If ``None``, all observed
            variables in ``model`` are used.
        n
            Number of observations along the split axis. If ``None``, the number is
            guessed from ``model`` along ``default_axis``.
        share_validate
            Share of observations assigned to the validation split.
        share_test
            Share of observations assigned to the test split.
        axes
            Optional mapping from position key to split axis. Keys missing from this
            mapping use ``default_axis``.
        default_axis
            Split axis for all position keys not listed in ``axes``.
        shuffle
            Whether observations are shuffled before splitting.
        seed
            Seed or JAX pseudo-random key used when ``shuffle=True``.
        multi_size
            How to handle observed variables with different inferred sample sizes.
            The default ``"error"`` keeps :class:`PositionSplit` scalar and raises
            a helpful error. Use ``"manager"`` to return a
            :class:`PositionSplitManager` with one split per sample-size group.

        Returns
        -------
        PositionSplit or PositionSplitManager
            Split observed position entries extracted from ``model``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.experimental.optim import PositionSplit
        >>> y = lsl.Var.new_obs(jnp.arange(10.0), name="y")
        >>> model = lsl.Model([y])
        >>> split = PositionSplit.from_model(
        ...     model,
        ...     position_keys=["y"],
        ...     share_validate=0.2,
        ...     share_test=0.1,
        ... )
        >>> split.n_train, split.n_validate, split.n_test
        (7, 2, 1)
        >>> split.train["y"].tolist()
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        Multi-size observed data must opt into the manager API:

        >>> y1_multi = lsl.Var.new_obs(jnp.arange(10.0), name="y1_multi")
        >>> y2_multi = lsl.Var.new_obs(jnp.arange(6.0), name="y2_multi")
        >>> model = lsl.Model([y1_multi, y2_multi])
        >>> managed = PositionSplit.from_model(
        ...     model,
        ...     position_keys=["y1_multi", "y2_multi"],
        ...     share_validate=0.2,
        ...     multi_size="manager",
        ... )
        >>> type(managed).__name__
        'PositionSplitManager'
        """
        if multi_size not in ("error", "manager"):
            raise ValueError("multi_size must be 'error' or 'manager'.")

        if multi_size == "manager":
            return PositionSplitManager.from_model(
                model,
                position_keys=position_keys,
                share_validate=share_validate,
                share_test=share_test,
                axes=axes,
                default_axis=default_axis,
                shuffle=shuffle,
                seed=seed,
            )

        pos_keys = (
            list(position_keys) if position_keys is not None else list(model.observed)
        )
        groups = _position_key_groups_from_model(model, pos_keys, axes, default_axis)
        if len(groups) > 1:
            raise ValueError(
                "PositionSplit.from_model() found observed variables with different "
                f"sample sizes: {groups}. Use "
                "PositionSplit.from_model(..., multi_size='manager') or "
                "PositionSplitManager.from_model(...)."
            )

        if n is None:
            n = next(iter(groups)) if groups else guess_n(model, axis=default_axis)
        splitter = Split.from_share(
            position_keys=pos_keys,
            n=n,
            share_validate=share_validate,
            share_test=share_test,
            axes=axes,
            default_axis=default_axis,
            shuffle=shuffle,
            seed=seed,
        )

        pos = model.extract_position(pos_keys)
        return splitter.split_position(pos)


@dataclass
class PositionSplitManager:
    """
    Coordinates multiple :class:`PositionSplit` objects as one split interface.

    ``PositionSplitManager`` is the split-side counterpart to
    :class:`.BatchManager`. It is useful when a model has observed branches with
    different sample sizes. Each child :class:`PositionSplit` stores the split data
    for one branch, while the manager exposes merged ``train``, ``validate``, and
    ``test`` positions.

    Parameters
    ----------
    splits
        Non-empty sequence of :class:`PositionSplit` objects. Their
        :attr:`PositionSplit.position_keys` must not overlap. Either all children
        must contain validation data or none may contain validation data; the same
        rule applies to test data.

    Raises
    ------
    ValueError
        If ``splits`` is empty, if position keys overlap, or if validation/test
        availability differs across children.

    Notes
    -----
    Branch-specific sizes are available as :attr:`ns`, :attr:`n_trains`,
    :attr:`n_validates`, and :attr:`n_tests`. Scalar aliases such as
    :attr:`n_train`, :attr:`share_validate`, and :attr:`scale_validate` are available
    only when all children have the same value.

    Examples
    --------
    Merge two branches with different sample sizes:

    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim import PositionSplitManager, Split
    >>> position = {"x": jnp.arange(10), "y": jnp.arange(6)}
    >>> split_x = Split(["x"], n=10, n_validate=2).split_position(position)
    >>> split_y = Split(["y"], n=6, n_validate=1).split_position(position)
    >>> manager = PositionSplitManager([split_x, split_y])
    >>> manager.position_keys
    ['x', 'y']
    >>> manager.n_trains
    (8, 5)
    >>> manager.train["x"].shape, manager.train["y"].shape
    ((8,), (5,))

    Unequal scalar aliases raise and direct users to the plural property:

    >>> try:
    ...     manager.n_train
    ... except ValueError as error:
    ...     print("n_trains" in str(error))
    True

    Build a manager directly from a model with two observation sizes:

    >>> import liesel.model as lsl
    >>> y1 = lsl.Var.new_obs(jnp.arange(10.0), name="y1")
    >>> y2 = lsl.Var.new_obs(jnp.arange(6.0), name="y2")
    >>> model = lsl.Model([y1, y2])
    >>> managed = PositionSplitManager.from_model(
    ...     model, position_keys=["y1", "y2"], share_validate=0.2
    ... )
    >>> managed.n_validates
    (2, 1)
    """

    splits: Sequence[PositionSplit]

    def __post_init__(self):
        self.splits = tuple(self.splits)

        if len(self.splits) == 0:
            raise ValueError(
                "PositionSplitManager requires at least one PositionSplit object."
            )

        _validate_unique_position_keys(
            [split.position_keys for split in self.splits], "split groups"
        )
        self._validate_split_availability("validation")
        self._validate_split_availability("test")

    def _validate_split_availability(self, part: Literal["validation", "test"]):
        if part == "validation":
            values = [split.has_validation for split in self.splits]
        else:
            values = [split.has_test for split in self.splits]

        if len(set(values)) != 1:
            raise ValueError(
                f"All contained PositionSplit objects must either have {part} data "
                f"or none may have {part} data."
            )

    @classmethod
    def from_model(
        cls,
        model: Model,
        position_keys: Sequence[str] | None = None,
        share_validate: float = 0.0,
        share_test: float = 0.0,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
    ) -> PositionSplitManager:
        """
        Builds grouped position splits from a model.

        Observed variables are grouped by inferred sample size along their split
        axes. One :class:`Split` is constructed for each group and immediately
        applied to the model's observed position.

        Parameters are the same as :meth:`SplitManager.from_model`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.experimental.optim import PositionSplitManager
        >>> x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        >>> y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        >>> model = lsl.Model([x, y])
        >>> split = PositionSplitManager.from_model(
        ...     model, position_keys=["x", "y"], share_validate=0.2
        ... )
        >>> split.ns
        (8, 5)
        >>> split.validate["x"].shape, split.validate["y"].shape
        ((1,), (1,))
        """
        splitter = SplitManager.from_model(
            model,
            position_keys=position_keys,
            share_validate=share_validate,
            share_test=share_test,
            axes=axes,
            default_axis=default_axis,
            shuffle=shuffle,
            seed=seed,
        )
        position = model.extract_position(splitter.position_keys)
        return splitter.split_position(position)

    @property
    def position_keys(self) -> list[str]:
        """
        Position keys claimed by all contained split objects.

        Examples
        --------
        >>> from liesel.experimental.optim import PositionSplitManager, PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> split = PositionSplitManager(
        ...     [PositionSplit(Position({"x": 1}), Position({}), Position({}), 1, 0, 0)]
        ... )
        >>> split.position_keys
        ['x']
        """
        keys: list[str] = []
        for split in self.splits:
            keys.extend(split.position_keys)
        return keys

    @property
    def train(self) -> Position:
        """
        Merged training position.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import PositionSplitManager, Split
        >>> pos = {"x": jnp.arange(3), "y": jnp.arange(4)}
        >>> manager = PositionSplitManager(
        ...     [
        ...         Split(["x"], n=3).split_position(pos),
        ...         Split(["y"], n=4).split_position(pos),
        ...     ]
        ... )
        >>> sorted(manager.train)
        ['x', 'y']
        """
        return _merge_positions([split.train for split in self.splits])

    @property
    def validate(self) -> Position:
        """
        Merged validation position.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import PositionSplitManager, Split
        >>> pos = {"x": jnp.arange(5), "y": jnp.arange(6)}
        >>> manager = PositionSplitManager(
        ...     [
        ...         Split(["x"], n=5, n_validate=1).split_position(pos),
        ...         Split(["y"], n=6, n_validate=1).split_position(pos),
        ...     ]
        ... )
        >>> sorted(manager.validate)
        ['x', 'y']
        """
        return _merge_positions([split.validate for split in self.splits])

    @property
    def test(self) -> Position:
        """
        Merged test position.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import PositionSplitManager, Split
        >>> pos = {"x": jnp.arange(5), "y": jnp.arange(6)}
        >>> manager = PositionSplitManager(
        ...     [
        ...         Split(["x"], n=5, n_test=1).split_position(pos),
        ...         Split(["y"], n=6, n_test=1).split_position(pos),
        ...     ]
        ... )
        >>> sorted(manager.test)
        ['x', 'y']
        """
        return _merge_positions([split.test for split in self.splits])

    @property
    def ns(self) -> tuple[int, ...]:
        """
        Total sample sizes for each contained split.

        Examples
        --------
        >>> from liesel.experimental.optim import PositionSplitManager, PositionSplit
        >>> from liesel.experimental.optim.types import Position
        >>> manager = PositionSplitManager(
        ...     [
        ...         PositionSplit(
        ...             Position({"x": 1}), Position({}), Position({}), 2, 0, 0
        ...         ),
        ...         PositionSplit(
        ...             Position({"y": 1}), Position({}), Position({}), 3, 0, 0
        ...         ),
        ...     ]
        ... )
        >>> manager.ns
        (2, 3)
        """
        return tuple(split.n for split in self.splits)

    @property
    def n_trains(self) -> tuple[int, ...]:
        """Training sample sizes for each contained split."""
        return tuple(split.n_train for split in self.splits)

    @property
    def n_validates(self) -> tuple[int, ...]:
        """Validation sample sizes for each contained split."""
        return tuple(split.n_validate for split in self.splits)

    @property
    def n_tests(self) -> tuple[int, ...]:
        """Test sample sizes for each contained split."""
        return tuple(split.n_test for split in self.splits)

    @property
    def n(self) -> int:
        """Common total sample size, available only when all branches agree."""
        return _common_value(self.ns, "n", "ns")

    @property
    def n_train(self) -> int:
        """Common training sample size, available only when all branches agree."""
        return _common_value(self.n_trains, "n_train", "n_trains")

    @property
    def n_validate(self) -> int:
        """Common validation sample size, available only when all branches agree."""
        return _common_value(self.n_validates, "n_validate", "n_validates")

    @property
    def n_test(self) -> int:
        """Common test sample size, available only when all branches agree."""
        return _common_value(self.n_tests, "n_test", "n_tests")

    @property
    def has_validation(self) -> bool:
        """
        Whether all child splits contain validation data.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import PositionSplitManager, Split
        >>> pos = {"x": jnp.arange(5), "y": jnp.arange(6)}
        >>> manager = PositionSplitManager(
        ...     [
        ...         Split(["x"], n=5, n_validate=1).split_position(pos),
        ...         Split(["y"], n=6, n_validate=1).split_position(pos),
        ...     ]
        ... )
        >>> manager.has_validation
        True
        """
        return self.splits[0].has_validation

    @property
    def has_test(self) -> bool:
        """Whether all child splits contain test data."""
        return self.splits[0].has_test

    @property
    def share_validates(self) -> tuple[float, ...]:
        """Validation shares for each contained split."""
        return tuple(split.share_validate for split in self.splits)

    @property
    def share_tests(self) -> tuple[float, ...]:
        """Test shares for each contained split."""
        return tuple(split.share_test for split in self.splits)

    @property
    def share_validate(self) -> float:
        """Common validation share, available only when all branches agree."""
        return _common_value(self.share_validates, "share_validate", "share_validates")

    @property
    def share_test(self) -> float:
        """Common test share, available only when all branches agree."""
        return _common_value(self.share_tests, "share_test", "share_tests")

    @property
    def validation_scales(self) -> tuple[float, ...]:
        """Validation likelihood scales for each contained split."""
        return tuple(split.scale_validate for split in self.splits)

    @property
    def scale_validate(self) -> float:
        """
        Common validation likelihood scale.

        Raises
        ------
        ValueError
            If child validation scales differ. Use :meth:`scaled_log_lik` for
            per-branch scaling with a Liesel :class:`.Model`.
        """
        return _common_value(
            self.validation_scales, "scale_validate", "validation_scales"
        )

    def _scales_for_part(
        self, part: Literal["train", "validate", "test"]
    ) -> tuple[float, ...]:
        return tuple(split._scale_for_part(part) for split in self.splits)

    def scaled_log_lik(
        self,
        model: Model | ModelInterface,
        model_state: ModelState,
        part: Literal["train", "validate", "test"] = "validate",
    ) -> jax.Array:
        """
        Returns the log likelihood with branch-specific split scaling.

        For a Liesel :class:`.Model`, each child split scales the observed
        likelihood terms belonging to its own ``position_keys``. For a generic
        :class:`.ModelInterface`, observed-variable decomposition is unavailable, so
        a common scalar scale is required.

        Parameters
        ----------
        model
            Liesel model or compatible model interface.
        model_state
            Updated model state containing log-likelihood values for ``part``.
        part
            Split part whose scale should be applied.

        Returns
        -------
        jax.Array
            Scaled log likelihood.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import tensorflow_probability.substrates.jax.distributions as tfd
        >>> from liesel.experimental.optim import PositionSplitManager, Split
        >>> y1 = lsl.Var.new_obs(
        ...     jnp.arange(10.0),
        ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        ...     name="y1",
        ... )
        >>> y2 = lsl.Var.new_obs(
        ...     jnp.arange(6.0),
        ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        ...     name="y2",
        ... )
        >>> model = lsl.Model([y1, y2])
        >>> pos = model.extract_position(["y1", "y2"])
        >>> split = PositionSplitManager(
        ...     [
        ...         Split(["y1"], n=10, n_validate=2).split_position(pos),
        ...         Split(["y2"], n=6, n_validate=1).split_position(pos),
        ...     ]
        ... )
        >>> state = model.update_state(split.validate, model.state)
        >>> manual = (
        ...     4.0 * state["y1_log_prob"].value.sum()
        ...     + 5.0 * state["y2_log_prob"].value.sum()
        ... )
        >>> bool(jnp.allclose(split.scaled_log_lik(model, state), manual))
        True
        """
        scales = self._scales_for_part(part)

        if isinstance(model, Model):
            groups = [
                (split.position_keys, scale)
                for split, scale in zip(self.splits, scales, strict=True)
            ]
            return scaled_liesel_log_lik(model, model_state, groups)

        try:
            scale = _common_value(scales, f"{part}_scale", f"{part}_scales")
        except ValueError as error:
            raise ValueError(
                "A generic ModelInterface cannot decompose observed log likelihoods "
                "by split branch. Use a Liesel Model, equal branch scales, or a "
                "custom validation loss."
            ) from error

        return scaled_common_log_lik(model_state, scale)

    def __repr__(self) -> str:
        name = type(self).__name__
        return (
            f"{name}(n={self.ns}, train={self.n_trains}, "
            f"validate={self.n_validates}, test={self.n_tests})"
        )


@dataclass
class SplitManager:
    """
    Wraps multiple :class:`Split` objects for multi-branch splitting.

    ``Split`` stays scalar: each instance assumes one sample size. ``SplitManager``
    coordinates several such scalar splitters and returns a
    :class:`PositionSplitManager` with merged train/validation/test positions.

    Parameters
    ----------
    splits
        Non-empty sequence of :class:`Split` objects. Their ``position_keys`` must
        not overlap. Either all children must define validation data or none may; the
        same rule applies to test data.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim import SplitManager, Split
    >>> manager = SplitManager(
    ...     [
    ...         Split(["x"], n=10, n_validate=2),
    ...         Split(["y"], n=6, n_validate=1),
    ...     ]
    ... )
    >>> split = manager.split_position({"x": jnp.arange(10), "y": jnp.arange(6)})
    >>> split.n_trains
    (8, 5)
    >>> split.validate["x"].tolist(), split.validate["y"].tolist()
    ([8, 9], [5])

    Automatically group model observations by sample size:

    >>> import liesel.model as lsl
    >>> x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
    >>> y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
    >>> model = lsl.Model([x, y])
    >>> manager = SplitManager.from_model(
    ...     model, position_keys=["x", "y"], share_validate=0.2
    ... )
    >>> manager.ns
    (8, 5)
    """

    splits: Sequence[Split]

    def __post_init__(self):
        self.splits = tuple(self.splits)

        if len(self.splits) == 0:
            raise ValueError("SplitManager requires at least one Split object.")

        _validate_unique_position_keys(
            [split.position_keys for split in self.splits], "splits"
        )
        self._validate_split_availability("validation")
        self._validate_split_availability("test")

    def _validate_split_availability(self, part: Literal["validation", "test"]):
        if part == "validation":
            values = [split.has_validation for split in self.splits]
        else:
            values = [split.has_test for split in self.splits]

        if len(set(values)) != 1:
            raise ValueError(
                f"All contained Split objects must either have {part} data or none "
                f"may have {part} data."
            )

    @classmethod
    def from_model(
        cls,
        model: Model,
        position_keys: Sequence[str] | None = None,
        share_validate: float = 0.0,
        share_test: float = 0.0,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
    ) -> SplitManager:
        """
        Builds a :class:`SplitManager` by grouping observed variables by size.

        Parameters
        ----------
        model
            Model containing the observed variables to split.
        position_keys
            Names of observed position entries to split. If ``None``, all observed
            variables in ``model`` are used.
        share_validate
            Share of observations assigned to validation in every child split.
        share_test
            Share of observations assigned to testing in every child split.
        axes
            Optional mapping from position key to split axis.
        default_axis
            Split axis for all position keys not listed in ``axes``.
        shuffle
            Whether each child split shuffles observations.
        seed
            Seed or JAX pseudo-random key used when ``shuffle=True``.

        Returns
        -------
        SplitManager
            Split manager with one child :class:`Split` per inferred sample size.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.experimental.optim import SplitManager
        >>> x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        >>> y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        >>> model = lsl.Model([x, y])
        >>> manager = SplitManager.from_model(
        ...     model, position_keys=["x", "y"], share_validate=0.2
        ... )
        >>> manager.position_keys
        ['x', 'y']
        >>> manager.n_validates
        (1, 1)
        """
        pos_keys = (
            list(position_keys) if position_keys is not None else list(model.observed)
        )
        groups = _position_key_groups_from_model(model, pos_keys, axes, default_axis)
        seeds = _child_seeds(seed, len(groups))
        splits = []

        for (n, keys), child_seed in zip(groups.items(), seeds, strict=True):
            splits.append(
                Split.from_share(
                    position_keys=keys,
                    n=n,
                    share_validate=share_validate,
                    share_test=share_test,
                    axes=axes,
                    default_axis=default_axis,
                    shuffle=shuffle,
                    seed=child_seed,
                )
            )

        return cls(splits)

    @property
    def position_keys(self) -> list[str]:
        """
        Position keys claimed by all contained splits.

        Examples
        --------
        >>> from liesel.experimental.optim import SplitManager, Split
        >>> SplitManager([Split(["x"], n=3), Split(["y"], n=4)]).position_keys
        ['x', 'y']
        """
        keys: list[str] = []
        for split in self.splits:
            keys.extend(split.position_keys)
        return keys

    @property
    def ns(self) -> tuple[int, ...]:
        """Total sample sizes for each contained split."""
        return tuple(split.n for split in self.splits)

    @property
    def n_trains(self) -> tuple[int, ...]:
        """Training sample sizes for each contained split."""
        return tuple(split._n_train for split in self.splits)

    @property
    def n_validates(self) -> tuple[int, ...]:
        """Validation sample sizes for each contained split."""
        return tuple(split.n_validate for split in self.splits)

    @property
    def n_tests(self) -> tuple[int, ...]:
        """Test sample sizes for each contained split."""
        return tuple(split.n_test for split in self.splits)

    @property
    def has_validation(self) -> bool:
        """Whether all child splits contain validation data."""
        return self.splits[0].has_validation

    @property
    def has_test(self) -> bool:
        """Whether all child splits contain test data."""
        return self.splits[0].has_test

    def split_position(self, position: Position) -> PositionSplitManager:
        """
        Splits a position with every child and merges the result.

        Parameters
        ----------
        position
            Mapping containing every key claimed by :attr:`position_keys`.

        Returns
        -------
        PositionSplitManager
            Merged split position object.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import SplitManager, Split
        >>> manager = SplitManager(
        ...     [
        ...         Split(["x"], n=4, n_validate=1, default_axis=1),
        ...         Split(["y"], n=6, n_validate=2),
        ...     ]
        ... )
        >>> split = manager.split_position(
        ...     {
        ...         "x": jnp.arange(8).reshape(2, 4),
        ...         "y": jnp.arange(6),
        ...     }
        ... )
        >>> split.train["x"].shape, split.validate["y"].shape
        ((2, 3), (2,))
        """
        return PositionSplitManager(
            [split.split_position(position) for split in self.splits]
        )

    def __repr__(self) -> str:
        name = type(self).__name__
        return (
            f"{name}(n={self.ns}, train={self.n_trains}, "
            f"validate={self.n_validates}, test={self.n_tests})"
        )


@dataclass
class Split:
    """
    Defines how observed position entries are split into train, validation, and test.

    ``Split`` stores a vector of observation indices. The first ``n_train`` indices
    become the training split, the next ``n_validate`` indices become the validation
    split, and the final ``n_test`` indices become the test split. If ``shuffle=True``,
    the index vector is permuted once during initialization.

    Parameters
    ----------
    position_keys
        Names of position entries that should be split.
    n
        Number of observations along each split axis. Must be positive.
    n_validate
        Number of validation observations.
    n_test
        Number of test observations.
    n_train
        Number of training observations. If left at ``None``, it is computed as
        ``n - n_validate - n_test``.
    axes
        Optional mapping from position key to split axis. Keys missing from this
        mapping use ``default_axis``.
    default_axis
        Split axis for all position keys not listed in ``axes``.
    shuffle
        Whether to shuffle observations during initialization.
    seed
        Seed or JAX pseudo-random key used when ``shuffle=True``. If ``None``, the
        current time is used.

    Attributes
    ----------
    indices
        Current observation order. Initialized as ``jnp.arange(n)`` and optionally
        shuffled in :meth:`__post_init__`.

    Raises
    ------
    ValueError
        If ``n`` is not positive, if any split size is negative, or if
        ``n_train + n_validate + n_test`` is not exactly equal to ``n``.

    Examples
    --------
    Split one vector without shuffling:

    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim import Split
    >>> splitter = Split(["x"], n=10, n_validate=2, n_test=1, shuffle=False)
    >>> splitter
    Split(train=7, validate=2, test=1)
    >>> splitter.indices_train.tolist()
    [0, 1, 2, 3, 4, 5, 6]
    >>> split = splitter.split_position({"x": jnp.arange(10)})
    >>> (
    ...     split.train["x"].tolist(),
    ...     split.validate["x"].tolist(),
    ...     split.test["x"].tolist(),
    ... )
    ([0, 1, 2, 3, 4, 5, 6], [7, 8], [9])

    Split different entries along different axes:

    >>> splitter = Split(
    ...     ["x", "y"],
    ...     n=4,
    ...     n_validate=1,
    ...     n_test=1,
    ...     axes={"x": 1},
    ...     shuffle=False,
    ... )
    >>> position = {
    ...     "x": jnp.arange(8).reshape(2, 4),
    ...     "y": jnp.arange(12).reshape(4, 3),
    ... }
    >>> split = splitter.split_position(position)
    >>> split.train["x"].tolist()
    [[0, 1], [4, 5]]
    >>> split.validate["y"].tolist()
    [[6, 7, 8]]
    """

    position_keys: Sequence[str]
    n: int
    n_validate: int = 0
    n_test: int = 0
    n_train: int | None = None
    axes: dict[str, int] | None = field(default_factory=dict)
    default_axis: int = 0
    shuffle: bool = False
    seed: jax.Array | int | None = None

    def __post_init__(self):
        if self.axes is None:
            self.axes = {}

        if self.n <= 0:
            raise ValueError(f"{self.n=} is <= 0, which is not allowed.")

        if self.n_train is None:
            self.n_train = self.n - self.n_validate - self.n_test

        assert self.n_train is not None
        self.indices = jnp.arange(self.n)

        if self.n_train < 0 or self.n_validate < 0 or self.n_test < 0:
            raise ValueError(
                f"Split sizes must be non-negative, but got {self.n_train=}, "
                f"{self.n_validate=}, and {self.n_test=}."
            )

        n_split = self.n_train + self.n_validate + self.n_test
        if n_split != self.n:
            raise ValueError(
                f"The given {self.n_train=}, {self.n_validate=}, and {self.n_test=} "
                f"sum to {n_split}, but must sum exactly to {self.n=}."
            )

        if self.shuffle:
            if isinstance(self.seed, jax.Array):
                key = self.seed
            else:
                seed = int(time.time()) if self.seed is None else self.seed
                key = jax.random.key(seed)
            self.indices = self.permute_indices(key)

    @property
    def _n_train(self) -> int:
        assert self.n_train is not None
        return self.n_train

    @property
    def has_validation(self) -> bool:
        """
        Whether this splitter assigns observations to validation.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=10, n_validate=2).has_validation
        True
        """
        return self.n_validate > 0

    @property
    def has_test(self) -> bool:
        """
        Whether this splitter assigns observations to testing.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=10).has_test
        False
        """
        return self.n_test > 0

    @property
    def share_validate(self) -> float:
        """
        Share of observations assigned to validation.

        Returns
        -------
        float
            The ratio ``n_validate / n``.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=10, n_validate=2).share_validate
        0.2
        """
        return self.n_validate / self.n

    @property
    def share_test(self) -> float:
        """
        Share of observations assigned to testing.

        Returns
        -------
        float
            The ratio ``n_test / n``.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=10, n_test=3).share_test
        0.3
        """
        return self.n_test / self.n

    @classmethod
    def from_share(
        cls,
        position_keys: Sequence[str],
        n: int,
        share_validate: float = 0.0,
        share_test: float = 0.0,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
    ) -> Split:
        """
        Builds a :class:`Split` from validation and test proportions.

        The number of validation and test observations is computed with
        ``int(n * share)``. Any fractional remainder is assigned to the training
        split, so the resulting split sizes always sum to ``n``.

        Parameters
        ----------
        position_keys
            Names of position entries that should be split.
        n
            Number of observations along each split axis.
        share_validate
            Share of observations assigned to validation.
        share_test
            Share of observations assigned to testing.
        axes
            Optional mapping from position key to split axis.
        default_axis
            Split axis for all position keys not listed in ``axes``.
        shuffle
            Whether to shuffle observations during initialization.
        seed
            Seed or JAX pseudo-random key used when ``shuffle=True``.

        Returns
        -------
        Split
            Splitter with integer split sizes derived from the shares.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> splitter = Split.from_share(
        ...     ["x"],
        ...     n=10,
        ...     share_validate=0.25,
        ...     share_test=0.25,
        ... )
        >>> splitter.n_train, splitter.n_validate, splitter.n_test
        (6, 2, 2)
        >>> splitter.share_validate, splitter.share_test
        (0.2, 0.2)
        """
        if n <= 0:
            raise ValueError(f"{n=} is <= 0, which is not allowed.")

        if share_validate < 0.0 or share_test < 0.0:
            raise ValueError(
                f"Shares must be non-negative, but got {share_validate=} "
                f"and {share_test=}."
            )

        share_observed = share_validate + share_test
        if share_observed > 1.0:
            raise ValueError(
                f"Validation and test shares sum to {share_observed}, which is > 1.0."
            )

        n_validate = int(n * share_validate)
        n_test = int(n * share_test)
        n_train = n - n_validate - n_test

        return cls(
            position_keys=position_keys,
            n=n,
            n_validate=n_validate,
            n_test=n_test,
            n_train=n_train,
            axes=axes,
            default_axis=default_axis,
            shuffle=shuffle,
            seed=seed,
        )

    def permute_indices(self, key: jax.Array) -> jax.Array:
        """
        Returns a random permutation of the current index vector.

        This method does not mutate :attr:`indices`.

        Parameters
        ----------
        key
            JAX pseudo-random key.

        Returns
        -------
        jax.Array
            Permuted copy of :attr:`indices`.

        Examples
        --------
        >>> import jax
        >>> from liesel.experimental.optim import Split
        >>> splitter = Split(["x"], n=5)
        >>> permuted = splitter.permute_indices(jax.random.key(0))
        >>> sorted(permuted.tolist())
        [0, 1, 2, 3, 4]
        >>> splitter.indices.tolist()
        [0, 1, 2, 3, 4]
        """
        return jax.random.permutation(key, self.indices)

    @property
    def indices_train(self) -> jax.Array:
        """
        Observation indices for the training split.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=6, n_validate=2, n_test=1).indices_train.tolist()
        [0, 1, 2]
        """
        return self.indices[: self._n_train]

    @property
    def indices_validate(self) -> jax.Array:
        """
        Observation indices for the validation split.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=6, n_validate=2, n_test=1).indices_validate.tolist()
        [3, 4]
        """
        start = self._n_train
        end = self._n_train + self.n_validate
        return self.indices[start:end]

    @property
    def indices_test(self) -> jax.Array:
        """
        Observation indices for the test split.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=6, n_validate=2, n_test=1).indices_test.tolist()
        [5]
        """
        start = self._n_train + self.n_validate
        end = self._n_train + self.n_validate + self.n_test
        return self.indices[start:end]

    def split_position(self, position: Position) -> PositionSplit:
        """
        Splits position entries into train, validation, and test positions.

        Parameters
        ----------
        position
            Mapping containing every key in :attr:`position_keys`. Each selected
            entry must have length ``n`` along its split axis.

        Returns
        -------
        PositionSplit
            Split position entries and split sizes.

        Raises
        ------
        ValueError
            If a selected position entry has an incompatible length along its split
            axis.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import Split
        >>> splitter = Split(["x"], n=5, n_validate=1, n_test=1)
        >>> split = splitter.split_position({"x": jnp.arange(5)})
        >>> split.train["x"].tolist()
        [0, 1, 2]
        >>> split.validate["x"].tolist(), split.test["x"].tolist()
        ([3], [4])

        Use ``axes`` to split an array along a non-leading axis:

        >>> splitter = Split(["x"], n=4, n_validate=1, axes={"x": 1})
        >>> split = splitter.split_position({"x": jnp.arange(8).reshape(2, 4)})
        >>> split.train["x"].tolist()
        [[0, 1, 2], [4, 5, 6]]
        >>> split.validate["x"].tolist()
        [[3], [7]]
        """
        train_position = {}
        validation_position = {}
        test_position = {}

        assert self.axes is not None
        for key in self.position_keys:
            axis = self.axes.get(key, self.default_axis)

            n_this_key = jnp.shape(position[key])[axis]
            if not jnp.shape(position[key])[axis] == self.n:
                raise ValueError(
                    f"{key} has n={n_this_key}, which is incompatible with the "
                    f"given sample size of n={self.n}."
                )

            train_values = jnp.take(position[key], self.indices_train, axis=axis)
            validation_values = jnp.take(
                position[key], self.indices_validate, axis=axis
            )
            test_values = jnp.take(position[key], self.indices_test, axis=axis)

            train_position[key] = train_values
            validation_position[key] = validation_values
            test_position[key] = test_values

        split = PositionSplit(
            train=Position(train_position),
            validate=Position(validation_position),
            test=Position(test_position),
            n_train=self._n_train,
            n_validate=self.n_validate,
            n_test=self.n_test,
        )
        return split

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validate={self.n_validate}, test={self.n_test})"
        )
        return out
