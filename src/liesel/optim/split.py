"""Train, validation, and test splitting utilities for optimizer positions."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp

from ..model import Model
from ._log_lik import scaled_common_log_lik, scaled_liesel_log_lik
from ._model_utils import position_key_groups_from_model
from .types import Array, ModelInterface, ModelState, Position

SplitPart = Literal["train", "validate", "test"]
SampleSizes = Mapping[SplitPart, int | float]
_SPLIT_PARTS: tuple[SplitPart, ...] = ("train", "validate", "test")


def _merge_positions(positions: Sequence[Position]) -> Position:
    """
    Merge split position dictionaries and reject duplicate keys.

    Examples
    --------
    >>> from liesel.optim.split import _merge_positions
    >>> from liesel.optim.types import Position
    >>> merged = _merge_positions([Position({"x": 1}), Position({"y": 2})])
    >>> sorted(merged.items())
    [('x', 1), ('y', 2)]
    """
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
    """
    Return a scalar value only when all branch values agree.

    ``name`` and ``plural_name`` are used to build the error message when branch
    values differ.

    Examples
    --------
    >>> from liesel.optim.split import _common_value
    >>> _common_value([3, 3, 3], "axis_size", "axis_sizes")
    3
    """
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
    """
    Validate that position keys are claimed by at most one group.

    Parameters
    ----------
    groups
        Position-key groups to compare.
    owner
        Name used in the duplicate-key error message.

    Examples
    --------
    >>> from liesel.optim.split import _validate_unique_position_keys
    >>> _validate_unique_position_keys([["x"], ["y"]], "splits") is None
    True
    """
    counts: dict[str, int] = {}

    for group in groups:
        for key in group:
            counts[key] = counts.get(key, 0) + 1

    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Position keys claimed by multiple {owner}: {duplicates}")


def _child_seeds(
    seed: jax.Array | int | None, n_children: int
) -> tuple[jax.Array | int | None, ...]:
    """
    Split one seed into deterministic child keys.

    ``None`` is converted to a time-based key before splitting.

    Examples
    --------
    >>> from liesel.optim.split import _child_seeds
    >>> len(_child_seeds(0, 2))
    2
    """
    if seed is None:
        key = jax.random.key(int(time.time()))
    else:
        key = seed if isinstance(seed, jax.Array) else jax.random.key(seed)

    return tuple(jax.random.split(key, n_children))


def _validate_child_split_availability(
    splits: Sequence[Split],
    validate_axis_share: float,
    test_axis_share: float,
) -> None:
    """
    Validate that requested validation/test shares produce data for every child.

    This catches small branches where ``int(axis_size * share)`` would round to
    zero while larger branches still receive validation or test observations.

    Examples
    --------
    >>> from liesel.optim import Split
    >>> from liesel.optim.split import _validate_child_split_availability
    >>> splits = [
    ...     Split(["x"], axis_size=5, validate_axis_size=1),
    ...     Split(["y"], axis_size=4, validate_axis_size=1),
    ... ]
    >>> _validate_child_split_availability(splits, 0.2, 0.0) is None
    True
    """
    if validate_axis_share > 0.0:
        validate_axis_sizes = [split.validate_axis_size for split in splits]
        if any(size == 0 for size in validate_axis_sizes) and any(
            size > 0 for size in validate_axis_sizes
        ):
            axis_sizes_with_zero = [
                split.axis_size for split in splits if split.validate_axis_size == 0
            ]
            raise ValueError(
                f"{validate_axis_share=} produced zero validation observations for "
                f"axis sizes {axis_sizes_with_zero}, while other split groups "
                "received validation data. Increase validate_axis_share, set "
                "validate_axis_share=0.0, or construct SplitManager manually."
            )

    if test_axis_share > 0.0:
        test_axis_sizes = [split.test_axis_size for split in splits]
        if any(size == 0 for size in test_axis_sizes) and any(
            size > 0 for size in test_axis_sizes
        ):
            axis_sizes_with_zero = [
                split.axis_size for split in splits if split.test_axis_size == 0
            ]
            raise ValueError(
                f"{test_axis_share=} produced zero test observations for axis sizes "
                f"{axis_sizes_with_zero}, while other split groups received test data. "
                "Increase test_axis_share, set test_axis_share=0.0, or construct "
                "SplitManager manually."
            )


def _position_size_is_compatible(value: Array, axis_size: int) -> bool:
    """
    Check whether a value can represent a split part of ``axis_size``.

    Scalar values are compatible only with ``axis_size=1``. Non-scalar values
    are compatible if one of their axes has the declared size.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.optim.split import _position_size_is_compatible
    >>> _position_size_is_compatible(jnp.ones((2, 3)), 3)
    True
    >>> _position_size_is_compatible(1.0, 2)
    False
    """
    shape = jnp.shape(value)
    if len(shape) == 0:
        return axis_size == 1
    return axis_size in shape


def _normalize_sample_sizes(
    sample_sizes: SampleSizes | None,
) -> dict[SplitPart, float] | None:
    """
    Validate sample-size mappings and convert their values to floats.

    Examples
    --------
    >>> from liesel.optim.split import _normalize_sample_sizes
    >>> _normalize_sample_sizes({"train": 10, "validate": 2})
    {'train': 10.0, 'validate': 2.0}
    >>> _normalize_sample_sizes(None) is None
    True
    """
    if sample_sizes is None:
        return None

    normalized: dict[SplitPart, float] = {}
    for part, size in sample_sizes.items():
        if part not in _SPLIT_PARTS:
            raise ValueError(
                "sample_sizes keys must be 'train', 'validate', or 'test', "
                f"but got {part!r}."
            )

        size_float = float(size)
        if not math.isfinite(size_float) or size_float < 0.0:
            raise ValueError(
                "sample_sizes values must be finite and non-negative, "
                f"but got {size!r} for {part!r}."
            )

        normalized[part] = size_float

    return normalized


def _count_likelihood_contributions(value) -> int:
    """
    Count scalar likelihood contributions in a log-probability value.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.optim.split import _count_likelihood_contributions
    >>> _count_likelihood_contributions(jnp.ones((2, 3)))
    6
    >>> _count_likelihood_contributions(0.0)
    1
    """
    shape = jnp.shape(value)
    if len(shape) == 0:
        return 1

    return math.prod(shape)


def _observed_dist_infos(
    model: Model, position_keys: Sequence[str]
) -> list[tuple[str, str, bool]]:
    """
    Return observed-variable names, log-prob node names, and ``per_obs`` flags.

    Only observed variables whose variable name or value-node name appears in
    ``position_keys`` are included.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.optim.split import _observed_dist_infos
    >>> y = lsl.Var.new_obs(
    ...     jnp.arange(2.0),
    ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    ...     name="y",
    ... )
    >>> model = lsl.Model([y])
    >>> _observed_dist_infos(model, ["y"])
    [('y', 'y_log_prob', True)]
    """
    keys = set(position_keys)
    infos: list[tuple[str, str, bool]] = []

    for var in model.observed.values():
        if var.dist_node is None:
            continue

        if var.name in keys or var.value_node.name in keys:
            infos.append((var.name, var.dist_node.name, var.dist_node.per_obs))

    return infos


def _has_custom_model_log_lik(model: Model) -> bool:
    """
    Check whether a model uses a custom aggregate log-likelihood node.

    The default model log likelihood depends directly on the observed log-prob
    nodes. A different dependency structure indicates custom aggregation, for
    which automatic split sample-size inference is ambiguous.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.optim.split import _has_custom_model_log_lik
    >>> y = lsl.Var.new_obs(
    ...     jnp.arange(2.0),
    ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    ...     name="y",
    ... )
    >>> _has_custom_model_log_lik(lsl.Model([y]))
    False
    """
    model_log_lik = model.nodes.get("_model_log_lik")
    if model_log_lik is None:
        return False

    direct_inputs = model_log_lik.inputs + tuple(model_log_lik.kwinputs.values())
    direct_input_names = {node.name for node in direct_inputs}
    observed_log_prob_names = {
        var.dist_node.name
        for var in model.observed.values()
        if var.dist_node is not None
    }

    return direct_input_names != observed_log_prob_names


def _infer_sample_size_for_part(
    model: Model,
    model_state: ModelState,
    split: PositionSplit,
    part: SplitPart,
) -> int:
    """
    Infer one split part's effective sample size from pointwise log likelihoods.

    If a split group has no observed likelihood nodes, the declared split-axis
    size is used as a fallback.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.optim import Split
    >>> from liesel.optim.split import _infer_sample_size_for_part
    >>> y = lsl.Var.new_obs(
    ...     jnp.arange(4.0),
    ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    ...     name="y",
    ... )
    >>> model = lsl.Model([y])
    >>> split = Split(["y"], axis_size=4, validate_axis_size=1).split_position(
    ...     model.extract_position(["y"])
    ... )
    >>> state = model.update_state(split.validate, model.state)
    >>> _infer_sample_size_for_part(model, state, split, "validate")
    1
    """
    infos = _observed_dist_infos(model, split.position_keys)
    sizes: dict[str, int] = {}

    for var_name, node_name, per_obs in infos:
        if not per_obs:
            raise ValueError(
                "Cannot infer sample sizes because "
                f"{var_name!r} has Var.dist_node.per_obs=False. Set "
                "infer_sample_sizes=False or provide sample_sizes manually."
            )

        sizes[var_name] = _count_likelihood_contributions(model_state[node_name].value)

    if not sizes:
        return split._axis_size_for_part(part)

    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(
            "Cannot infer a scalar sample size because observed variables in one "
            f"PositionSplit imply incompatible pointwise sample sizes: {sizes}. "
            "Use PositionSplitManager, provide sample_sizes manually, or set "
            "infer_sample_sizes=False."
        )

    return unique_sizes.pop()


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
    train_axis_size
        Number of training observations.
    validate_axis_size
        Number of validation observations.
    test_axis_size
        Number of test observations.
    sample_sizes
        Optional effective sample sizes for train, validation, and test scaling.
        If omitted, split-axis counts are used.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.optim import PositionSplit
    >>> from liesel.optim.types import Position
    >>> split = PositionSplit(
    ...     train=Position({"x": jnp.arange(3)}),
    ...     validate=Position({"x": jnp.arange(3, 5)}),
    ...     test=Position({"x": jnp.arange(5, 6)}),
    ...     train_axis_size=3,
    ...     validate_axis_size=2,
    ...     test_axis_size=1,
    ... )
    >>> split
    PositionSplit(train=3, validate=2, test=1)
    >>> split.validate["x"].tolist()
    [3, 4]
    """

    train: Position
    validate: Position
    test: Position

    train_axis_size: int
    validate_axis_size: int
    test_axis_size: int
    sample_sizes: SampleSizes | None = None

    def __post_init__(self):
        self.set_sample_sizes(self.sample_sizes)

        if (
            self.train_axis_size < 0
            or self.validate_axis_size < 0
            or self.test_axis_size < 0
        ):
            raise ValueError(
                f"Split sizes must be non-negative, but got {self.train_axis_size=}, "
                f"{self.validate_axis_size=}, and {self.test_axis_size=}."
            )

        if self.axis_size <= 0:
            raise ValueError(
                "PositionSplit must contain at least one observation, but got "
                f"{self.axis_size=}."
            )

        expected_keys = set(self.position_keys)
        if not expected_keys:
            for part, n_part in (
                ("train", self.train_axis_size),
                ("validate", self.validate_axis_size),
                ("test", self.test_axis_size),
            ):
                if n_part > 0:
                    raise ValueError(
                        f"PositionSplit.{part} must contain position entries when "
                        f"{part}_axis_size > 0."
                    )

            raise ValueError("PositionSplit must contain position entries.")

        for part, position, n_part in (
            ("train", self.train, self.train_axis_size),
            ("validate", self.validate, self.validate_axis_size),
            ("test", self.test, self.test_axis_size),
        ):
            keys = set(position)
            if n_part > 0 and not keys:
                raise ValueError(
                    f"PositionSplit.{part} must contain position entries when "
                    f"{part}_axis_size > 0."
                )

            if keys and keys != expected_keys:
                raise ValueError(
                    f"PositionSplit.{part} must contain the same position keys as "
                    "the other non-empty split parts."
                )

            for key, value in position.items():
                if not _position_size_is_compatible(value, n_part):
                    raise ValueError(
                        f"PositionSplit.{part}[{key!r}] has shape {jnp.shape(value)}, "
                        f"which is incompatible with the declared split size "
                        f"{n_part}."
                    )

    @property
    def position_keys(self) -> list[str]:
        """
        Position keys contained in this split.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplit
        >>> from liesel.optim.types import Position
        >>> split = PositionSplit(
        ...     Position({"x": jnp.arange(2)}),
        ...     Position({}),
        ...     Position({}),
        ...     train_axis_size=2,
        ...     validate_axis_size=0,
        ...     test_axis_size=0,
        ... )
        >>> split.position_keys
        ['x']
        """
        for position in (self.train, self.validate, self.test):
            if position:
                return list(position)

        return []

    def set_sample_sizes(self, sample_sizes: SampleSizes | None) -> PositionSplit:
        """
        Set effective likelihood sample sizes on this split.

        The supplied mapping is normalized to floats. The method mutates and
        returns ``self``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplit
        >>> from liesel.optim.types import Position
        >>> split = PositionSplit(
        ...     Position({"x": jnp.arange(3)}),
        ...     Position({"x": jnp.arange(1)}),
        ...     Position({}),
        ...     train_axis_size=3,
        ...     validate_axis_size=1,
        ...     test_axis_size=0,
        ... )
        >>> split.set_sample_sizes({"train": 12, "validate": 3}) is split
        True
        >>> split.sample_sizes
        {'train': 12.0, 'validate': 3.0}
        """
        normalized = _normalize_sample_sizes(sample_sizes)

        for part, n_part in (
            ("train", self.train_axis_size),
            ("validate", self.validate_axis_size),
            ("test", self.test_axis_size),
        ):
            if (
                n_part > 0
                and normalized is not None
                and normalized.get(part, n_part) == 0.0
            ):
                raise ValueError(
                    f"sample_sizes[{part!r}] must be positive when "
                    f"{part}_axis_size > 0."
                )

        self.sample_sizes = normalized
        return self

    def add_inferred_sample_sizes_from_model(self, model: Model) -> PositionSplit:
        """
        Infer effective sample sizes from ``model`` and attach them to this split.

        Empty validation or test parts are omitted from the inferred mapping. The
        method mutates and returns ``self``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import tensorflow_probability.substrates.jax.distributions as tfd
        >>> from liesel.optim import Split
        >>> y = lsl.Var.new_obs(
        ...     jnp.arange(4.0),
        ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        ...     name="y",
        ... )
        >>> model = lsl.Model([y])
        >>> split = Split(["y"], axis_size=4, validate_axis_size=1).split_position(
        ...     model.extract_position(["y"])
        ... )
        >>> split.add_inferred_sample_sizes_from_model(model) is split
        True
        >>> split.sample_sizes
        {'train': 3.0, 'validate': 1.0}
        """
        if _has_custom_model_log_lik(model):
            raise ValueError(
                "Cannot infer sample sizes for a model with a custom log_lik_node. "
                "Set infer_sample_sizes=False or provide sample_sizes manually."
            )

        sizes: dict[SplitPart, int] = {}
        for part, position, n_part in (
            ("train", self.train, self.train_axis_size),
            ("validate", self.validate, self.validate_axis_size),
            ("test", self.test, self.test_axis_size),
        ):
            if n_part == 0:
                continue

            state = model.update_state(position, model.state)
            sizes[part] = _infer_sample_size_for_part(model, state, self, part)

        return self.set_sample_sizes(sizes)

    @property
    def axis_size(self) -> int:
        """
        Total axis size represented by the split.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplit
        >>> from liesel.optim.types import Position
        >>> train = Position({"x": jnp.arange(7)})
        >>> validate = Position({"x": jnp.arange(2)})
        >>> test = Position({"x": jnp.arange(1)})
        >>> PositionSplit(train, validate, test, 7, 2, 1).axis_size
        10
        """
        return self.train_axis_size + self.validate_axis_size + self.test_axis_size

    @property
    def has_validation(self) -> bool:
        """
        Whether this split contains validation observations.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplit
        >>> from liesel.optim.types import Position
        >>> train = Position({"x": jnp.arange(7)})
        >>> validate = Position({"x": jnp.arange(2)})
        >>> test = Position({"x": jnp.arange(1)})
        >>> PositionSplit(train, validate, test, 7, 2, 1).has_validation
        True
        """
        return self.validate_axis_size > 0

    @property
    def has_test(self) -> bool:
        """
        Whether this split contains test observations.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplit
        >>> from liesel.optim.types import Position
        >>> train = Position({"x": jnp.arange(7)})
        >>> PositionSplit(train, Position({}), Position({}), 7, 0, 0).has_test
        False
        """
        return self.test_axis_size > 0

    @property
    def validate_axis_share(self) -> float:
        """
        Share of observations assigned to validation.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplit
        >>> from liesel.optim.types import Position
        >>> train = Position({"x": jnp.arange(7)})
        >>> validate = Position({"x": jnp.arange(2)})
        >>> test = Position({"x": jnp.arange(1)})
        >>> PositionSplit(train, validate, test, 7, 2, 1).validate_axis_share
        0.2
        """
        return self.validate_axis_size / self.axis_size

    @property
    def test_axis_share(self) -> float:
        """
        Share of observations assigned to testing.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplit
        >>> from liesel.optim.types import Position
        >>> train = Position({"x": jnp.arange(7)})
        >>> validate = Position({"x": jnp.arange(2)})
        >>> test = Position({"x": jnp.arange(1)})
        >>> PositionSplit(train, validate, test, 7, 2, 1).test_axis_share
        0.1
        """
        return self.test_axis_size / self.axis_size

    @property
    def validate_sample_scale(self) -> float:
        """
        Likelihood scale for validation data.

        Returns ``1.0`` when no validation split is present.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplit
        >>> from liesel.optim.types import Position
        >>> train = Position({"x": jnp.arange(8)})
        >>> validate = Position({"x": jnp.arange(2)})
        >>> PositionSplit(train, validate, Position({}), 8, 2, 0).validate_sample_scale
        4.0
        """
        return self.sample_scale("validate")

    def _axis_size_for_part(self, part: SplitPart) -> int:
        if part == "train":
            return self.train_axis_size

        if part == "validate":
            return self.validate_axis_size

        if part == "test":
            return self.test_axis_size

        raise ValueError(f"Unrecognized {part=}.")

    def sample_size(self, part: SplitPart) -> float:
        """
        Effective likelihood sample size for one split part.

        Directly constructed splits fall back to the corresponding axis size when
        explicit or inferred sample sizes are unavailable.
        """
        if self.sample_sizes is not None and part in self.sample_sizes:
            return self.sample_sizes[part]

        return float(self._axis_size_for_part(part))

    @property
    def train_sample_size(self) -> float:
        """Effective training likelihood sample size."""
        return self.sample_size("train")

    @property
    def validate_sample_size(self) -> float:
        """Effective validation likelihood sample size."""
        return self.sample_size("validate")

    @property
    def test_sample_size(self) -> float:
        """Effective test likelihood sample size."""
        return self.sample_size("test")

    def sample_scale(self, part: SplitPart) -> float:
        """Scale a part likelihood to the training sample size."""
        if part == "train":
            return 1.0

        if part == "validate":
            if not self.has_validation:
                return 1.0
            return self.train_sample_size / self.validate_sample_size

        if part == "test":
            if not self.has_test:
                return 1.0
            return self.train_sample_size / self.test_sample_size

        raise ValueError(f"Unrecognized {part=}.")

    def scaled_log_lik(
        self,
        model: Model | ModelInterface,
        model_state: ModelState,
        part: SplitPart = "validate",
    ) -> jax.Array:
        """
        Returns the log likelihood scaled for one split part.

        The result is placed on the same effective likelihood scale as the
        training split. Validation and test likelihoods are therefore multiplied
        by ``train_sample_size / sample_size(part)``.

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
            ``train_axis_size / validate_axis_size``.

        Returns
        -------
        jax.Array
            Scaled log likelihood.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> import tensorflow_probability.substrates.jax.distributions as tfd
        >>> from liesel.optim import Split
        >>> y = lsl.Var.new_obs(
        ...     jnp.arange(10.0),
        ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        ...     name="y",
        ... )
        >>> model = lsl.Model([y])
        >>> split = Split(["y"], axis_size=10, validate_axis_size=2).split_position(
        ...     model.extract_position(["y"])
        ... )
        >>> state = model.update_state(split.validate, model.state)
        >>> bool(
        ...     jnp.allclose(
        ...         split.scaled_log_lik(model, state),
        ...         split.validate_sample_scale * state["_model_log_lik"].value,
        ...     )
        ... )
        True
        """
        scale = self.sample_scale(part)

        if isinstance(model, Model):
            # A scalar split scale applies only to likelihood nodes covered by this
            # split. Other observed model branches may still be evaluated on full
            # data and must remain unscaled; scaling "_model_log_lik" would scale
            # them too.
            return scaled_liesel_log_lik(
                model=model,
                model_state=model_state,
                groups=[(self.position_keys, scale)],
            )

        return scaled_common_log_lik(model_state, scale)

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.train_axis_size}, "
            f"validate={self.validate_axis_size}, test={self.test_axis_size})"
        )
        return out

    @staticmethod
    def from_model(
        model: Model,
        position_keys: Sequence[str] | None = None,
        axis_size: int | None = None,
        validate_axis_share: float = 0.0,
        test_axis_share: float = 0.0,
        split_axes: dict[str, int] | None = None,
        default_split_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
        multi_size: Literal["error", "manager"] = "error",
        sample_sizes: SampleSizes | None = None,
        infer_sample_sizes: bool = True,
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
        axis_size
            Number of observations along the split axis. If ``None``, the number is
            guessed from ``model`` along ``default_split_axis``.
        validate_axis_share
            Share of observations assigned to the validation split.
        test_axis_share
            Share of observations assigned to the test split.
        split_axes
            Optional mapping from position key to split axis. Keys missing from this
            mapping use ``default_split_axis``.
        default_split_axis
            Split axis for all position keys not listed in ``split_axes``.
        shuffle
            Whether observations are shuffled before splitting.
        seed
            Seed or JAX pseudo-random key used when ``shuffle=True``.
        multi_size
            How to handle observed variables with different inferred axis sizes.
            The default ``"error"`` keeps :class:`PositionSplit` scalar and raises
            a helpful error. Use ``"manager"`` to return a
            :class:`PositionSplitManager` when multiple axis sizes are detected.
        sample_sizes
            Optional effective sample sizes for train, validation, and test
            scaling. If supplied, these values are used instead of automatic
            inference.
        infer_sample_sizes
            Whether to infer effective sample sizes from pointwise observed
            log-probability arrays.

        Returns
        -------
        PositionSplit or PositionSplitManager
            Split observed position entries extracted from ``model``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.optim import PositionSplit
        >>> y = lsl.Var.new_obs(jnp.arange(10.0), name="y")
        >>> model = lsl.Model([y])
        >>> split = PositionSplit.from_model(
        ...     model,
        ...     position_keys=["y"],
        ...     validate_axis_share=0.2,
        ...     test_axis_share=0.1,
        ... )
        >>> split.train_axis_size, split.validate_axis_size, split.test_axis_size
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
        ...     validate_axis_share=0.2,
        ...     multi_size="manager",
        ... )
        >>> type(managed).__name__
        'PositionSplitManager'
        """
        if multi_size not in ("error", "manager"):
            raise ValueError("multi_size must be 'error' or 'manager'.")

        pos_keys = (
            list(position_keys) if position_keys is not None else list(model.observed)
        )
        if not pos_keys:
            raise ValueError(
                "PositionSplit.from_model() requires at least one position key."
            )

        groups = position_key_groups_from_model(
            model, pos_keys, split_axes, default_split_axis
        )
        if len(groups) > 1 and multi_size == "manager":
            if axis_size is not None:
                raise ValueError(
                    "A single axis_size value cannot configure multiple axis-size "
                    "groups. Omit axis_size when using multi_size='manager'."
                )

            return PositionSplitManager.from_model(
                model,
                position_keys=pos_keys,
                validate_axis_share=validate_axis_share,
                test_axis_share=test_axis_share,
                split_axes=split_axes,
                default_split_axis=default_split_axis,
                shuffle=shuffle,
                seed=seed,
                sample_sizes=sample_sizes,
                infer_sample_sizes=infer_sample_sizes,
            )

        if len(groups) > 1:
            raise ValueError(
                "PositionSplit.from_model() found observed variables with different "
                f"axis sizes: {groups}. Use "
                "PositionSplit.from_model(..., multi_size='manager') or "
                "PositionSplitManager.from_model(...)."
            )

        if axis_size is None:
            axis_size = next(iter(groups))

        splitter = Split.from_axis_shares(
            position_keys=pos_keys,
            axis_size=axis_size,
            validate_axis_share=validate_axis_share,
            test_axis_share=test_axis_share,
            split_axes=split_axes,
            default_split_axis=default_split_axis,
            shuffle=shuffle,
            seed=seed,
            sample_sizes=sample_sizes,
        )

        pos = model.extract_position(pos_keys)
        split = splitter.split_position(pos)
        if infer_sample_sizes and sample_sizes is None:
            split.add_inferred_sample_sizes_from_model(model)

        return split


@dataclass
class PositionSplitManager:
    """
    Coordinates multiple :class:`PositionSplit` objects as one split interface.

    ``PositionSplitManager`` is the split-side counterpart to
    :class:`.BatchManager`. It is useful when a model has observed branches with
    different axis sizes. Each child :class:`PositionSplit` stores the split data
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
    Branch-specific sizes are available as :attr:`axis_sizes`,
    :attr:`train_axis_sizes`, :attr:`validate_axis_sizes`, and
    :attr:`test_axis_sizes`. Branch-specific likelihood sizes are available as
    :attr:`sample_sizes`, :attr:`train_sample_sizes`,
    :attr:`validate_sample_sizes`, and :attr:`test_sample_sizes`. Scalar aliases
    such as :attr:`train_axis_size`, :attr:`validate_axis_share`, and
    :attr:`validate_sample_scale` are available only when all children have the
    same value. Use :meth:`sample_size` for the total likelihood sample size of a
    split part across all branches.

    Examples
    --------
    Merge two branches with different axis sizes:

    >>> import jax.numpy as jnp
    >>> from liesel.optim import PositionSplitManager, Split
    >>> position = {"x": jnp.arange(10), "y": jnp.arange(6)}
    >>> split_x = Split(["x"], axis_size=10, validate_axis_size=2).split_position(
    ...     position
    ... )
    >>> split_y = Split(["y"], axis_size=6, validate_axis_size=1).split_position(
    ...     position
    ... )
    >>> manager = PositionSplitManager([split_x, split_y])
    >>> manager.position_keys
    ['x', 'y']
    >>> manager.train_axis_sizes
    (8, 5)
    >>> manager.train["x"].shape, manager.train["y"].shape
    ((8,), (5,))

    Unequal scalar aliases raise and direct users to the plural property:

    >>> try:
    ...     manager.train_axis_size
    ... except ValueError as error:
    ...     print("train_axis_sizes" in str(error))
    True

    Build a manager directly from a model with two observation sizes:

    >>> import liesel.model as lsl
    >>> y1 = lsl.Var.new_obs(jnp.arange(10.0), name="y1")
    >>> y2 = lsl.Var.new_obs(jnp.arange(6.0), name="y2")
    >>> model = lsl.Model([y1, y2])
    >>> managed = PositionSplitManager.from_model(
    ...     model, position_keys=["y1", "y2"], validate_axis_share=0.2
    ... )
    >>> managed.validate_axis_sizes
    (2, 1)
    """

    splits: Sequence[PositionSplit]
    _train: Position = field(init=False, repr=False)
    _validate: Position = field(init=False, repr=False)
    _test: Position = field(init=False, repr=False)

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
        self._train = _merge_positions([split.train for split in self.splits])
        self._validate = _merge_positions([split.validate for split in self.splits])
        self._test = _merge_positions([split.test for split in self.splits])

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
        validate_axis_share: float = 0.0,
        test_axis_share: float = 0.0,
        split_axes: dict[str, int] | None = None,
        default_split_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
        sample_sizes: SampleSizes | None = None,
        infer_sample_sizes: bool = True,
    ) -> PositionSplitManager:
        """
        Builds grouped position splits from a model.

        Observed variables are grouped by inferred axis size along their split
        axes. One :class:`Split` is constructed for each group and immediately
        applied to the model's observed position.

        Parameters are the same as :meth:`SplitManager.from_model`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.optim import PositionSplitManager
        >>> x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        >>> y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        >>> model = lsl.Model([x, y])
        >>> split = PositionSplitManager.from_model(
        ...     model, position_keys=["x", "y"], validate_axis_share=0.2
        ... )
        >>> split.axis_sizes
        (8, 5)
        >>> split.validate["x"].shape, split.validate["y"].shape
        ((1,), (1,))
        """
        splitter = SplitManager.from_model(
            model,
            position_keys=position_keys,
            validate_axis_share=validate_axis_share,
            test_axis_share=test_axis_share,
            split_axes=split_axes,
            default_split_axis=default_split_axis,
            shuffle=shuffle,
            seed=seed,
        )
        position = model.extract_position(splitter.position_keys)
        split = splitter.split_position(position)

        if sample_sizes is not None:
            for child in split.splits:
                child.set_sample_sizes(sample_sizes)
            return split

        if infer_sample_sizes:
            for child in split.splits:
                child.add_inferred_sample_sizes_from_model(model)
            return split

        return split

    @property
    def position_keys(self) -> list[str]:
        """
        Position keys claimed by all contained split objects.

        Examples
        --------
        >>> from liesel.optim import PositionSplitManager, PositionSplit
        >>> from liesel.optim.types import Position
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
        >>> from liesel.optim import PositionSplitManager, Split
        >>> pos = {"x": jnp.arange(3), "y": jnp.arange(4)}
        >>> manager = PositionSplitManager(
        ...     [
        ...         Split(["x"], axis_size=3).split_position(pos),
        ...         Split(["y"], axis_size=4).split_position(pos),
        ...     ]
        ... )
        >>> sorted(manager.train)
        ['x', 'y']
        """
        return self._train

    @property
    def validate(self) -> Position:
        """
        Merged validation position.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplitManager, Split
        >>> pos = {"x": jnp.arange(5), "y": jnp.arange(6)}
        >>> manager = PositionSplitManager(
        ...     [
        ...         Split(["x"], axis_size=5, validate_axis_size=1).split_position(pos),
        ...         Split(["y"], axis_size=6, validate_axis_size=1).split_position(pos),
        ...     ]
        ... )
        >>> sorted(manager.validate)
        ['x', 'y']
        """
        return self._validate

    @property
    def test(self) -> Position:
        """
        Merged test position.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplitManager, Split
        >>> pos = {"x": jnp.arange(5), "y": jnp.arange(6)}
        >>> manager = PositionSplitManager(
        ...     [
        ...         Split(["x"], axis_size=5, test_axis_size=1).split_position(pos),
        ...         Split(["y"], axis_size=6, test_axis_size=1).split_position(pos),
        ...     ]
        ... )
        >>> sorted(manager.test)
        ['x', 'y']
        """
        return self._test

    @property
    def axis_sizes(self) -> tuple[int, ...]:
        """
        Total axis sizes for each contained split.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplitManager, PositionSplit
        >>> from liesel.optim.types import Position
        >>> manager = PositionSplitManager(
        ...     [
        ...         PositionSplit(
        ...             Position({"x": jnp.arange(2)}),
        ...             Position({}),
        ...             Position({}),
        ...             2,
        ...             0,
        ...             0,
        ...         ),
        ...         PositionSplit(
        ...             Position({"y": jnp.arange(3)}),
        ...             Position({}),
        ...             Position({}),
        ...             3,
        ...             0,
        ...             0,
        ...         ),
        ...     ]
        ... )
        >>> manager.axis_sizes
        (2, 3)
        """
        return tuple(split.axis_size for split in self.splits)

    @property
    def train_axis_sizes(self) -> tuple[int, ...]:
        """Training axis sizes for each contained split."""
        return tuple(split.train_axis_size for split in self.splits)

    @property
    def validate_axis_sizes(self) -> tuple[int, ...]:
        """Validation axis sizes for each contained split."""
        return tuple(split.validate_axis_size for split in self.splits)

    @property
    def test_axis_sizes(self) -> tuple[int, ...]:
        """Test axis sizes for each contained split."""
        return tuple(split.test_axis_size for split in self.splits)

    @property
    def axis_size(self) -> int:
        """Common total axis size, available only when all branches agree."""
        return _common_value(self.axis_sizes, "axis_size", "axis_sizes")

    @property
    def train_axis_size(self) -> int:
        """Common training axis size, available only when all branches agree."""
        return _common_value(
            self.train_axis_sizes, "train_axis_size", "train_axis_sizes"
        )

    @property
    def validate_axis_size(self) -> int:
        """Common validation axis size, available only when all branches agree."""
        return _common_value(
            self.validate_axis_sizes, "validate_axis_size", "validate_axis_sizes"
        )

    @property
    def test_axis_size(self) -> int:
        """Common test axis size, available only when all branches agree."""
        return _common_value(self.test_axis_sizes, "test_axis_size", "test_axis_sizes")

    @property
    def has_validation(self) -> bool:
        """
        Whether all child splits contain validation data.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.optim import PositionSplitManager, Split
        >>> pos = {"x": jnp.arange(5), "y": jnp.arange(6)}
        >>> manager = PositionSplitManager(
        ...     [
        ...         Split(["x"], axis_size=5, validate_axis_size=1).split_position(pos),
        ...         Split(["y"], axis_size=6, validate_axis_size=1).split_position(pos),
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
    def validate_axis_shares(self) -> tuple[float, ...]:
        """Validation shares for each contained split."""
        return tuple(split.validate_axis_share for split in self.splits)

    @property
    def test_axis_shares(self) -> tuple[float, ...]:
        """Test shares for each contained split."""
        return tuple(split.test_axis_share for split in self.splits)

    @property
    def validate_axis_share(self) -> float:
        """Common validation share, available only when all branches agree."""
        return _common_value(
            self.validate_axis_shares, "validate_axis_share", "validate_axis_shares"
        )

    @property
    def test_axis_share(self) -> float:
        """Common test share, available only when all branches agree."""
        return _common_value(
            self.test_axis_shares, "test_axis_share", "test_axis_shares"
        )

    @property
    def validate_sample_scales(self) -> tuple[float, ...]:
        """Validation likelihood scales for each contained split."""
        return tuple(split.validate_sample_scale for split in self.splits)

    @property
    def validate_sample_scale(self) -> float:
        """
        Common validation likelihood scale.

        Raises
        ------
        ValueError
            If child validation scales differ. Use :meth:`scaled_log_lik` for
            per-branch scaling with a Liesel :class:`.Model`.
        """
        return _common_value(
            self.validate_sample_scales,
            "validate_sample_scale",
            "validate_sample_scales",
        )

    @property
    def sample_sizes(self) -> tuple[dict[SplitPart, float] | None, ...]:
        """Raw effective sample-size mappings for each contained split."""
        return tuple(
            None if split.sample_sizes is None else dict(split.sample_sizes)
            for split in self.splits
        )

    def sample_size(self, part: SplitPart) -> float:
        """Total effective likelihood sample size for one split part."""
        return sum(split.sample_size(part) for split in self.splits)

    @property
    def train_sample_sizes(self) -> tuple[float, ...]:
        """Training sample sizes for each contained split."""
        return tuple(split.train_sample_size for split in self.splits)

    @property
    def validate_sample_sizes(self) -> tuple[float, ...]:
        """Validation sample sizes for each contained split."""
        return tuple(split.validate_sample_size for split in self.splits)

    @property
    def test_sample_sizes(self) -> tuple[float, ...]:
        """Test sample sizes for each contained split."""
        return tuple(split.test_sample_size for split in self.splits)

    @property
    def train_sample_size(self) -> float:
        """Common training sample size, available only when all branches agree."""
        return _common_value(
            self.train_sample_sizes, "train_sample_size", "train_sample_sizes"
        )

    @property
    def validate_sample_size(self) -> float:
        """Common validation sample size, available only when all branches agree."""
        return _common_value(
            self.validate_sample_sizes, "validate_sample_size", "validate_sample_sizes"
        )

    @property
    def test_sample_size(self) -> float:
        """Common test sample size, available only when all branches agree."""
        return _common_value(
            self.test_sample_sizes, "test_sample_size", "test_sample_sizes"
        )

    def sample_scales(self, part: SplitPart) -> tuple[float, ...]:
        """Sample scaling factors for each contained split."""
        return tuple(split.sample_scale(part) for split in self.splits)

    def sample_scale(self, part: SplitPart) -> float:
        """Common sample scale, available only when all branches agree."""
        return _common_value(
            self.sample_scales(part), f"{part}_sample_scale", f"{part}_sample_scales"
        )

    def scaled_log_lik(
        self,
        model: Model | ModelInterface,
        model_state: ModelState,
        part: SplitPart = "validate",
    ) -> jax.Array:
        """
        Returns the log likelihood with branch-specific split scaling.

        The result is placed on the same effective likelihood scale as the
        training split. Validation and test likelihoods are therefore multiplied
        by each branch's ``train_sample_size / sample_size(part)``.

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
        >>> from liesel.optim import PositionSplitManager, Split
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
        ...         Split(["y1"], axis_size=10, validate_axis_size=2).split_position(
        ...             pos
        ...         ),
        ...         Split(["y2"], axis_size=6, validate_axis_size=1).split_position(
        ...             pos
        ...         ),
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
        scales = self.sample_scales(part)

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
            f"{name}(axis_size={self.axis_sizes}, train={self.train_axis_sizes}, "
            f"validate={self.validate_axis_sizes}, test={self.test_axis_sizes})"
        )


@dataclass
class SplitManager:
    """
    Wraps multiple :class:`Split` objects for multi-branch splitting.

    ``Split`` stays scalar: each instance assumes one axis size. ``SplitManager``
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
    >>> from liesel.optim import SplitManager, Split
    >>> manager = SplitManager(
    ...     [
    ...         Split(["x"], axis_size=10, validate_axis_size=2),
    ...         Split(["y"], axis_size=6, validate_axis_size=1),
    ...     ]
    ... )
    >>> split = manager.split_position({"x": jnp.arange(10), "y": jnp.arange(6)})
    >>> split.train_axis_sizes
    (8, 5)
    >>> split.validate["x"].tolist(), split.validate["y"].tolist()
    ([8, 9], [5])

    Automatically group model observations by axis size:

    >>> import liesel.model as lsl
    >>> x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
    >>> y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
    >>> model = lsl.Model([x, y])
    >>> manager = SplitManager.from_model(
    ...     model, position_keys=["x", "y"], validate_axis_share=0.2
    ... )
    >>> manager.axis_sizes
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
        validate_axis_share: float = 0.0,
        test_axis_share: float = 0.0,
        split_axes: dict[str, int] | None = None,
        default_split_axis: int = 0,
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
        validate_axis_share
            Share of observations assigned to validation in every child split.
        test_axis_share
            Share of observations assigned to testing in every child split.
        split_axes
            Optional mapping from position key to split axis.
        default_split_axis
            Split axis for all position keys not listed in ``split_axes``.
        shuffle
            Whether each child split shuffles observations.
        seed
            Seed or JAX pseudo-random key used when ``shuffle=True``.

        Returns
        -------
        SplitManager
            Split manager with one child :class:`Split` per inferred axis size.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.optim import SplitManager
        >>> x = lsl.Var.new_obs(jnp.arange(8.0), name="x")
        >>> y = lsl.Var.new_obs(jnp.arange(5.0), name="y")
        >>> model = lsl.Model([x, y])
        >>> manager = SplitManager.from_model(
        ...     model, position_keys=["x", "y"], validate_axis_share=0.2
        ... )
        >>> manager.position_keys
        ['x', 'y']
        >>> manager.validate_axis_sizes
        (1, 1)
        """
        pos_keys = (
            list(position_keys) if position_keys is not None else list(model.observed)
        )
        groups = position_key_groups_from_model(
            model, pos_keys, split_axes, default_split_axis
        )
        seeds = _child_seeds(seed, len(groups)) if shuffle else (seed,) * len(groups)
        splits = []

        for (axis_size, keys), child_seed in zip(groups.items(), seeds, strict=True):
            splits.append(
                Split.from_axis_shares(
                    position_keys=keys,
                    axis_size=axis_size,
                    validate_axis_share=validate_axis_share,
                    test_axis_share=test_axis_share,
                    split_axes=split_axes,
                    default_split_axis=default_split_axis,
                    shuffle=shuffle,
                    seed=child_seed,
                )
            )

        _validate_child_split_availability(splits, validate_axis_share, test_axis_share)

        return cls(splits)

    @property
    def position_keys(self) -> list[str]:
        """
        Position keys claimed by all contained splits.

        Examples
        --------
        >>> from liesel.optim import SplitManager, Split
        >>> SplitManager(
        ...     [Split(["x"], axis_size=3), Split(["y"], axis_size=4)]
        ... ).position_keys
        ['x', 'y']
        """
        keys: list[str] = []
        for split in self.splits:
            keys.extend(split.position_keys)
        return keys

    @property
    def axis_sizes(self) -> tuple[int, ...]:
        """Total axis sizes for each contained split."""
        return tuple(split.axis_size for split in self.splits)

    @property
    def train_axis_sizes(self) -> tuple[int, ...]:
        """Training axis sizes for each contained split."""
        return tuple(split._train_axis_size for split in self.splits)

    @property
    def validate_axis_sizes(self) -> tuple[int, ...]:
        """Validation axis sizes for each contained split."""
        return tuple(split.validate_axis_size for split in self.splits)

    @property
    def test_axis_sizes(self) -> tuple[int, ...]:
        """Test axis sizes for each contained split."""
        return tuple(split.test_axis_size for split in self.splits)

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
        >>> from liesel.optim import SplitManager, Split
        >>> manager = SplitManager(
        ...     [
        ...         Split(
        ...             ["x"], axis_size=4, validate_axis_size=1, default_split_axis=1
        ...         ),
        ...         Split(["y"], axis_size=6, validate_axis_size=2),
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
            f"{name}(axis_size={self.axis_sizes}, train={self.train_axis_sizes}, "
            f"validate={self.validate_axis_sizes}, test={self.test_axis_sizes})"
        )


@dataclass
class Split:
    """
    Defines how observed position entries are split into train, validation, and test.

    ``Split`` stores a vector of observation indices. The first
    ``train_axis_size`` indices become the training split, the next
    ``validate_axis_size`` indices become the validation split, and the final
    ``test_axis_size`` indices become the test split. If ``shuffle=True``, the
    index vector is permuted once during initialization.

    Parameters
    ----------
    position_keys
        Names of position entries that should be split.
    axis_size
        Number of observations along each split axis. Must be positive.
    validate_axis_size
        Number of validation observations.
    test_axis_size
        Number of test observations.
    train_axis_size
        Number of training observations. If left at ``None``, it is computed as
        ``axis_size - validate_axis_size - test_axis_size``.
    split_axes
        Optional mapping from position key to split axis. Keys missing from this
        mapping use ``default_split_axis``.
    default_split_axis
        Split axis for all position keys not listed in ``split_axes``.
    shuffle
        Whether to shuffle observations during initialization.
    seed
        Seed or JAX pseudo-random key used when ``shuffle=True``. If ``None``, the
        current time is used.
    sample_sizes
        Optional effective sample sizes passed to the resulting
        :class:`PositionSplit`.

    Attributes
    ----------
    indices
        Current observation order. Initialized as ``jnp.arange(axis_size)`` and
        optionally shuffled in :meth:`__post_init__`.

    Raises
    ------
    ValueError
        If ``axis_size`` is not positive, if any split size is negative, or if
        ``train_axis_size + validate_axis_size + test_axis_size`` is not exactly
        equal to ``axis_size``.

    Examples
    --------
    Split one vector without shuffling:

    >>> import jax.numpy as jnp
    >>> from liesel.optim import Split
    >>> splitter = Split(
    ...     ["x"], axis_size=10, validate_axis_size=2, test_axis_size=1, shuffle=False
    ... )
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

    Split different entries along different split_axes:

    >>> splitter = Split(
    ...     ["x", "y"],
    ...     axis_size=4,
    ...     validate_axis_size=1,
    ...     test_axis_size=1,
    ...     split_axes={"x": 1},
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
    axis_size: int
    validate_axis_size: int = 0
    test_axis_size: int = 0
    train_axis_size: int | None = None
    split_axes: dict[str, int] | None = field(default_factory=dict)
    default_split_axis: int = 0
    shuffle: bool = False
    seed: jax.Array | int | None = None
    sample_sizes: SampleSizes | None = None

    def __post_init__(self):
        if self.split_axes is None:
            self.split_axes = {}

        self.sample_sizes = _normalize_sample_sizes(self.sample_sizes)

        if self.axis_size <= 0:
            raise ValueError(f"{self.axis_size=} is <= 0, which is not allowed.")

        if len(set(self.position_keys)) != len(self.position_keys):
            raise ValueError(
                f"Duplicate position_keys are not allowed: {list(self.position_keys)}"
            )

        if self.train_axis_size is None:
            self.train_axis_size = (
                self.axis_size - self.validate_axis_size - self.test_axis_size
            )

        assert self.train_axis_size is not None
        self.indices = jnp.arange(self.axis_size)

        if (
            self.train_axis_size < 0
            or self.validate_axis_size < 0
            or self.test_axis_size < 0
        ):
            raise ValueError(
                f"Split sizes must be non-negative, but got {self.train_axis_size=}, "
                f"{self.validate_axis_size=}, and {self.test_axis_size=}."
            )

        if self.train_axis_size == 0:
            raise ValueError("train_axis_size must be positive.")

        n_split = self.train_axis_size + self.validate_axis_size + self.test_axis_size
        if n_split != self.axis_size:
            raise ValueError(
                f"The given {self.train_axis_size=}, {self.validate_axis_size=}, "
                f"and {self.test_axis_size=} sum to {n_split}, but must sum "
                f"exactly to {self.axis_size=}."
            )

        if self.shuffle:
            if isinstance(self.seed, jax.Array):
                key = self.seed
            else:
                seed = int(time.time()) if self.seed is None else self.seed
                key = jax.random.key(seed)
            self.indices = self.permute_indices(key)

    @property
    def _train_axis_size(self) -> int:
        assert self.train_axis_size is not None
        return self.train_axis_size

    @property
    def has_validation(self) -> bool:
        """
        Whether this splitter assigns observations to validation.

        Examples
        --------
        >>> from liesel.optim import Split
        >>> Split(["x"], axis_size=10, validate_axis_size=2).has_validation
        True
        """
        return self.validate_axis_size > 0

    @property
    def has_test(self) -> bool:
        """
        Whether this splitter assigns observations to testing.

        Examples
        --------
        >>> from liesel.optim import Split
        >>> Split(["x"], axis_size=10).has_test
        False
        """
        return self.test_axis_size > 0

    @property
    def validate_axis_share(self) -> float:
        """
        Share of observations assigned to validation.

        Returns
        -------
        float
            The ratio ``validate_axis_size / axis_size``.

        Examples
        --------
        >>> from liesel.optim import Split
        >>> Split(["x"], axis_size=10, validate_axis_size=2).validate_axis_share
        0.2
        """
        return self.validate_axis_size / self.axis_size

    @property
    def test_axis_share(self) -> float:
        """
        Share of observations assigned to testing.

        Returns
        -------
        float
            The ratio ``test_axis_size / axis_size``.

        Examples
        --------
        >>> from liesel.optim import Split
        >>> Split(["x"], axis_size=10, test_axis_size=3).test_axis_share
        0.3
        """
        return self.test_axis_size / self.axis_size

    @classmethod
    def from_axis_shares(
        cls,
        position_keys: Sequence[str],
        axis_size: int,
        validate_axis_share: float = 0.0,
        test_axis_share: float = 0.0,
        split_axes: dict[str, int] | None = None,
        default_split_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
        sample_sizes: SampleSizes | None = None,
    ) -> Split:
        """
        Builds a :class:`Split` from validation and test proportions.

        The number of validation and test observations is computed with
        ``int(axis_size * share)``. Any fractional remainder is assigned to the training
        split, so the resulting split sizes always sum to ``axis_size``.

        Parameters
        ----------
        position_keys
            Names of position entries that should be split.
        axis_size
            Number of observations along each split axis.
        validate_axis_share
            Share of observations assigned to validation.
        test_axis_share
            Share of observations assigned to testing.
        split_axes
            Optional mapping from position key to split axis.
        default_split_axis
            Split axis for all position keys not listed in ``split_axes``.
        shuffle
            Whether to shuffle observations during initialization.
        seed
            Seed or JAX pseudo-random key used when ``shuffle=True``.
        sample_sizes
            Optional effective sample sizes passed to the resulting
            :class:`PositionSplit`.

        Returns
        -------
        Split
            Splitter with integer split sizes derived from the shares.

        Examples
        --------
        >>> from liesel.optim import Split
        >>> splitter = Split.from_axis_shares(
        ...     ["x"],
        ...     axis_size=10,
        ...     validate_axis_share=0.25,
        ...     test_axis_share=0.25,
        ... )
        >>> (
        ...     splitter.train_axis_size,
        ...     splitter.validate_axis_size,
        ...     splitter.test_axis_size,
        ... )
        (6, 2, 2)
        >>> splitter.validate_axis_share, splitter.test_axis_share
        (0.2, 0.2)
        """
        if axis_size <= 0:
            raise ValueError(f"{axis_size=} is <= 0, which is not allowed.")

        if validate_axis_share < 0.0 or test_axis_share < 0.0:
            raise ValueError(
                f"Shares must be non-negative, but got {validate_axis_share=} "
                f"and {test_axis_share=}."
            )

        share_observed = validate_axis_share + test_axis_share
        if share_observed > 1.0:
            raise ValueError(
                f"Validation and test shares sum to {share_observed}, which is > 1.0."
            )

        validate_axis_size = int(axis_size * validate_axis_share)
        test_axis_size = int(axis_size * test_axis_share)
        train_axis_size = axis_size - validate_axis_size - test_axis_size

        return cls(
            position_keys=position_keys,
            axis_size=axis_size,
            validate_axis_size=validate_axis_size,
            test_axis_size=test_axis_size,
            train_axis_size=train_axis_size,
            split_axes=split_axes,
            default_split_axis=default_split_axis,
            shuffle=shuffle,
            seed=seed,
            sample_sizes=sample_sizes,
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
        >>> from liesel.optim import Split
        >>> splitter = Split(["x"], axis_size=5)
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
        >>> from liesel.optim import Split
        >>> Split(
        ...     ["x"], axis_size=6, validate_axis_size=2, test_axis_size=1
        ... ).indices_train.tolist()
        [0, 1, 2]
        """
        return self.indices[: self._train_axis_size]

    @property
    def indices_validate(self) -> jax.Array:
        """
        Observation indices for the validation split.

        Examples
        --------
        >>> from liesel.optim import Split
        >>> Split(
        ...     ["x"], axis_size=6, validate_axis_size=2, test_axis_size=1
        ... ).indices_validate.tolist()
        [3, 4]
        """
        start = self._train_axis_size
        end = self._train_axis_size + self.validate_axis_size
        return self.indices[start:end]

    @property
    def indices_test(self) -> jax.Array:
        """
        Observation indices for the test split.

        Examples
        --------
        >>> from liesel.optim import Split
        >>> Split(
        ...     ["x"], axis_size=6, validate_axis_size=2, test_axis_size=1
        ... ).indices_test.tolist()
        [5]
        """
        start = self._train_axis_size + self.validate_axis_size
        end = self._train_axis_size + self.validate_axis_size + self.test_axis_size
        return self.indices[start:end]

    def split_position(self, position: Position) -> PositionSplit:
        """
        Splits position entries into train, validation, and test positions.

        Parameters
        ----------
        position
            Mapping containing every key in :attr:`position_keys`. Each selected
            entry must have length ``axis_size`` along its split axis.

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
        >>> from liesel.optim import Split
        >>> splitter = Split(["x"], axis_size=5, validate_axis_size=1, test_axis_size=1)
        >>> split = splitter.split_position({"x": jnp.arange(5)})
        >>> split.train["x"].tolist()
        [0, 1, 2]
        >>> split.validate["x"].tolist(), split.test["x"].tolist()
        ([3], [4])

        Use ``split_axes`` to split an array along a non-leading axis:

        >>> splitter = Split(
        ...     ["x"], axis_size=4, validate_axis_size=1, split_axes={"x": 1}
        ... )
        >>> split = splitter.split_position({"x": jnp.arange(8).reshape(2, 4)})
        >>> split.train["x"].tolist()
        [[0, 1, 2], [4, 5, 6]]
        >>> split.validate["x"].tolist()
        [[3], [7]]
        """
        train_position = {}
        validation_position = {}
        test_position = {}

        assert self.split_axes is not None
        for key in self.position_keys:
            axis = self.split_axes.get(key, self.default_split_axis)

            n_this_key = jnp.shape(position[key])[axis]
            if not jnp.shape(position[key])[axis] == self.axis_size:
                raise ValueError(
                    f"{key} has axis_size={n_this_key}, which is incompatible with the "
                    f"given axis_size={self.axis_size}."
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
            train_axis_size=self._train_axis_size,
            validate_axis_size=self.validate_axis_size,
            test_axis_size=self.test_axis_size,
            sample_sizes=self.sample_sizes,
        )
        return split

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.train_axis_size}, "
            f"validate={self.validate_axis_size}, test={self.test_axis_size})"
        )
        return out
