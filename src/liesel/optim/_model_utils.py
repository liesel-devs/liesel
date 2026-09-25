"""Internal helpers for model-derived optimizer configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import networkx as nx

from ..model import Model, TransientNode, Value
from ._log_lik import observed_log_lik_node_names, validate_likelihood_groups


def strong_observed_keys(model: Model) -> list[str]:
    """Default observed inputs; computed variables must be selected explicitly."""
    return [name for name, var in model.observed.items() if not var.weak]


def multi_size_error_message(
    factory: str, groups: Sequence[tuple[int, list[str]]]
) -> str:
    """Explain inferred groups and the explicit grouping and shared-data routes."""
    description = "; ".join(f"{keys} (axis length {size})" for size, keys in groups)
    manager = "BatchManager" if factory == "Batches" else f"{factory}Manager"
    return (
        f"{factory}.from_model() found multiple observation groups: {description}. "
        "Matching lengths do not establish row alignment. Check the groups before "
        "setting multi_size='manager'. For LieselOptim, pass "
        "split=opt.PositionSplit.from_model(model, multi_size='manager'). "
        f"Use nested position_keys with {manager}.from_model(...) to choose groups "
        "explicitly. For shared values, "
        "construct a split with split_axes={key: None}, then pass it as split=split "
        "or use Batches.from_split(split, ...)."
    )


def validate_model_data_keys(
    model: Model,
    position_keys: Sequence[str],
    parameter_keys: Sequence[str] = (),
) -> None:
    """Require writable, non-transient data keys with unambiguous computed values."""
    for key in position_keys:
        node = model._node_for_position_key(key)
        if isinstance(node, TransientNode):
            raise ValueError(  # noqa: TRY004
                f"Cannot split or batch transient data key {key!r}."
            )
        if key not in model.vars and not isinstance(node, Value):
            raise ValueError(
                f"Use a variable name for computed data, not node key {key!r}."
            )
        if parameter_keys and key in model.vars and model.vars[key].weak:
            ancestors = nx.ancestors(model.node_graph, node)
            for parameter in parameter_keys:
                if model._node_for_position_key(parameter) in ancestors:
                    raise ValueError(
                        f"Computed data key {key!r} depends on optimized parameter "
                        f"{parameter!r}; batch its fixed inputs instead."
                    )
    try:
        model._validate_weak_var_position(model.extract_position(position_keys))
    except RuntimeError as error:
        raise ValueError(str(error)) from error


def position_key_groups_from_model(
    model: Model,
    position_keys: Sequence[str] | Sequence[Sequence[str]],
    split_axes: Mapping[str, int | None] | None,
    default_split_axis: int,
) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Return flat selected keys and ordered groups with their axis lengths.

    ``split_axes`` may override the axis used for individual keys. Keys mapped to
    ``None`` are passthrough data and are omitted from the groups. Keys without an
    override use ``default_split_axis``. Flat inputs are grouped by axis length;
    nested inputs retain explicit boundaries, even for equal-sized groups.
    Inferred groups must contain an observed likelihood; otherwise callers must
    specify explicit nested groups or mark shared data as passthrough.
    The flat keys retain passthrough entries for extraction from the model.
    """
    split_axes = split_axes or {}
    explicit = any(not isinstance(key, str) for key in position_keys)
    selections: list[list[str]] = []
    if explicit:
        for group in position_keys:
            if isinstance(group, str) or not isinstance(group, Sequence):
                raise TypeError("position_keys must be uniformly flat or nested.")
            if not group:
                raise ValueError("Explicit position-key groups must not be empty.")
            if any(not isinstance(key, str) for key in group):
                raise TypeError("Position keys within each group must be strings.")
            selections.append(list(group))
    else:
        selections.append([key for key in position_keys if isinstance(key, str)])
    flat_keys = [key for group in selections for key in group]
    if len(set(flat_keys)) != len(flat_keys):
        raise ValueError(f"Duplicate position_keys are not allowed: {flat_keys}")
    validate_model_data_keys(model, flat_keys)
    position = model.extract_position(flat_keys)
    groups = []

    for selection in selections:
        by_size: dict[int, list[str]] = {}
        for key in selection:
            axis = split_axes.get(key, default_split_axis)
            if axis is None:
                continue

            shape = jnp.shape(position[key])
            if not -len(shape) <= axis < len(shape):
                raise ValueError(
                    f"Cannot split or batch {key!r} with shape {shape} on axis "
                    f"{axis}. For scalars or shared data, construct a split with "
                    f"PositionSplit.from_model(..., split_axes={{{key!r}: None}}), "
                    "then use LieselOptim(..., split=split) or "
                    "Batches.from_split(split, ...)."
                )
            n_key = int(shape[axis])
            by_size.setdefault(n_key, []).append(key)
        if explicit and not by_size:
            raise ValueError(
                f"Explicit group {selection} contains only passthrough entries. "
                "Each group must contain at least one position key to be split."
            )
        if explicit and len(by_size) > 1:
            raise ValueError(
                f"Explicit group {selection} must have "
                f"matching axis lengths, got {list(by_size)}."
            )
        groups.extend(by_size.items())

    if not explicit:
        for _, keys in groups:
            if not observed_log_lik_node_names(model, keys):
                passthrough = dict.fromkeys(keys)
                raise ValueError(
                    f"Cannot infer an observation group for {keys}: no observed "
                    "likelihood belongs to this group. For shared data, construct "
                    "PositionSplit.from_model(..., "
                    f"split_axes={passthrough!r}), then use "
                    "LieselOptim(..., split=split) or Batches.from_split(split, ...). "
                    "For intentional row groups without a likelihood, provide "
                    "explicit nested position_keys."
                )

    validate_likelihood_groups(model, [keys for _, keys in groups])
    return flat_keys, groups
