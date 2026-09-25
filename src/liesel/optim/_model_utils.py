"""Internal helpers for model-derived optimizer configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions.distribution import (
    DiscreteDistributionMixin,
)

from ..model import Model, Value
from ._log_lik import observed_log_lik_node_names, validate_likelihood_groups


def strong_observed_keys(model: Model) -> list[str]:
    """Observed inputs whose values can be replaced by split or batch data."""
    return [name for name, var in model.observed.items() if not var.weak]


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
    for var in model.vars.values():
        if var.weak and (var.name in flat_keys or var.value_node.name in flat_keys):
            raise ValueError(
                f"Cannot split or batch weak variable {var.name!r} directly. "
                "Select its strong source data instead; weak observed values "
                "and their likelihoods are recomputed from those inputs."
            )
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


def continuous_coordinate_nodes(
    model: Model, keys: Sequence[str], data_nodes: set
) -> set:
    """Resolve distinct writable continuous parameters, excluding data."""
    if isinstance(keys, str):
        raise ValueError("Pass coordinate names as a sequence, not a string.")  # noqa: TRY004
    nodes = set()
    for key in keys:
        if key in model.nodes:
            node = model.nodes[key]
        elif key in model.vars:
            node = model.vars[key].value_node
        else:
            raise ValueError(f"Unknown coordinate {key!r}.")
        if node in nodes:
            raise ValueError(f"Duplicate coordinate or alias {key!r}.")
        if not isinstance(node, Value) or not jnp.issubdtype(
            jnp.asarray(node.value).dtype, jnp.floating
        ):
            raise ValueError(f"{key!r} must be a writable continuous coordinate.")
        if node in data_nodes or (node.var is not None and node.var.observed):
            raise ValueError(f"{key!r} is a training input or observation.")
        dist = (
            node.var.dist_node.init_dist()
            if node.var is not None and node.var.dist_node is not None
            else None
        )
        while dist is not None:
            if isinstance(dist, DiscreteDistributionMixin):
                raise ValueError(f"{key!r} has a discrete distribution.")  # noqa: TRY004
            dist = getattr(
                dist, "distribution", getattr(dist, "components_distribution", None)
            )
        nodes.add(node)
    return nodes
