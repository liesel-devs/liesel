"""Internal helpers for model-derived optimizer configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp

from ..model import Model


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
    position = model.extract_position(flat_keys)
    groups = []

    for selection in selections:
        by_size: dict[int, list[str]] = {}
        for key in selection:
            axis = split_axes.get(key, default_split_axis)
            if axis is None:
                continue

            n_key = int(jnp.shape(position[key])[axis])
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

    return flat_keys, groups
