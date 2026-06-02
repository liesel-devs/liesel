"""Internal helpers for model-derived optimizer configuration."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from ..model import Model


def position_key_groups_from_model(
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
