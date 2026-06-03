"""Internal helpers for model-derived optimizer configuration."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from ..model import Model


def position_key_groups_from_model(
    model: Model,
    position_keys: Sequence[str],
    split_axes: dict[str, int] | None,
    default_split_axis: int,
) -> dict[int, list[str]]:
    """Group position keys by the length of their split axis.

    ``split_axes`` may override the axis used for individual keys. Keys without
    an override use ``default_split_axis``. The returned dictionary maps each
    observed axis length to the position keys with that length, so callers can
    build compatible batches or data splits for each group.

    For example, if ``"y"`` and ``"x"`` have length 100 along their selected
    axes and ``"z"`` has length 20, the result might be
    ``{100: ["y", "x"], 20: ["z"]}``.
    """
    split_axes = split_axes or {}
    position = model.extract_position(position_keys)
    groups: dict[int, list[str]] = {}

    for key in position_keys:
        axis = split_axes.get(key, default_split_axis)
        n_key = int(jnp.shape(position[key])[axis])
        groups.setdefault(n_key, []).append(key)

    return groups
