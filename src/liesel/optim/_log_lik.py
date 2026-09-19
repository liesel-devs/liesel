"""Internal helpers for observed log-likelihood scaling."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from ..model import Model
from .types import ModelState


def sum_value(value):
    return value.sum() if hasattr(value, "sum") else value


def sum_state_value(model_state: ModelState, name: str):
    return sum_value(model_state[name].value)


def observed_log_lik_node_names(
    model: Model, position_keys: Sequence[str]
) -> list[str]:
    """
    Retrieve names of the log prob nodes of variables named in ``position_keys``.
    """
    keys = set(position_keys)
    node_names: list[str] = []

    for var in model.observed.values():
        if var.dist_node is None:
            continue

        if var.name in keys or var.value_node.name in keys:
            node_names.append(var.dist_node.name)

    return node_names


def all_observed_log_lik_node_names(model: Model) -> list[str]:
    node_names: list[str] = []

    for var in model.observed.values():
        if var.dist_node is not None:
            node_names.append(var.dist_node.name)

    return node_names


def scaled_liesel_log_lik(
    model: Model,
    model_state: ModelState,
    groups: Sequence[tuple[Sequence[str], float]],
    corrections: dict[str, tuple[jax.Array, int]] | None = None,
):
    """Return the model log likelihood with per-group scaling.

    Each entry in ``groups`` describes observed variables or value nodes that
    belong to one batched data group and the scaling factor for that group. The
    corresponding observed log-likelihood nodes are summed and multiplied by the
    group scale.

    Optional ``corrections`` maps likelihood node names to per-index factors and
    the likelihood axis they apply to. These factors are applied before summation.

    Observed likelihood contributions that are not covered by any group are
    still included with scale 1.0. This supports partially batched models, where
    some observed variables are evaluated on a batch while other observed
    variables are evaluated on their full data.

    Raises
    ------
    ValueError
        If one observed log-likelihood node is covered by more than one group.
    """
    scaled_log_lik = 0.0
    covered_nodes: set[str] = set()

    for position_keys, scale in groups:
        node_names = observed_log_lik_node_names(model, position_keys)

        for node_name in node_names:
            if node_name in covered_nodes:
                raise ValueError(
                    f"The observed log-likelihood node {node_name!r} is covered by "
                    "more than one data group."
                )

            value = model_state[node_name].value
            if corrections and node_name in corrections:
                factors, axis = corrections[node_name]
                shape = [1] * value.ndim
                shape[axis] = factors.size
                value = value * jnp.reshape(factors, shape)
            scaled_log_lik += scale * sum_value(value)
            covered_nodes.add(node_name)

    for node_name in all_observed_log_lik_node_names(model):
        if node_name not in covered_nodes:
            # Nodes not linked to a batched data group are assumed to be full-data
            # likelihood terms. Include them unscaled so partial batching does not
            # silently drop observed model branches.
            scaled_log_lik += sum_state_value(model_state, node_name)

    return scaled_log_lik


def scaled_common_log_lik(model_state: ModelState, scale: float):
    try:
        log_lik = model_state["_model_log_lik"].value
    except (KeyError, TypeError, AttributeError) as error:
        raise TypeError(
            "Per-branch likelihood scaling requires a liesel.model.Model. For a "
            "generic model interface, the model state must expose "
            "'_model_log_lik'."
        ) from error

    return scale * log_lik
