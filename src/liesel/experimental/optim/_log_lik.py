"""Internal helpers for observed log-likelihood scaling."""

from __future__ import annotations

from collections.abc import Sequence

from ...model import Model
from .types import ModelState


def sum_value(value):
    return value.sum() if hasattr(value, "sum") else value


def sum_state_value(model_state: ModelState, name: str):
    return sum_value(model_state[name].value)


def observed_log_lik_node_names(
    model: Model, position_keys: Sequence[str]
) -> list[str]:
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
):
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

            scaled_log_lik += scale * sum_state_value(model_state, node_name)
            covered_nodes.add(node_name)

    for node_name in all_observed_log_lik_node_names(model):
        if node_name not in covered_nodes:
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
