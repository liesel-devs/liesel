"""Internal helpers for likelihood and prior factor scaling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import networkx as nx

from ..model import Calc, Model
from ..model.model import _reduced_sum
from .types import ModelState


def sum_value(value):
    return value.sum() if hasattr(value, "sum") else value


def sum_state_value(model_state: ModelState, name: str):
    return sum_value(model_state[name].value)


def observed_log_lik_sources(
    model: Model, position_keys: Sequence[str]
) -> dict[str, list[str]]:
    """Map observed factors to selected keys through all log-density inputs."""
    return log_prob_sources(model, position_keys, prior=False)


def log_prob_sources(
    model: Model, position_keys: Sequence[str], *, prior: bool | None = None
) -> dict[str, list[str]]:
    """Map factors to selected keys through value and distribution inputs.

    ``prior=None`` includes both observed and parameter factors. Dependence on a
    selected key assumes the factor represents the selected units additively.
    """
    selected = {
        key: model.vars[key].value_node if key in model.vars else model.nodes[key]
        for key in position_keys
    }
    sources = {}

    variables = (
        {**model.observed, **model.parameters}
        if prior is None
        else model.parameters
        if prior
        else model.observed
    )
    for var in variables.values():
        if var.dist_node is None:
            continue
        nodes = {var.dist_node}
        pending = list(nodes)
        while pending:
            node = pending.pop()
            for parent in node.all_input_nodes():
                if parent not in nodes:
                    nodes.add(parent)
                    pending.append(parent)
        keys = [key for key, node in selected.items() if node in nodes]
        if keys:
            sources[var.name] = keys
    return sources


def validate_factor_groups(model: Model, groups: Sequence[Sequence[str]]) -> None:
    """Reject factors shared by independently sampled row groups."""
    covered: dict[str, Sequence[str]] = {}
    for keys in groups:
        for name in log_prob_sources(model, keys):
            if name in covered:
                raise ValueError(
                    f"Log-density factor {name!r} is covered by more than one data "
                    f"group: {list(covered[name])} and {list(keys)}. Put aligned "
                    "inputs in one explicit group, mark shared data as passthrough "
                    "with split_axes={key: None}, or use a custom Loss."
                )
            covered[name] = keys


def scaled_liesel_log_lik(
    model: Model,
    model_state: ModelState,
    groups: Sequence[tuple[Sequence[str], float]],
    corrections: dict[str, tuple[jax.Array, int]] | None = None,
    *,
    include_uncovered: bool = True,
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
    variables are evaluated on their full data. Set ``include_uncovered=False``
    for held-out scores, which must exclude unsplit training likelihoods.

    Raises
    ------
    ValueError
        If one observed log-likelihood node is covered by more than one group.
    """
    return _scaled_log_factors(
        model,
        model_state,
        groups,
        corrections,
        prior=False,
        include_uncovered=include_uncovered,
    )


def scaled_liesel_log_prior(
    model: Model,
    model_state: ModelState,
    groups: Sequence[tuple[Sequence[str], float]],
    corrections: dict[str, tuple[jax.Array, int]] | None = None,
):
    """Scale covered parameter priors, including uncovered priors once."""
    validate_factor_groups(model, [keys for keys, _ in groups])
    node = model.nodes["_model_log_prior"]
    if not any(log_prob_sources(model, keys, prior=True) for keys, _ in groups):
        selected = {
            model._node_for_position_key(key) for keys, _ in groups for key in keys
        }
        if not selected & nx.ancestors(model.node_graph, node):
            return model_state["_model_log_prior"].value

    expected = Counter(
        var.dist_node.name for var in model.parameters.values() if var.has_dist
    )
    if not (
        isinstance(node, Calc)
        and node.function is _reduced_sum
        and not node.kwinputs
        and Counter(parent.name for parent in node.inputs) == expected
    ):
        raise ValueError(
            "Cannot scale batch-dependent priors with a custom log_prior_node. "
            "Use a custom Loss."
        )
    return _scaled_log_factors(model, model_state, groups, corrections, prior=True)


def _scaled_log_factors(
    model: Model,
    model_state: ModelState,
    groups: Sequence[tuple[Sequence[str], float]],
    corrections: dict[str, tuple[jax.Array, int]] | None,
    *,
    prior: bool,
    include_uncovered: bool = True,
):
    validate_factor_groups(model, [keys for keys, _ in groups])
    variables = model.parameters if prior else model.observed
    total = 0.0
    covered_nodes: set[str] = set()

    for position_keys, scale in groups:
        node_names = [
            variables[name].dist_node.name
            for name in log_prob_sources(model, position_keys, prior=prior)
        ]

        for node_name in node_names:
            value = model_state[node_name].value
            if corrections and node_name in corrections:
                factors, axis = corrections[node_name]
                shape = [1] * value.ndim
                shape[axis] = factors.size
                value = value * jnp.reshape(factors, shape)
            total += scale * sum_value(value)
            covered_nodes.add(node_name)

    if include_uncovered:
        for var in variables.values():
            if var.has_dist and var.dist_node.name not in covered_nodes:
                total += sum_state_value(model_state, var.dist_node.name)

    return total


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
