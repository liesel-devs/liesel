"""
Interface to define a target for inference.

KernelTargets are a weighted sums of selected log-probability terms of a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

from .nodes import Dist, Var

if TYPE_CHECKING:
    from ..goose.types import ModelState, Scalar
    from .model import Model

type Component = Literal["log_prob", "log_lik", "log_prior"]


@dataclass(frozen=True)
class LogProbTarget:
    """
    A weighted sum of distribution log-probabilities, evaluated on a model state.

    Usually created with :meth:`~liesel.model.Model.target`. Can be passed as
    ``target`` to some Goose kernels.

    TODO: consider allowing the weights to be part of the model state,
    which could updateed on a schedule allowing for tempering or annealing.
    Also, if weights are chain specific, one could easily sweep in a single run,
    with multiple chains. Would imply changes to `TargetSpec`.
    """

    weights: Mapping[str, float]
    """Weights of the terms, keyed by distribution node name."""

    def __call__(self, model_state: ModelState) -> Scalar:
        return sum(
            w * jnp.sum(model_state[name].value) for name, w in self.weights.items()
        )


@dataclass(frozen=True, eq=False)
class TargetSpec:
    """
    Describes a :class:`.LogProbTarget` without needing a model.

    Use it in :class:`~liesel.goose.MCMCSpec`, where the model does not exist yet.
    The arguments are those of :meth:`~liesel.model.Model.target`.

    TODO: decide whether Component should be kept, e.g., for prior sampling.
    """

    include: Sequence[Var | str] | None = None
    exclude: Sequence[Var | str] = ()
    weights: Mapping[Var | str, float] = field(default_factory=dict)
    component: Component = "log_prob"

    def resolve(self, model: Model) -> LogProbTarget:
        """Maps the variables to the distribution nodes of ``model``."""
        if self.include is not None and self.component != "log_prob":
            raise ValueError("Use either 'include' or 'component', not both.")

        if self.include is not None:
            terms = [_dist_name(model, v) for v in self.include]
        else:
            terms = _component_terms(model, self.component)

        excluded = {_dist_name(model, v) for v in self.exclude}
        weights = {name: 1.0 for name in terms if name not in excluded}

        for v, w in self.weights.items():
            name = _dist_name(model, v)
            if name not in weights:
                raise ValueError(f"Cannot weight {v!r}, it is not part of the target.")
            weights[name] = w

        if not weights:
            raise ValueError("The target does not contain any terms.")

        return LogProbTarget(weights)


def _component_terms(model: Model, component: Component) -> list[str]:
    match component:
        case "log_prob":
            return [n.name for n in model.nodes.values() if isinstance(n, Dist)]
        case "log_lik":
            vars_ = model.observed.values()
        case "log_prior":
            vars_ = model.parameters.values()
        case _:
            raise ValueError(f"Unknown component {component!r}.")
    return [v.dist_node.name for v in vars_ if v.has_dist]


def _dist_name(model: Model, var: Var | str) -> str:
    """
    Name of the distribution node that carries the density of ``var``.

    For a transformed variable, this is the distribution of the transformed
    variable.
    """
    name = var if isinstance(var, str) else var.name
    if name not in model.vars:
        raise ValueError(f"Variable {name!r} is not part of the model.")
    var = model.vars[name]

    if var.dist_node is not None:
        return var.dist_node.name
    if var.has_bijected_var and var.bijected_var.dist_node is not None:
        return var.bijected_var.dist_node.name
    raise ValueError(f"Variable {name!r} has no distribution.")
