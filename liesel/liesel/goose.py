"""
# The Liesel-Goose interface
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Callable, Iterable

from .types import ModelState, Position

if TYPE_CHECKING:
    from .model import Model


def get_position(position_keys: Iterable[str], model_state: ModelState) -> Position:
    """
    Extracts a position from a model state.

    A position is a dictionary of node names and values.
    """

    return Position({key: model_state[key].value for key in position_keys})


def update_state(
    position: Position, model_state: ModelState, model: Model
) -> ModelState:
    """Updates and returns a model state given a position."""

    model.state = model_state

    for node in model.sorted_nodes:
        node.outdated = False

    for name, value in position.items():
        model.nodes[name].set_value(value, update=False)

    model.update()
    return model.state


def get_log_prob(model_state: ModelState) -> float:
    """Computes and returns the log-probability from a model state."""

    log_probs = [node_state.log_prob for node_state in model_state.values()]
    return reduce(lambda x, y: x + y, log_probs)


def make_update_state_fn(
    model: Model, jaxify=True
) -> Callable[[Position, ModelState], ModelState]:
    """Returns a pure and jittable `update_state` function for the provided model."""

    model = model.empty_copy()

    if jaxify:
        model.jaxify()

    def fn(position: Position, model_state: ModelState) -> ModelState:
        return update_state(position, model_state, model)

    return fn


def make_log_prob_fn(
    model: Model, jaxify=True
) -> Callable[[Position, ModelState], float]:
    """Returns a pure and jittable `log_prob` function for the provided model."""

    model = model.empty_copy()

    if jaxify:
        model.jaxify()

    def fn(position: Position, model_state: ModelState) -> float:
        return get_log_prob(update_state(position, model_state, model))

    return fn


class GooseModel:
    """A `liesel.goose.ModelInterface` for a `liesel.liesel.Model`."""

    def __init__(self, model: Model):
        self._update_state_fn = make_update_state_fn(model)

    def extract_position(
        self, position_keys: Iterable[str], model_state: ModelState
    ) -> Position:
        return get_position(position_keys, model_state)

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        return self._update_state_fn(position, model_state)

    def log_prob(self, model_state: ModelState) -> float:
        return get_log_prob(model_state)
