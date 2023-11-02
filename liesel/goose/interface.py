"""
Model interfaces.
"""

import copy
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..docs import usedocs
from .types import ModelInterface, ModelState, Position

if TYPE_CHECKING:
    from ..model.model import Model

LogProbFunction = Callable[[ModelState], float]


@usedocs(ModelInterface)
class DictInterface:
    """
    A model interface for a model state represented by a ``dict[str, Array]`` and a
    corresponding log-probability function.
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        self._log_prob_fn = log_prob_fn

    def extract_position(
        self, position_keys: Sequence[str], model_state: ModelState
    ) -> Position:
        return Position({key: model_state[key] for key in position_keys})

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        return model_state | position

    def log_prob(self, model_state: ModelState) -> float:
        return self._log_prob_fn(model_state)


@usedocs(ModelInterface)
class DataclassInterface:
    """
    A model interface for a model state represented by a :obj:`~dataclasses.dataclass`
    and a corresponding log-probability function.
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        self._log_prob_fn = log_prob_fn

    def extract_position(
        self, position_keys: Sequence[str], model_state: ModelState
    ) -> Position:
        return Position({key: getattr(model_state, key) for key in position_keys})

    def log_prob(self, model_state: ModelState) -> float:
        return self._log_prob_fn(model_state)

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        new_state = copy.copy(model_state)  # don't change the input
        for key, value in position.items():
            if hasattr(new_state, key):
                setattr(new_state, key, value)
            else:
                raise RuntimeError(
                    f"ModelState {model_state!r} does not have field with name {key}"
                )
        return new_state


class DictModel(DictInterface):
    """
    Alias for :class:`.DictInterface`, provided for backwards compatibility.

    .. deprecated:: v0.2.6
        Use :class:`.DictInterface` instead. This alias will be removed in v0.4.0.
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        self._log_prob_fn = log_prob_fn

        warnings.warn(
            "Use gs.DictInterface instead. This alias will be removed in v0.4.0.",
            FutureWarning,
        )


class DataClassModel(DataclassInterface):
    """
    Alias for :class:`.DataclassInterface`, provided for backwards compatibility.

    .. deprecated:: v0.2.6
        Use :class:`.DataclassInterface` instead. This alias will be removed in v0.4.0.
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        self._log_prob_fn = log_prob_fn

        warnings.warn(
            "Use gs.DataclassInterface instead. This alias will be removed in v0.4.0.",
            FutureWarning,
        )


class LieselInterface:
    """
    A :class:`.ModelInterface` for a Liesel :class:`.Model`.

    Parameters
    ----------
    model
        A Liesel :class:`.Model`.

    See Also
    --------
    :class:`.Model` : The liesel model class.
    """

    def __init__(self, model: "Model"):
        self._model = model._copy_computational_model()

    def extract_position(
        self, position_keys: Sequence[str], model_state: ModelState
    ) -> Position:
        """
        Extracts a position from a model state.

        Parameters
        ----------
        position_keys
            An iterable of variable or node names.
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.
        """
        position = {}

        for key in position_keys:
            try:
                position[key] = model_state[key].value
            except KeyError:
                node_key = self._model.vars[key].value_node.name
                position[key] = model_state[node_key].value

        return Position(position)

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        """
        Updates and returns a model state given a position.

        Parameters
        ----------
        position
            A dictionary of variable or node names and values.
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.

        Warnings
        --------
        The ``model_state`` must be up-to-date, i.e. it must *not* contain any outdated
        nodes. Updates can only be triggered through new variable or node values in the
        ``position``. If you supply a ``model_state`` with outdated nodes, these nodes
        and their outputs will not be updated.
        """

        # sets all outdated flags in the model state to false
        # this is required to make the function jittable

        self._model.state = model_state

        for node in self._model.nodes.values():
            node._outdated = False

        for key, value in position.items():
            try:
                self._model.nodes[key].value = value  # type: ignore  # data node
            except KeyError:
                self._model.vars[key].value = value

        self._model.update()
        return self._model.state

    def log_prob(self, model_state: ModelState) -> float:
        """
        Returns the log-probability from a model state.

        Parameters
        ----------
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.
        """
        return model_state["_model_log_prob"].value
