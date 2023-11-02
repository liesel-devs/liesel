"""
Model interfaces.
"""

import copy
import warnings
from collections.abc import Callable, Sequence

from ..docs import usedocs
from .types import ModelInterface, ModelState, Position

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
