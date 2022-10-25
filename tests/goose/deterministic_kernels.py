"""contains a deterministic kernel used for test purposes"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp

from liesel.goose.epoch import EpochState, EpochType
from liesel.goose.kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    TransitionMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from liesel.goose.pytree import register_dataclass_as_pytree
from liesel.goose.types import KeyArray, ModelInterface, ModelState, Position


@register_dataclass_as_pytree
@dataclass
class DetCountingKernelState:
    increment_per_transition: int
    total_transitions: int
    total_adaptive_transitions: int
    post_warmup_transitions: int
    epoch_counter: int
    in_epoch: bool
    tune_counter: int
    warmup_finialized: bool

    @staticmethod
    def default() -> "DetCountingKernelState":
        return DetCountingKernelState(
            increment_per_transition=1,
            total_transitions=0,
            total_adaptive_transitions=0,
            post_warmup_transitions=0,
            epoch_counter=0,
            in_epoch=False,
            tune_counter=0,
            warmup_finialized=False,
        )


@register_dataclass_as_pytree
@dataclass
class DetCountingKernelTuningInfo(DefaultTuningInfo):
    pass


@register_dataclass_as_pytree
@dataclass
class DetCountingTransInfo(DefaultTransitionInfo):
    useless_field: float = 1.0


class DetCountingKernel(TransitionMixin[DetCountingKernelState, DetCountingTransInfo]):
    error_book: ClassVar[dict[int, str]] = {0: "no errors"}
    needs_history: ClassVar[bool] = False
    identifier: str = ""

    def __init__(
        self,
        position_keys: Sequence[str],
        initial_state: DetCountingKernelState,
    ):
        self._model: None | ModelInterface = None
        self.position_keys = tuple(position_keys)
        self._initial_state = initial_state

    @property
    def model(self) -> ModelInterface:
        """Returns the model interface if it is set. Raises error otherwise."""

        if self._model is None:
            raise RuntimeError("Model interface not set")

        return self._model

    def set_model(self, model: ModelInterface):
        self._model = model

    def has_model(self) -> bool:
        return self._model is not None

    def init_state(
        self, prng_key: KeyArray, model_state: ModelState
    ) -> DetCountingKernelState:
        return self._initial_state

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: DetCountingKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> DetCountingKernelState:
        kernel_state.epoch_counter += 1
        kernel_state.in_epoch = True

        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: DetCountingKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> DetCountingKernelState:

        kernel_state.in_epoch = False
        return kernel_state

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: DetCountingKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[DetCountingKernelState, DetCountingTransInfo]:
        kernel_state.total_transitions += 1
        kernel_state.post_warmup_transitions += 1 * (
            1 - EpochType.is_warmup(epoch.config.type)
        )
        position = self.model.extract_position(self.position_keys, model_state)

        epoch_value = kernel_state.epoch_counter * 10000
        transition_value = kernel_state.increment_per_transition * epoch.time_in_epoch
        for pkey in position.keys():
            position[pkey] = jnp.asarray(epoch_value + transition_value)

        new_model_state = self.model.update_state(position, model_state)

        info = DetCountingTransInfo(0, 1.0, 1)
        return TransitionOutcome(info, kernel_state, new_model_state)

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: DetCountingKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[DetCountingKernelState, DetCountingTransInfo]:
        outcome = self._standard_transition(prng_key, kernel_state, model_state, epoch)
        outcome.kernel_state.total_adaptive_transitions += 1
        return outcome

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: DetCountingKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[DetCountingKernelState, DetCountingKernelTuningInfo]:
        kernel_state.tune_counter += 1
        info = DetCountingKernelTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: DetCountingKernelState,
        model_state: ModelState,
        tuning_history: DetCountingKernelTuningInfo | None,
    ) -> WarmupOutcome[DetCountingKernelState]:
        kernel_state.warmup_finialized = True
        return WarmupOutcome(0, kernel_state)
