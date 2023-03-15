from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from liesel.goose.epoch import EpochState
from liesel.goose.kernel import (
    DefaultTransitionInfo,
    ModelMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from liesel.goose.pytree import register_dataclass_as_pytree
from liesel.goose.types import Kernel, KeyArray, ModelInterface, ModelState, Position


@register_dataclass_as_pytree
@dataclass
class MockKernelState:
    pass

    @staticmethod
    def default() -> "MockKernelState":
        return MockKernelState()


@register_dataclass_as_pytree
@dataclass
class MockKernelTuningInfo:
    error_code: int
    time: int


@register_dataclass_as_pytree
@dataclass
class MockKernelTransInfo:
    error_code: int = 0
    acceptance_prob: float = 1
    position_moved: int = 1

    def minimize(self) -> DefaultTransitionInfo:
        return DefaultTransitionInfo(
            self.error_code, self.acceptance_prob, self.position_moved
        )


class MockKernel(ModelMixin):
    error_book: ClassVar[dict[int, str]] = {0: "no errors", 1: "error 1", 2: "error 2"}
    needs_history: ClassVar[bool] = False
    identifier: str = ""

    def __init__(
        self,
        position_keys: Sequence[str],
    ):
        self._model: None | ModelInterface = None
        self.position_keys = tuple(position_keys)

    def init_state(
        self, prng_key: KeyArray, model_state: ModelState
    ) -> MockKernelState:
        return MockKernelState.default()

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> MockKernelState:
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> MockKernelState:
        return kernel_state

    def transition(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[MockKernelState, MockKernelTransInfo]:
        position = self.model.extract_position(self.position_keys, model_state)
        for key in position:
            position[key] = position[key] + 1.0

        new_model_state = self.model.update_state(position, model_state)

        info = MockKernelTransInfo()
        return TransitionOutcome(info, kernel_state, new_model_state)

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[MockKernelState, MockKernelTuningInfo]:
        info = MockKernelTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        tuning_history: MockKernelTuningInfo | None,
    ) -> WarmupOutcome[MockKernelState]:
        return WarmupOutcome(0, kernel_state)


def typecheck() -> None:
    _: Kernel[MockKernelState, MockKernelTransInfo, MockKernelTuningInfo] = MockKernel(
        ["foo", "bar"]
    )
