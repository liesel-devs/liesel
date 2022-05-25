"""
# Gibbs sampler
"""

from typing import Callable, ClassVar, Sequence

from .epoch import EpochState
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from .types import Kernel, KernelState, KeyArray, ModelState, Position, TuningInfo

GibbsKernelState = KernelState
GibbsTransitionInfo = DefaultTransitionInfo
GibbsTuningInfo = DefaultTuningInfo


class GibbsKernel(
    ModelMixin, Kernel[GibbsKernelState, GibbsTransitionInfo, GibbsTuningInfo]
):
    """
    A Gibbs kernel implementing the `liesel.goose.types.Kernel` protocol.
    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors"}
    needs_history: ClassVar[bool] = False
    identifier: str = ""

    def __init__(
        self,
        position_keys: Sequence[str],
        transition_fn: Callable[[KeyArray, ModelState], Position],
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self._transition_fn = transition_fn

    def init_state(self, prng_key, model_state):
        """
        Initializes an (empty) kernel state.
        """

        return {}

    def transition(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[KernelState, GibbsTransitionInfo]:
        """
        Performs an MCMC transition.
        """

        info = GibbsTransitionInfo(
            error_code=0,
            acceptance_prob=1.0,
            position_moved=1,
        )

        position = self._transition_fn(prng_key, model_state)
        model_state = self.model.update_state(position, model_state)
        return TransitionOutcome(info, kernel_state, model_state)

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[KernelState, GibbsTuningInfo]:
        """
        Currently does nothing.
        """

        info = GibbsTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KernelState:
        """
        Currently does nothing.
        """

        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> KernelState:
        """
        Currently does nothing.
        """

        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: KernelState,
        model_state: ModelState,
        tuning_history: TuningInfo | None,
    ) -> WarmupOutcome[KernelState]:
        """
        Currently does nothing.
        """

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
