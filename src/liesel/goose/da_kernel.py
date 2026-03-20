"""
General kernel with step size tuning via dual averaging.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from .da import da_finalize, da_init, da_step
from .epoch import EpochState
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    ReprMixin,
    TransitionMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from .pytree import register_dataclass_as_pytree
from .types import KeyArray, ModelState, Position, TuningInfo


@register_dataclass_as_pytree
@dataclass
class DAKernelState:
    """
    A dataclass for the state of a ``RWKernel``, implementing the
    :class:`.DAKernelState` protocol.
    """

    step_size: float
    error_sum: float = field(default=0.0, init=False)
    log_avg_step_size: float = field(default=0.0, init=False)
    mu: float = field(init=False)

    def __post_init__(self):
        da_init(self)


DATransitionInfo = DefaultTransitionInfo
DATuningInfo = DefaultTuningInfo


class DAKernel(ModelMixin, TransitionMixin[DAKernelState, DATransitionInfo], ReprMixin):
    """

    Parameters
    ----------
    position_keys
        Sequence of position keys (variable names) handled by this kernel.
    initial_step_size
        Value at which to start step size tuning.
    da_target_accept
        Target acceptance probability for dual averaging algorithm.
    da_gamma
        The adaptation regularization scale.
    da_kappa
        The adaptation relaxation exponent.
    da_t0
        The adaptation iteration offset.
    identifier
        An string acting as a unique identifier for this kernel.

    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors", 90: "nan acceptance prob"}
    """Dict of error codes and their meaning."""
    needs_history: ClassVar[bool] = False
    """Whether this kernel needs its history for tuning."""
    identifier: str = ""
    """Kernel identifier, set by :class:`~.goose.EngineBuilder`"""
    position_keys: tuple[str, ...]
    """Tuple of position keys handled by this kernel."""

    def __init__(
        self,
        position_keys: Sequence[str],
        initial_step_size: float,
        da_target_accept: float,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
        identifier: str = "",
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self.initial_step_size = initial_step_size
        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0
        self.identifier = identifier

    def init_state(self, prng_key, model_state):
        """
        Initializes the kernel state.
        """

        return DAKernelState(step_size=self.initial_step_size)

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: DAKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[DAKernelState, DefaultTransitionInfo]:
        """
        Performs an MCMC transition *without* dual averaging.
        """
        raise NotImplementedError

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: DAKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[DAKernelState, DefaultTransitionInfo]:
        """
        Performs an MCMC transition *with* dual averaging.
        """

        outcome = self._standard_transition(prng_key, kernel_state, model_state, epoch)

        da_step(
            outcome.kernel_state,
            outcome.info.acceptance_prob,
            epoch.time_in_epoch,
            self.da_target_accept,
            self.da_gamma,
            self.da_kappa,
            self.da_t0,
        )

        return outcome

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: DAKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[DAKernelState, DefaultTuningInfo]:
        """
        Currently does nothing.
        """

        info = DATuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: DAKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> DAKernelState:
        """
        Resets the state of the dual averaging algorithm.
        """

        da_init(kernel_state)
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: DAKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> DAKernelState:
        """
        Sets the step size as found by the dual averaging algorithm.
        """

        da_finalize(kernel_state)
        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: DAKernelState,
        model_state: ModelState,
        tuning_history: TuningInfo | None,
    ) -> WarmupOutcome[DAKernelState]:
        """
        Currently does nothing.
        """

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
