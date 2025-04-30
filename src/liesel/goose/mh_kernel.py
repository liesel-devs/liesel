"""
Metroplis Hastings kernel. This kernel allows for a user-defined proposal functions and
adds the MH step. Optional, the kernel supports a stepsize adaptation.
"""

from collections.abc import Callable, Sequence
from typing import ClassVar, NamedTuple

import jax

from .da import da_finalize, da_init, da_step
from .epoch import EpochState
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    TransitionMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from .mh import mh_step
from .rw import RWKernelState
from .types import KeyArray, ModelState, Position, TuningInfo


class MHProposal(NamedTuple):
    position: Position
    log_correction: float
    """
    Let :math:`q(x' | x)` be the prosal density, then :math:`log(q(x'|x) / q(x | x'))`
    is the log_mh_correction.
    """


MHTransitionInfo = DefaultTransitionInfo
MHTuningInfo = DefaultTuningInfo
MHProposalFn = Callable[[KeyArray, ModelState, float], MHProposal]


class MHKernel(ModelMixin, TransitionMixin[RWKernelState, MHTransitionInfo]):
    """
    A Metropolis-Hastings kernel implementing the :class:`.Kernel` protocol.

    The user needs to provide a proposal function that proposes a new state and the
    log_correction.

    If ``da_tune_step_size`` is ``True`` the stepsize passed as an argument to the
    proposal function is tuned using the dual averging algorithm. Step size is tuned on
    the fly during all adaptive epochs.
    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors", 90: "nan acceptance prob"}
    """Dict of error codes and their meaning."""
    needs_history: ClassVar[bool] = False
    """Whether this kernel needs its history for tuning."""
    identifier: str = ""
    """Kernel identifier, set by :class:`.EngineBuilder`"""
    position_keys: tuple[str, ...]
    """Tuple of position keys handled by this kernel."""

    def __init__(
        self,
        position_keys: Sequence[str],
        proposal_fn: MHProposalFn,
        initial_step_size: float = 1.0,
        da_tune_step_size=False,
        da_target_accept: float = 0.234,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self._proposal_fn = proposal_fn
        self.initial_step_size = initial_step_size
        self.da_tune_step_size = da_tune_step_size
        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0

    def init_state(self, prng_key, model_state):
        """Initializes the kernel state."""

        return RWKernelState(step_size=self.initial_step_size)

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[RWKernelState, DefaultTransitionInfo]:
        """Performs an MCMC transition *without* dual averaging."""

        key, subkey = jax.random.split(prng_key)
        step_size = kernel_state.step_size

        # generate a proposal
        proposal = self._proposal_fn(key, model_state, step_size)

        # metropolis-hastings calibration
        info, model_state = mh_step(
            subkey,
            self.model,
            proposal.position,
            model_state,
            proposal.log_correction,
        )
        return TransitionOutcome(info, kernel_state, model_state)

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[RWKernelState, DefaultTransitionInfo]:
        """Performs an MCMC transition *with* dual averaging."""

        outcome = self._standard_transition(prng_key, kernel_state, model_state, epoch)

        if self.da_tune_step_size:
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
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[RWKernelState, DefaultTuningInfo]:
        """Currently does nothing."""

        info = MHTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> RWKernelState:
        """Resets the state of the dual averaging algorithm."""

        da_init(kernel_state)
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> RWKernelState:
        """
        Sets the step size as found by the dual averaging algorithm.
        """

        da_finalize(kernel_state)
        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        tuning_history: TuningInfo | None,
    ) -> WarmupOutcome[RWKernelState]:
        """Currently does nothing."""

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
