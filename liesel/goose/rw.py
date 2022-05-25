"""
# Random walk sampler
"""

from dataclasses import dataclass, field
from typing import ClassVar, Sequence

import jax
import jax.flatten_util

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
from .pytree import register_dataclass_as_pytree
from .types import KeyArray, ModelState, Position, TuningInfo


@register_dataclass_as_pytree
@dataclass
class RWKernelState:
    """
    A dataclass for the state of a `RWKernel`, implementing the
    `liesel.goose.da.DAKernelState` protocol.
    """

    step_size: float
    error_sum: float = field(default=0.0, init=False)
    log_avg_step_size: float = field(default=0.0, init=False)
    mu: float = field(init=False)

    def __post_init__(self):
        da_init(self)


RWTransitionInfo = DefaultTransitionInfo
RWTuningInfo = DefaultTuningInfo


class RWKernel(ModelMixin, TransitionMixin[RWKernelState, RWTransitionInfo]):
    """
    A random walk kernel with Gaussian proposals, Metropolis-Hastings correction and
    dual averaging, implementing the `liesel.goose.types.Kernel` protocol.

    The kernel uses a default Metropolis-Hastings target acceptance probability
    of 0.234, which is optimal for a random walk sampler (in a certain sense). See
    Gelman et al., [Weak convergence and optimal scaling of random walk Metropolis
    algorithms (1997)](https://doi.org/10.1214/aoap/1034625254).
    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors", 90: "nan acceptance prob"}
    needs_history: ClassVar[bool] = False
    identifier: str = ""

    def __init__(
        self,
        position_keys: Sequence[str],
        initial_step_size: float = 1.0,
        da_target_accept: float = 0.234,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self.initial_step_size = initial_step_size
        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0

    def init_state(self, prng_key, model_state):
        """
        Initializes the kernel state.
        """

        return RWKernelState(step_size=self.initial_step_size)

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[RWKernelState, DefaultTransitionInfo]:
        """
        Performs an MCMC transition *without* dual averaging.
        """

        key, subkey = jax.random.split(prng_key)
        step_size = kernel_state.step_size

        # random walk proposal
        position = self.position(model_state)
        flat_position, unravel_fn = jax.flatten_util.ravel_pytree(position)
        step = step_size * jax.random.normal(key, flat_position.shape)
        flat_proposal = flat_position + step
        proposal = unravel_fn(flat_proposal)

        # metropolis-hastings calibration
        info, model_state = mh_step(subkey, self.model, proposal, model_state)
        return TransitionOutcome(info, kernel_state, model_state)

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[RWKernelState, DefaultTransitionInfo]:
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
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[RWKernelState, DefaultTuningInfo]:
        """
        Currently does nothing.
        """

        info = RWTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: RWKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> RWKernelState:
        """
        Resets the state of the dual averaging algorithm.
        """

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
        """
        Currently does nothing.
        """

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
