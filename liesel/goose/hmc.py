"""
# Hamiltonian/Hybrid Monte Carlo (HMC)
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, ClassVar, Sequence

import jax.numpy as jnp
from blackjax.adaptation.step_size import find_reasonable_step_size
from blackjax.mcmc import hmc
from jax.flatten_util import ravel_pytree

from .da import da_finalize, da_init, da_step
from .epoch import EpochState
from .kernel import (
    DefaultTransitionInfo,
    DefaultTuningInfo,
    ModelMixin,
    TransitionMixin,
    TransitionOutcome,
    TuningMixin,
    TuningOutcome,
    WarmupOutcome,
)
from .mm import tune_inv_mm_diag, tune_inv_mm_full
from .pytree import register_dataclass_as_pytree
from .types import Array, KeyArray, ModelState, Position


@register_dataclass_as_pytree
@dataclass
class HMCKernelState:
    """
    A dataclass for the state of a `HMCKernel`, implementing the
    `liesel.goose.da.DAKernelState` protocol.
    """

    step_size: float
    inverse_mass_matrix: Array
    error_sum: float = field(init=False)
    log_avg_step_size: float = field(init=False)
    mu: float = field(init=False)

    def __post_init__(self):
        da_init(self)


@register_dataclass_as_pytree
@dataclass
class HMCTransitionInfo(DefaultTransitionInfo):
    divergent: bool
    """
    Whether the difference in energy between the original and the new state exceeded
    the divergence threshold of 1000.
    """


def _goose_info(hmc_info: hmc.HMCInfo) -> HMCTransitionInfo:
    error_code = 1 * hmc_info.is_divergent
    acceptance_prob = hmc_info.acceptance_probability
    position_moved = hmc_info.is_accepted

    return HMCTransitionInfo(
        error_code,
        acceptance_prob,
        position_moved,
        hmc_info.is_divergent,
    )


HMCTuningInfo = DefaultTuningInfo


class HMCKernel(
    ModelMixin,
    TransitionMixin[HMCKernelState, HMCTransitionInfo],
    TuningMixin[HMCKernelState, HMCTuningInfo],
):
    """
    A HMC kernel with dual averaging and an inverse mass matrix tuner,
    implementing the `liesel.goose.types.Kernel` protocol.
    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors", 1: "divergent transition"}
    needs_history: ClassVar[bool] = True
    identifier: str = ""

    def __init__(
        self,
        position_keys: Sequence[str],
        initial_step_size: float | None = None,
        initial_inverse_mass_matrix: Array | None = None,
        num_integration_steps: int = 10,
        da_target_accept: float = 0.8,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
        mm_diag: bool = True,
    ):
        self.position_keys = tuple(position_keys)
        self._model = None

        self.initial_step_size = initial_step_size
        self.initial_inverse_mass_matrix = initial_inverse_mass_matrix
        self.num_integration_steps = num_integration_steps

        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0

        self.mm_diag = mm_diag

    def _blackjax_state(self, model_state: ModelState) -> hmc.HMCState:
        return hmc.init(self.position(model_state), self.log_prob_fn(model_state))

    def _blackjax_kernel(self) -> Callable:
        return hmc.kernel()

    def init_state(self, prng_key, model_state):
        """
        Initializes the kernel state with an identity inverse mass matrix
        and a reasonable step size (unless explicit arguments were provided
        by the user).
        """

        if self.initial_inverse_mass_matrix is None:
            flat_position, _ = ravel_pytree(self.position(model_state))

            if self.mm_diag:
                inverse_mass_matrix = jnp.ones_like(flat_position)
            else:
                inverse_mass_matrix = jnp.eye(flat_position.size)
        else:
            inverse_mass_matrix = self.initial_inverse_mass_matrix

        if self.initial_step_size is None:
            blackjax_kernel = self._blackjax_kernel()
            blackjax_state = self._blackjax_state(model_state)
            log_prob_fn = self.log_prob_fn(model_state)

            def kernel_generator(step_size: float) -> Callable:
                return partial(
                    blackjax_kernel,
                    logprob_fn=log_prob_fn,
                    step_size=step_size,
                    inverse_mass_matrix=inverse_mass_matrix,
                    num_integration_steps=self.num_integration_steps,
                )

            step_size = find_reasonable_step_size(
                prng_key,
                kernel_generator,
                blackjax_state,
                initial_step_size=0.001,
                target_accept=self.da_target_accept,
            )
        else:
            step_size = self.initial_step_size

        return HMCKernelState(step_size, inverse_mass_matrix)

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: HMCKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[HMCKernelState, HMCTransitionInfo]:
        """
        Performs an MCMC transition *without* dual averaging.
        """

        blackjax_kernel = self._blackjax_kernel()
        blackjax_state = self._blackjax_state(model_state)
        log_prob_fn = self.log_prob_fn(model_state)

        blackjax_state, blackjax_info = blackjax_kernel(
            prng_key,
            blackjax_state,
            log_prob_fn,
            kernel_state.step_size,
            kernel_state.inverse_mass_matrix,
            self.num_integration_steps,
        )

        info = _goose_info(blackjax_info)
        model_state = self.model.update_state(blackjax_state.position, model_state)
        return TransitionOutcome(info, kernel_state, model_state)

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: HMCKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[HMCKernelState, HMCTransitionInfo]:
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

    def _tune_fast(
        self,
        prng_key: KeyArray,
        kernel_state: HMCKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[HMCKernelState, HMCTuningInfo]:
        """
        Currently does nothing.
        """

        info = HMCTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def _tune_slow(
        self,
        prng_key: KeyArray,
        kernel_state: HMCKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[HMCKernelState, HMCTuningInfo]:
        """
        Tunes the inverse mass vector or matrix using the samples from the last epoch.
        """

        if history is not None:
            history = Position({k: history[k] for k in self.position_keys})

            if self.mm_diag:
                new_inv_mm = tune_inv_mm_diag(history)
                trace_fn = jnp.sum
            else:
                new_inv_mm = tune_inv_mm_full(history)
                trace_fn = jnp.trace

            old_inv_mm = kernel_state.inverse_mass_matrix
            adjustment = jnp.sqrt(trace_fn(old_inv_mm) / trace_fn(new_inv_mm))
            kernel_state.step_size = adjustment * kernel_state.step_size

            kernel_state.inverse_mass_matrix = new_inv_mm

        return self._tune_fast(prng_key, kernel_state, model_state, epoch, history)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: HMCKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> HMCKernelState:
        """
        Resets the state of the dual averaging algorithm.
        """

        da_init(kernel_state)
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: HMCKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> HMCKernelState:
        """
        Sets the step size as found by the dual averaging algorithm.
        """

        da_finalize(kernel_state)
        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: HMCKernelState,
        model_state: ModelState,
        tuning_history: HMCTuningInfo | None,
    ) -> WarmupOutcome[HMCKernelState]:
        """
        Currently does nothing.
        """

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
