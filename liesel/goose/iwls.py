"""
# Iteratively weighted least squares (IWLS) sampler
"""

from dataclasses import dataclass, field
from typing import Callable, ClassVar, Sequence

import jax
import jax.numpy as jnp
from jax import grad, jacfwd
from jax.flatten_util import ravel_pytree

from .da import da_finalize, da_init, da_step
from .epoch import EpochState
from .iwls_utils import mvn_log_prob, mvn_sample, solve
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
from .types import Array, KeyArray, ModelState, Position


@register_dataclass_as_pytree
@dataclass
class IWLSKernelState:
    """
    A dataclass for the state of a `IWLSKernel`, implementing the
    `liesel.goose.da.DAKernelState` protocol.
    """

    step_size: float
    error_sum: float = field(init=False)
    log_avg_step_size: float = field(init=False)
    mu: float = field(init=False)

    def __post_init__(self):
        da_init(self)


IWLSTransitionInfo = DefaultTransitionInfo
IWLSTuningInfo = DefaultTuningInfo


class IWLSKernel(ModelMixin, TransitionMixin[IWLSKernelState, IWLSTransitionInfo]):
    """
    An IWLS kernel with dual averaging and an (optional) user-defined function
    for computing the Cholesky decomposition of the Fisher information matrix,
    implementing the `liesel.goose.types.Kernel` protocol.
    """

    error_book: ClassVar[dict[int, str]] = {0: "no errors", 90: "nan acceptance prob"}
    needs_history: ClassVar[bool] = False
    identifier: str = ""

    def __init__(
        self,
        position_keys: Sequence[str],
        chol_info_fn: Callable[[ModelState], Array] | None = None,
        initial_step_size: float = 0.01,
        da_target_accept: float = 0.8,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
    ):
        self._model = None
        self.position_keys = tuple(position_keys)
        self.chol_info_fn = chol_info_fn

        self.initial_step_size = initial_step_size

        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0

    def _flat_log_prob_fn(
        self, model_state: ModelState, unravel_fn: Callable[[Array], Position]
    ) -> Callable[[Array], float]:
        """
        Returns a callable which takes a flat position and returns the log-probability
        of the model.
        """

        def flat_log_prob_fn(flat_position: Array) -> float:
            position = unravel_fn(flat_position)
            new_model_state = self.model.update_state(position, model_state)
            return self.model.log_prob(new_model_state)

        return flat_log_prob_fn

    def _score(
        self, model_state: ModelState, flat_score_fn: Callable[[Array], Array]
    ) -> Array:
        """
        Calls `flat_score_fn` on a flat position.

        The flat position is extracted from the `model_state`.
        """

        flat_position, _ = ravel_pytree(self.position(model_state))
        return flat_score_fn(flat_position)

    def _chol_info(
        self, model_state: ModelState, flat_hessian_fn: Callable[[Array], Array]
    ) -> Array:
        """
        Computes the Cholesky decomposition of the Fisher information matrix
        via `flat_hessian_fn`.

        The flat position is extracted from the `model_state`. If the user provided a
        `chol_info_fn` when initializing the kernel, this function is called instead.
        """

        if self.chol_info_fn is None:
            flat_position, _ = ravel_pytree(self.position(model_state))
            return jnp.linalg.cholesky(-flat_hessian_fn(flat_position))

        return self.chol_info_fn(model_state)

    def init_state(self, prng_key, model_state):
        """
        Initializes the kernel state.
        """

        return IWLSKernelState(self.initial_step_size)

    def _standard_transition(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[IWLSKernelState, IWLSTransitionInfo]:
        """
        Performs an MCMC transition *without* dual averaging.
        """

        key, subkey = jax.random.split(prng_key)
        step_size = kernel_state.step_size

        flat_pos, unravel_fn = ravel_pytree(self.position(model_state))
        flat_log_prob_fn = self._flat_log_prob_fn(model_state, unravel_fn)
        flat_score_fn = grad(flat_log_prob_fn)
        flat_hessian_fn = jacfwd(flat_score_fn)

        # proposal and forward probability

        score_pos = self._score(model_state, flat_score_fn)
        chol_info_pos = self._chol_info(model_state, flat_hessian_fn)
        mu_pos = flat_pos + ((step_size**2) / 2) * solve(chol_info_pos, score_pos)
        flat_prop = mvn_sample(key, mu_pos, chol_info_pos / step_size)
        proposal = unravel_fn(flat_prop)

        fwd_log_prob = mvn_log_prob(flat_prop, mu_pos, chol_info_pos / step_size)

        # backward probability

        model_state_prop = self.model.update_state(proposal, model_state)

        score_prop = self._score(model_state_prop, flat_score_fn)
        chol_info_prop = self._chol_info(model_state_prop, flat_hessian_fn)
        mu_prop = flat_prop + ((step_size**2) / 2) * solve(chol_info_prop, score_prop)
        bwd_log_prob = mvn_log_prob(flat_pos, mu_prop, chol_info_prop / step_size)

        correction = bwd_log_prob - fwd_log_prob

        info, model_state = mh_step(
            subkey, self.model, proposal, model_state, correction
        )

        return TransitionOutcome(info, kernel_state, model_state)

    def _adaptive_transition(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[IWLSKernelState, IWLSTransitionInfo]:
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
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None = None,
    ) -> TuningOutcome[IWLSKernelState, IWLSTuningInfo]:
        """
        Currently does nothing.
        """

        info = IWLSTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> IWLSKernelState:
        """
        Resets the state of the dual averaging algorithm.
        """

        da_init(kernel_state)
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> IWLSKernelState:
        """
        Sets the step size as found by the dual averaging algorithm.
        """

        da_finalize(kernel_state)
        return kernel_state

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: IWLSKernelState,
        model_state: ModelState,
        tuning_history: IWLSTuningInfo | None,
    ) -> WarmupOutcome[IWLSKernelState]:
        """
        Currently does nothing.
        """

        return WarmupOutcome(error_code=0, kernel_state=kernel_state)
