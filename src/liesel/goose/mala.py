"""Metropolis-adjusted Langevin samplers."""

from collections.abc import Callable, Sequence
from math import isfinite
from types import SimpleNamespace
from typing import Self

import jax.numpy as jnp
from blackjax.adaptation.step_size import find_reasonable_step_size
from jax.flatten_util import ravel_pytree

from .da import da_finalize, da_init, da_step
from .epoch import EpochConfig, EpochState, EpochType
from .iwls import (
    CholInfoFallbackOptions,
    IWLSKernelState,
    IWLSTransitionInfo,
    _GaussianKernel,
)
from .iwls_utils import solve
from .kernel import TransitionOutcome
from .mh import mh_error_book
from .types import Array, KeyArray, ModelState

MALAKernelState = IWLSKernelState


class _LangevinKernel(_GaussianKernel):
    """Shared Langevin scaling, initial-step search, and adaptation."""

    def __init__(
        self,
        position_keys: Sequence[str],
        chol_info_fn: Callable[[ModelState], Array] | None = None,
        initial_step_size: float | None = None,
        da_tune_step_size: bool = True,
        da_target_accept: float = 0.8,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
        identifier: str = "",
        fallback_chol_info: CholInfoFallbackOptions | None = "identity",
    ):
        if initial_step_size is not None and (
            not isfinite(initial_step_size) or initial_step_size <= 0.0
        ):
            raise ValueError("initial_step_size must be finite and positive, or None")
        if not 0.0 < da_target_accept < 1.0:
            raise ValueError("da_target_accept must lie strictly between 0 and 1")
        super().__init__(
            position_keys,
            chol_info_fn,
            identifier=identifier,
            fallback_chol_info=fallback_chol_info,
        )
        self.initial_step_size = initial_step_size

        self.da_tune_step_size = da_tune_step_size
        self.da_target_accept = da_target_accept
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.da_t0 = da_t0

    def init_state(self, prng_key, model_state):
        """Initialize the Langevin step size and dual averaging state."""
        if self.initial_step_size is not None:
            return MALAKernelState(self.initial_step_size)

        epoch = EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None).to_state(0, 0)

        def kernel_generator(step_size):
            def kernel(key, reference_state):
                outcome = self._standard_transition(
                    key, MALAKernelState(step_size), reference_state, epoch
                )
                info = SimpleNamespace(acceptance_rate=outcome.info.acceptance_prob)
                return outcome.model_state, info

            return kernel

        # Evaluate our actual proposal: s is a standard-deviation scale, and
        # SMMALA must use its configured metric in both proposal directions.
        step_size = find_reasonable_step_size(
            prng_key,
            kernel_generator,
            model_state,
            initial_step_size=0.001,
            target_accept=self.da_target_accept,
        )
        return MALAKernelState(step_size)

    def _proposal_parameters(self, position, score, chol_info, kernel_state):
        step_size = kernel_state.step_size
        mean = position + ((step_size**2) / 2) * solve(chol_info, score)
        return mean, chol_info / step_size

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


class SMMALAKernel(_LangevinKernel):
    r"""Simplified manifold MALA using a regularized posterior precision.

    The proposal mean is beta + (s^2 / 2) P^{-1} g and its covariance is
    s^2 P^{-1}. The metric may depend on position; metric derivatives are
    omitted. Both proposal directions are evaluated for the MH correction.

    The constructor retains the argument order of the pre-0.6 IWLS kernel.

    Parameters
    ----------
    position_keys
        Names of the variables sampled together.
    chol_info_fn
        Optional function returning the lower Cholesky factor of the precision.
        The default is the negative block Hessian of the log posterior with
        ``1e-6 * mean(diag(P)) * I`` added before factorization.
    initial_step_size
        Positive standard-deviation scale s. With ``None`` (the default), search
        for an initial step using this kernel's proposal and acceptance target.
        The search does not advance the chain. Pass ``0.01`` to reproduce the
        default initialization of the pre-0.6 IWLS kernel.
    da_tune_step_size
        Adapt the step size during warmup, default True. If False, retain the
        supplied or automatically selected initial scale.
    da_target_accept
        Target acceptance probability, default 0.8.
    da_gamma
        Dual-averaging regularization scale, default 0.05.
    da_kappa
        Dual-averaging relaxation exponent, default 0.75.
    da_t0
        Dual-averaging iteration offset, default 10.
    identifier
        Unique identifier, normally assigned by :class:`.EngineBuilder`.
    fallback_chol_info
        Precision fallback, as described in :class:`.IWLSKernel`.

    Notes
    -----
    A constant custom precision gives preconditioned MALA. Use
    :class:`.MALAKernel` for identity precision without Hessian computation.
    This is simplified manifold MALA: it does not include metric-derivative
    drift terms. See Girolami and Calderhead (2011),
    https://doi.org/10.1111/j.1467-9868.2010.00765.x.
    """

    @classmethod
    def untuned(
        cls,
        position_keys: Sequence[str],
        chol_info_fn: Callable[[ModelState], Array] | None = None,
        fallback_chol_info: CholInfoFallbackOptions | None = "identity",
    ) -> Self:
        """
        Fix the step size to 1, without initial-step search or warmup adaptation.
        """
        kernel = cls(
            position_keys=position_keys,
            chol_info_fn=chol_info_fn,
            initial_step_size=1.0,
            da_tune_step_size=False,
            fallback_chol_info=fallback_chol_info,
        )
        return kernel


class MALAKernel(_LangevinKernel):
    r"""MALA with identity precision, using only first derivatives.

    The proposal mean is beta + (s^2 / 2) g and its covariance is s^2 I.
    The default dual-averaging target is 0.574, a heuristic from the asymptotic
    MALA scaling result, not a universal optimum.

    Parameters
    ----------
    position_keys
        Names of the variables sampled together.
    initial_step_size
        Positive standard-deviation scale s, or ``None`` (default) for automatic
        initial-step selection. The proposal covariance is s^2 I.
    da_tune_step_size
        Adapt the step during warmup, default True. If False, retain the supplied
        or automatically selected initial scale.
    da_target_accept
        Target acceptance probability, default 0.574.
    da_gamma
        Dual-averaging regularization scale, default 0.05.
    da_kappa
        Dual-averaging relaxation exponent, default 0.75.
    da_t0
        Dual-averaging iteration offset, default 10.
    identifier
        Unique identifier, normally assigned by :class:`.EngineBuilder`.

    Notes
    -----
    For a fixed preconditioner or position-dependent metric, use
    :class:`.SMMALAKernel` with a custom ``chol_info_fn``. For the acceptance
    heuristic, see Roberts and Rosenthal (1998),
    https://doi.org/10.1111/1467-9868.00123.
    """

    error_book = mh_error_book

    def __init__(
        self,
        position_keys: Sequence[str],
        initial_step_size: float | None = None,
        da_tune_step_size: bool = True,
        da_target_accept: float = 0.574,
        da_gamma: float = 0.05,
        da_kappa: float = 0.75,
        da_t0: int = 10,
        identifier: str = "",
    ):
        super().__init__(
            position_keys,
            initial_step_size=initial_step_size,
            da_tune_step_size=da_tune_step_size,
            da_target_accept=da_target_accept,
            da_gamma=da_gamma,
            da_kappa=da_kappa,
            da_t0=da_t0,
            identifier=identifier,
        )

    def _chol_info(self, model_state, flat_hessian_fn):
        position, _ = ravel_pytree(self.position(model_state))
        # Scalar identity avoids allocating and solving dense matrices for MALA.
        return jnp.array(1.0, dtype=position.dtype), 0

    @classmethod
    def untuned(cls, position_keys: Sequence[str]) -> Self:
        """Use a fixed unit step size without adaptation."""
        return cls(position_keys, initial_step_size=1.0, da_tune_step_size=False)
