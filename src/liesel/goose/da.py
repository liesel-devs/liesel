"""
Dual averaging.

This module uses the error codes 80-89.
"""

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp

from .pytree import register_dataclass_as_pytree


@register_dataclass_as_pytree
@dataclass
class DualAvgState:
    """
    The state of the dual averaging algorithm.
    """

    error_sum: float
    """The error sum of the acceptance probability."""

    log_avg_step_size: float
    """The logarithm of the average step size."""

    mu: float
    """The bias of the step size proposals."""

    @classmethod
    def from_step_size(cls, step_size: float) -> "DualAvgState":
        """Initializes a dual averaging state for ``step_size``."""
        return cls(
            error_sum=0.0,
            log_avg_step_size=jnp.log(step_size),
            mu=jnp.log(10.0 * step_size),
        )


class DAKernelState(Protocol):
    """
    A protocol for a kernel state with dual averaging support. For an introduction
    to dual averaging, see the blog post by Colin Carroll [#carroll]_ and the Stan
    Reference Manual [#stan]_.

    .. [#carroll] `Colin Carroll, Step Size Adaptation in Hamiltonian Monte Carlo (2019)
       <https://colindcarroll.com/blog/step_size_adapt_hmc.html>`_.
    .. [#stan] `Stan Development Team, Stan Reference Manual (2021), Chapter 15.2
       <https://mc-stan.org/docs/2_28/reference-manual/hmc-algorithm-parameters.html>`_.
    """

    step_size: float
    """The step size of the kernel."""

    da_state: DualAvgState | None
    """The internal state of the dual averaging algorithm."""


def da_init(kernel_state: DAKernelState) -> None:
    """
    Initializes (or resets) a :class:`.DAKernelState`. Returns ``None`` and should be
    called for the side effect on the ``kernel_state`` argument.
    """

    kernel_state.da_state = DualAvgState.from_step_size(kernel_state.step_size)


def da_step(
    kernel_state: DAKernelState,
    acceptance_prob: float,
    time_in_epoch: int,
    target_accept: float = 0.8,
    gamma: float = 0.05,
    kappa: float = 0.75,
    t0: int = 10,
) -> None:
    """
    Performs an dual averaging update on a :class:`.DAKernelState`. Returns ``None``
    and should be called for the side effect on the ``kernel_state`` argument.

    Parameters
    ----------
    kernel_state
        A kernel state implementing the :class:`.DAKernelState` protocol.
    acceptance_prob
        The acceptance probability of this MCMC iteration.
    time_in_epoch
        The number of completed MCMC iterations in this epoch.
    target_accept
        The target acceptance probability.
    gamma
        The adaptation regularization scale.
    kappa
        The adaptation relaxation exponent.
    t0
        The adaptation iteration offset.

    Notes
    -----

    For an introduction
    to dual averaging, see the blog post by Colin Carroll [#carroll]_ and the Stan
    Reference Manual [#stan]_.

    .. [#carroll] `Colin Carroll, Step Size Adaptation in Hamiltonian Monte Carlo (2019)
       <https://colindcarroll.com/blog/step_size_adapt_hmc.html>`_.
    .. [#stan] `Stan Development Team, Stan Reference Manual (2021), Chapter 15.2
       <https://mc-stan.org/docs/2_28/reference-manual/hmc-algorithm-parameters.html>`_.
    """

    ks = kernel_state
    t = time_in_epoch + 1
    eta = t ** (-kappa)

    da_state = ks.da_state
    if da_state is None:
        raise RuntimeError("Dual averaging state has not been initialized.")

    da_state.error_sum += target_accept - acceptance_prob
    log_step_size = da_state.mu - (da_state.error_sum * jnp.sqrt(t)) / (
        gamma * (t0 + t)
    )
    ks.step_size = jnp.exp(log_step_size)

    log_avg_step_size = (1 - eta) * da_state.log_avg_step_size + eta * log_step_size
    da_state.log_avg_step_size = log_avg_step_size


def da_finalize(kernel_state: DAKernelState) -> None:
    """
    Sets the new step size in a :class:`.DAKernelState`. Returns ``None`` and should be
    called for the side effect on the ``kernel_state`` argument.
    """

    da_state = kernel_state.da_state
    if da_state is None:
        raise RuntimeError("Dual averaging state has not been initialized.")

    kernel_state.step_size = jnp.exp(da_state.log_avg_step_size)
