"""
# Dual averaging

This module uses the error codes 80-89.
"""

from typing import Protocol

import jax.numpy as jnp


class DAKernelState(Protocol):
    """
    A protocol for a kernel state with dual averaging support. For an introduction to
    dual averaging, see:

    - Colin Carroll, [Step Size Adaptation in Hamiltonian Monte Carlo (2019)](
      https://colindcarroll.com/2019/04/21/step-size-adaptation-in-hamiltonian-monte-carlo).
    - Stan Development Team, [Stan Reference Manual (2021), Chapter 15.2](
      https://mc-stan.org/docs/2_28/reference-manual/hmc-algorithm-parameters.html).
    """

    step_size: float
    """The step size of the kernel."""

    error_sum: float
    """The error sum of the acceptance probability. Should not be set by the user,
    but is used by the `da_step` function."""

    log_avg_step_size: float
    """The logarithm of the average step size. Should not be set by the user, but is
    used by the `da_step` function."""

    mu: float
    """The bias of the step size proposals. Should not be set by the user, but is
    used by the `da_step` function."""


def da_init(kernel_state: DAKernelState) -> None:
    """
    Initializes (or resets) a `DAKernelState`. Returns `None` and should be called for
    the side effect on the `kernel_state` argument.
    """

    kernel_state.error_sum = 0.0
    kernel_state.log_avg_step_size = jnp.log(kernel_state.step_size)
    kernel_state.mu = jnp.log(10.0 * kernel_state.step_size)


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
    Performs an dual averaging update on a `DAKernelState`. Returns `None` and should
    be called for the side effect on the `kernel_state` argument.

    ## Parameters

    - `kernel_state`: A kernel state implementing the `DAKernelState` protocol.
    - `acceptance_prob`: The acceptance probability of this MCMC iteration.
    - `time_in_epoch`: The number of completed MCMC iterations in this epoch.
    - `target_accept`: The target acceptance probability.
    - `gamma`: The adaptation regularization scale.
    - `kappa`: The adaptation relaxation exponent.
    - `t0`: The adaptation iteration offset.
    """

    ks = kernel_state
    t = time_in_epoch + 1
    eta = t ** (-kappa)

    ks.error_sum += target_accept - acceptance_prob
    log_step_size = ks.mu - (ks.error_sum * jnp.sqrt(t)) / (gamma * (t0 + t))
    ks.step_size = jnp.exp(log_step_size)

    ks.log_avg_step_size = (1 - eta) * ks.log_avg_step_size + eta * log_step_size


def da_finalize(kernel_state: DAKernelState) -> None:
    """
    Sets the new step size in a `DAKernelState`. Returns `None` and should be called
    for the side effect on the `kernel_state` argument.
    """

    kernel_state.step_size = jnp.exp(kernel_state.log_avg_step_size)
