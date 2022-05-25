"""
# Metropolis-Hastings

This module uses the error codes 90-99.
"""

import jax
import jax.numpy as jnp

from .kernel import DefaultTransitionInfo
from .types import KeyArray, ModelInterface, ModelState, Position

mh_error_book = {0: "no errors", 90: "nan acceptance prob"}
"""The error book of the `mh_step` function."""


def mh_step(
    prng_key: KeyArray,
    model: ModelInterface,
    proposal: Position,
    model_state: ModelState,
    correction: float = 0.0,
) -> tuple[DefaultTransitionInfo, ModelState]:
    """
    Decides if an MCMC proposal is accepted in a Metropolis-Hastings step.

    ## Parameters

    - `correction`: The Metropolis-Hastings correction in the case of an asymmetric
      proposal distribution.

    ## Returns

    A tuple of a `liesel.goose.types.TransitionInfo` and a
    `liesel.goose.types.ModelState` (= a pytree).
    """

    current_log_prob = model.log_prob(model_state)
    proposed_model_state = model.update_state(proposal, model_state)
    proposed_log_prob = model.log_prob(proposed_model_state)

    log_acc_prob = proposed_log_prob - current_log_prob + correction

    log_acc_prob, error_code = jax.lax.cond(
        jnp.isnan(log_acc_prob),
        lambda: (-jnp.inf, 90),
        lambda: (log_acc_prob, 0),
    )

    acceptance_prob = jnp.clip(jnp.exp(log_acc_prob), a_max=1.0)
    do_accept = jax.random.uniform(prng_key) <= acceptance_prob

    info = DefaultTransitionInfo(error_code, acceptance_prob, do_accept)

    model_state = jax.lax.cond(
        do_accept,
        lambda: proposed_model_state,
        lambda: model_state,
    )

    return info, model_state
