from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .types import Array


@dataclass
class Stopper:
    """
    Handles (early) stopping..

    Parameters
    ----------
    epochs
        The maximum number of optimization epochs.
    patience
        Early stopping happens only, if there was no improvement for the number of\
        patience epochs, and there were at least as many epochs as the length\
        of the patience window.
    atol
        The absolute tolerance for early stopping.
    rtol
        The relative tolerance for early stopping. The default of ``0.0`` means that \
        no early stopping happens based on the relative tolerance.

    Notes
    -----
    Early stopping happens, when the oldest loss value within the patience window is
    the best loss value within the patience window. A simplified pseudo-implementation
    is:

    .. code-block:: python

        def stop(patience, i, loss_history):
            recent_history = loss_history[-patience:]
            oldest_within_patience = recent_history[0]
            best_within_patience = np.min(recent_history)

            return oldest_within_patience <= best_within_patience

    Absolute and relative tolerance make it possible to stop even in cases when the
    oldest loss within patience is *not* the best. Instead, the algorithm stops, when
    the absolute *or* relative difference between the oldest loss within patience and
    the best loss within patience is so small that it can be neglected.
    To be clear: If either of the two conditions is met, then early stopping happens.
    The relative magnitude of the difference is calculated with respect to the best
    loss within patience. A simplified pseudo-implementation is:

    .. code-block:: python

        def stop(patience, i, loss_history, atol, rtol):
            recent_history = loss_history[-patience:]
            oldest_within_patience = recent_history[0]
            best_within_patience = np.min(recent_history)

            diff = oldest_within_patience - best_within_patience
            rel_diff = diff / np.abs(best_within_patience)

            abs_improvement_is_neglectable = diff <= atol
            rel_improvement_is_neglectable = rel_diff <= rtol

            return (abs_improvement_is_neglectable | rel_improvement_is_neglectable)

    """

    epochs: int
    patience: int
    atol: float = 0.0
    rtol: float = 0.0

    def stop_early(self, i: int | Array, loss_history: Array):
        p = self.patience
        lower = jnp.max(jnp.array([i - p, 0]))
        recent_history = jax.lax.dynamic_slice(
            loss_history, start_indices=(lower,), slice_sizes=(p,)
        )

        best_loss_in_recent = jnp.min(recent_history)
        oldest_loss_in_recent = recent_history[0]

        diff = oldest_loss_in_recent - best_loss_in_recent
        abs_improvement_is_neglectable = diff <= self.atol

        rel_diff = diff / jnp.abs(best_loss_in_recent)
        rel_improvement_is_neglectable = rel_diff <= self.rtol

        current_i_is_after_patience = i > p
        """
        Stopping happens only if we actually went through a full patience period.
        """

        stop = abs_improvement_is_neglectable | rel_improvement_is_neglectable
        return stop & current_i_is_after_patience

    def stop_now(self, i: int | Array, loss_history: Array):
        """Whether optimization should stop now."""
        stop_early = self.stop_early(i=i, loss_history=loss_history)
        stop_epochs = i >= self.epochs

        return stop_early | stop_epochs

    def continue_(self, i: int | Array, loss_history: Array):
        """Whether optimization should continue (inverse of :meth:`.stop_now`)."""
        return ~self.stop_now(i=i, loss_history=loss_history)

    def which_best_in_recent_history(self, i: int, loss_history: Array):
        """
        Identifies the index of the best observation in recent history.

        Recent history includes the last ``p`` epochs looking backwards from the
        current iteration `ì``., where ``p`` is the patience.
        """
        p = self.patience
        recent_history = jax.lax.dynamic_slice(
            loss_history, start_indices=(i - p,), slice_sizes=(p,)
        )
        imin = jnp.argmin(recent_history)
        return i - self.patience + imin
