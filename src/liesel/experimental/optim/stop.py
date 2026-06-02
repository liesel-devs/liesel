from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .types import Array


@dataclass
class Stopper:
    """
    Handles maximum-epoch and early stopping decisions.

    ``Stopper`` uses a rolling patience window over the recorded monitoring loss.
    In the experimental optimizer engine, the argument ``i`` passed to the stopper is
    the number of completed epochs, which is also the next epoch index to be written.
    Therefore, the recent patience window is ``loss_history[i - patience : i]`` once
    at least ``patience`` epochs have been completed.

    Parameters
    ----------
    epochs
        The maximum number of optimization epochs.
    patience
        Length of the rolling early-stopping window. Early stopping can only happen
        after more than ``patience`` epochs have been completed.
    atol
        Absolute tolerance for early stopping.
    rtol
        Relative tolerance for early stopping. The default of ``0.0`` disables
        additional relative tolerance beyond exact non-improvement.

    Notes
    -----
    Early stopping happens when the oldest loss value within the patience window is
    the best loss value within the patience window. With experimental optimizer
    indexing, a simplified pseudo-implementation is:

    .. code-block:: python

        def stop(patience, i, loss_history):
            recent_history = loss_history[i - patience : i]
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
            recent_history = loss_history[i - patience : i]
            oldest_within_patience = recent_history[0]
            best_within_patience = np.min(recent_history)

            diff = oldest_within_patience - best_within_patience
            rel_diff = diff / np.abs(best_within_patience)

            abs_improvement_is_neglectable = diff <= atol
            rel_improvement_is_neglectable = rel_diff <= rtol

            return abs_improvement_is_neglectable | rel_improvement_is_neglectable

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim import Stopper
    >>> stopper = Stopper(epochs=10, patience=3)
    >>> loss_history = jnp.array([3.0, 2.0, 1.0, 1.5, 1.6, 1.7])
    >>> bool(stopper.stop_early(i=5, loss_history=loss_history))
    True
    >>> int(stopper.which_best_in_recent_history(i=5, loss_history=loss_history))
    2

    ``max_iter`` is available as a read-only alias for ``epochs``:

    >>> stopper.max_iter
    10
    """

    epochs: int
    patience: int
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")
        if self.patience > self.epochs:
            raise ValueError("patience must be less than or equal to epochs.")
        if self.atol < 0:
            raise ValueError("atol must be non-negative.")
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative.")

    @property
    def max_iter(self) -> int:
        """Read-only alias for :attr:`epochs`."""
        return self.epochs

    def stop_early(self, i: int | Array, loss_history: Array):
        """
        Returns whether the recent patience window should trigger early stopping.

        Parameters
        ----------
        i
            Number of completed epochs, or equivalently the next epoch index to be
            written.
        loss_history
            Loss history whose first ``i`` entries have been written.
        """
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
        Identifies the index of the best observation in the recent loss window.

        The recent loss window contains ``loss_history[i - patience : i]``, where
        ``i`` is the number of completed epochs. This returns the best index in that
        recent window, not necessarily the global best index in the full loss
        history.
        """
        p = self.patience
        recent_history = jax.lax.dynamic_slice(
            loss_history, start_indices=(i - p,), slice_sizes=(p,)
        )
        imin = jnp.argmin(recent_history)
        return i - self.patience + imin
