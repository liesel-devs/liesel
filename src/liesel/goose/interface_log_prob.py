from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.flatten_util

from .types import ModelInterface, ModelState, Position

Array = Any


class InterfaceLogProb:
    """
    Interface for evaluating the unnormalized log probability represented by a model
    interface.

    Also provides access to the first and second derivatives.

    Parameters
    ----------
    model
        A model interface.
    model_state
        A model state.
    diff_mode
        Which auto-diff mode to use for the Hessian.

    See Also
    --------
    .FlatInterfaceLogProb : A similar class that returns gradients and hessians as
        arrays. liesel.model.LogProb : Similar class, specialized for liesel models.
    liesel.model.FlatLogProb : Similar class, specialized for liesel models.
    """

    def __init__(
        self,
        model: ModelInterface,
        model_state: ModelState,
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model
        self.model_state = model_state
        self._grad_fn = jax.grad(self.log_prob)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argument value {diff_mode=}")
        self.diff_mode = diff_mode

    def __call__(self, position: Position) -> Array:
        return self.log_prob(position=position)

    def log_prob(self, position: Position) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
        updated_state = self.model.update_state(position, self.model_state)
        return self.model.log_prob(updated_state)

    def grad(self, position: Position) -> dict[str, Array]:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(position)

    def hessian(self, position: Position) -> dict[str, Array]:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(position)


class FlatInterfaceLogProb:
    """
    Interface for evaluating the unnormalized log probability represented by a model
    interface.

    Also provides access to the first and second derivatives. The methods
    :meth:`.FlatLogProb.grad` and :meth:`.FlatLogProb.hessian` are flattened, which
    means they expect arrays as inputs and return arrays.

    Parameters
    ----------
    model
        A model interface.
    model_state
        A model state
    position_keys
        Names of the variables at which to evaluate the log probability. Other \
        variables will be kept fixed at their current values in the model state.
    diff_mode
        Which auto-diff mode to use for the Hessian.

    See Also
    --------
    .InterfaceLogProb : A similar class that returns gradients and hessians as
        dictionaries. liesel.model.LogProb : Similar class, specialized for liesel
        models.
    liesel.model.FlatLogProb : Similar class, specialized for liesel models.

    """

    def __init__(
        self,
        model: ModelInterface,
        model_state: ModelState,
        position_keys: Sequence[str],
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model
        self.model_state = model_state

        position = self.model.extract_position(position_keys, model_state)
        _, unravel_fn = jax.flatten_util.ravel_pytree(position)
        self.unravel_fn = unravel_fn

        self._grad_fn = jax.grad(self)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argument value {diff_mode=}")
        self.diff_mode = diff_mode

    def __call__(self, flat_position: Array) -> Array:
        return self.log_prob(flat_position=flat_position)

    def log_prob(self, flat_position: Array) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
        position = self.unravel_fn(flat_position)
        updated_state = self.model.update_state(position, self.model_state)
        return self.model.log_prob(updated_state)

    def grad(self, flat_position: Array) -> Array:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(flat_position)

    def hessian(self, flat_position: Array) -> Array:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(flat_position)
