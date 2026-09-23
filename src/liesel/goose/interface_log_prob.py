from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.flatten_util
import jax.numpy as jnp
from jax.typing import ArrayLike

from .types import ModelInterface, ModelState, PositionInput, PyTree, Scalar

Array = Any
"""Deprecated compatibility alias for Any; use PyTree or an array-specific type."""


class InterfaceLogProb:
    """
    Interface for evaluating the unnormalized log probability represented by a model
    interface.

    Also provides access to the first and second derivatives.

    Derivative methods convert position mappings to dictionaries before differentiation.
    When applying JAX transformations externally, pass a dictionary or another
    registered pytree to the transformed function.

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

    def __call__(self, position: PositionInput) -> Scalar:
        return self.log_prob(position=position)

    def log_prob(self, position: PositionInput) -> Scalar:
        """
        Log probability function evaluated at provided ``position``.
        """
        updated_state = self.model.update_state(position, self.model_state)
        return self.model.log_prob(updated_state)

    def grad(self, position: PositionInput) -> dict[str, PyTree]:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(dict(position))

    def hessian(self, position: PositionInput) -> dict[str, PyTree]:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(dict(position))


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

    def __call__(self, flat_position: ArrayLike) -> Scalar:
        return self.log_prob(flat_position=flat_position)

    def log_prob(self, flat_position: ArrayLike) -> Scalar:
        """
        Log probability function evaluated at provided ``position``.
        """
        position = self.unravel_fn(jnp.asarray(flat_position))
        updated_state = self.model.update_state(position, self.model_state)
        return self.model.log_prob(updated_state)

    def grad(self, flat_position: ArrayLike) -> jax.Array:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(jnp.asarray(flat_position))

    def hessian(self, flat_position: ArrayLike) -> jax.Array:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(jnp.asarray(flat_position))
