from typing import Literal

import jax
import jax.flatten_util

from ..goose.types import Array
from .model import Model


class LogProb:
    """
    Interface for evaluating the unnormalized log probability of a Liesel model.

    Also provides access to the first and second derivatives.

    Parameters
    ----------
    model
        A Liesel model instance.
    component
        Which component of the model's log probability to evaluate.
    diff_mode
        Which auto-diff mode to use for the Hessian.
    """

    def __init__(
        self,
        model: Model,
        component: Literal["log_prob", "log_lik", "log_prior"] = "log_prob",
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model
        self._grad_fn = jax.grad(self.log_prob)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argumetn value {diff_mode=}")

        self.component = component
        self.diff_mode = diff_mode

    def __call__(self, position: dict[str, Array | float]) -> Array:
        return self.log_prob(position=position)

    def log_prob(self, position: dict[str, Array | float]) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
        updated_state = self.model.update_state(position, self.model.state)
        return updated_state[f"_model_{self.component}"].value

    def grad(self, position: dict[str, Array | float]) -> dict[str, Array]:
        """
        Gradient of the log probability function with respect to the ``position``.
        """
        return self._grad_fn(position)

    def hessian(
        self,
        position: dict[str, Array | float],
    ) -> dict[str, Array]:
        """
        Hessian of the log probability function with respect to the ``position``.
        """
        return self._hessian_fn(position)


class FlatLogProb:
    """
    Interface for evaluating the unnormalized log probability of a Liesel model.

    Also provides access to the first and second derivatives.
    The methods :meth:`.FlatLogProb.grad` and :meth:`.FlatLogProb.hessian` are
    flattened, which means the expect arrays as inputs and return arrays.

    Parameters
    ----------
    *names
        Names of the variables at which to evaluate the log probability. Other \
        variables will be kept fixed at their current values in the model state.
    model
        A Liesel model instance.
    component
        Which component of the model's log probability to evaluate.
    diff_mode
        Which auto-diff mode to use for the Hessian.
    """

    def __init__(
        self,
        *names: str,
        model: Model,
        component: Literal["log_prob", "log_lik", "log_prior"] = "log_prob",
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model

        position = self.model.extract_position(names, self.model.state)
        _, unravel_fn = jax.flatten_util.ravel_pytree(position)
        self.unravel_fn = unravel_fn

        self._grad_fn = jax.grad(self)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argumetn value {diff_mode=}")

        self.component = component
        self.diff_mode = diff_mode

    def __call__(self, flat_position: Array) -> Array:
        return self.log_prob(flat_position=flat_position)

    def log_prob(self, flat_position: Array) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
        position = self.unravel_fn(flat_position)
        updated_state = self.model.update_state(position, self.model.state)
        return updated_state[f"_model_{self.component}"].value

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
