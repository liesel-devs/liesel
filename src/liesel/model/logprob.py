from collections.abc import Sequence
from typing import Literal

import jax
import jax.flatten_util

from ..goose.types import Array
from .model import Model


class LieselLogProb:
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

    See Also
    ---------
    .FlatLieselLogProb: A similar class that returns gradients and hessians as arrays.

    Examples
    --------
    We initialize a very basic Liesel model:

    >>> x = lsl.Var.new_param(
    ...     jnp.zeros(2),
    ...     distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    ...     name="x",
    ... )
    >>> model = lsl.Model([x])

    Now we initialize the log prob object:

    >>> lp = lsl.LieselLogProb(model)

    And evaluate the log prob (the unnormalized log posterior) at a new position:

    >>> lp({"x": jnp.array([1.0, 2.0])})
    Array(-4.3378773, dtype=float32)

    Now we evaluate the gradient of the unnormalized log posterior at the new position:

    >>> lp.grad({"x": jnp.array([1.0, 2.0])})
    {'x': Array([-1., -2.], dtype=float32)}

    And, finally, the hessian:

    >>> lp.hessian({"x": jnp.array([1.0, 2.0])})
    {'x': {'x': Array([[-1., -0.],
           [-0., -1.]], dtype=float32)}}

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
            raise ValueError(f"Unrecognized argument value {diff_mode=}")
        self.component = component
        self.diff_mode = diff_mode

    def __call__(self, position: dict[str, Array | float]) -> Array:
        """
        Log probability function evaluated at provided ``position``.
        """
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


class FlatLieselLogProb:
    """
    Interface for evaluating the unnormalized log probability of a Liesel model.

    Also provides access to the first and second derivatives.
    The methods :meth:`.FlatLieselLogProb.grad` and
    :meth:`.FlatLieselLogProb.hessian` are
    flattened, which means the expect arrays as inputs and return arrays.

    Parameters
    ----------
    model
        A Liesel model instance.
    position_keys
        Names of the variables at which to evaluate the log probability. Other \
        variables will be kept fixed at their current values in the model state.
    component
        Which component of the model's log probability to evaluate.
    diff_mode
        Which auto-diff mode to use for the Hessian.

    See Also
    --------
    .LieselLogProb: A similar class that returns gradients and hessians as dictionaries.


    Examples
    --------
    We initialize a very basic Liesel model:

    >>> x = lsl.Var.new_param(
    ...     jnp.zeros(2),
    ...     distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    ...     name="x",
    ... )
    >>> model = lsl.Model([x])

    Now we initialize the log prob object:

    >>> lp = lsl.FlatLieselLogProb(model, ["x"])

    And an array of new values to evaluate the log probability at:

    >>> xnew = jnp.array([1.0, 2.0])

    >>> lp(xnew)
    Array(-4.3378773, dtype=float32)

    >>> lp.grad(xnew)
    Array([-1., -2.], dtype=float32)

    >>> lp.hessian(xnew)
    Array([[-1., -0.],
           [-0., -1.]], dtype=float32)
    """

    def __init__(
        self,
        model: Model,
        position_keys: Sequence[str],
        component: Literal["log_prob", "log_lik", "log_prior"] = "log_prob",
        diff_mode: Literal["forward", "reverse"] = "forward",
    ):
        self.model = model

        if position_keys is None:
            position_keys = [
                var.name for var in self.model.vars.values() if var.parameter
            ]

        position = self.model.extract_position(position_keys, self.model.state)
        _, unravel_fn = jax.flatten_util.ravel_pytree(position)
        self.unravel_fn = unravel_fn

        self._grad_fn = jax.grad(self)

        if diff_mode == "forward":
            self._hessian_fn = jax.jacfwd(self._grad_fn)
        elif diff_mode == "reverse":
            self._hessian_fn = jax.jacrev(self._grad_fn)
        else:
            raise ValueError(f"Unrecognized argument value {diff_mode=}")
        self.component = component
        self.diff_mode = diff_mode

    def __call__(self, flat_position: Array) -> Array:
        """
        Log probability function evaluated at provided ``flat_position``.
        """
        return self.log_prob(flat_position=flat_position)

    def log_prob(self, flat_position: Array) -> Array:
        """
        Log probability function evaluated at provided ``flat_position``.
        """
        position = self.unravel_fn(flat_position)
        updated_state = self.model.update_state(position, self.model.state)
        return updated_state[f"_model_{self.component}"].value

    def grad(self, flat_position: Array) -> Array:
        """
        Gradient of the log probability function with respect to the ``flat_position``.
        """
        return self._grad_fn(flat_position)

    def hessian(self, flat_position: Array) -> Array:
        """
        Hessian of the log probability function with respect to the ``flat_position``.
        """
        return self._hessian_fn(flat_position)
