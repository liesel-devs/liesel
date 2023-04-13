"""
Goose model interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from ..goose.gibbs import GibbsKernel
from ..goose.types import ModelState, Position
from .model import Model

Array = Any


class GooseModel:
    """
    A :class:`.ModelInterface` for a Liesel :class:`.Model`.

    Parameters
    ----------
    model
        A Liesel :class:`.Model`.
    """

    def __init__(self, model: Model):
        self._model = model._copy_computational_model()

    def extract_position(
        self, position_keys: Iterable[str], model_state: ModelState
    ) -> Position:
        """
        Extracts a position from a model state.

        Parameters
        ----------
        position_keys
            An iterable of variable or node names.
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.
        """
        position = {}

        for key in position_keys:
            try:
                position[key] = model_state[key].value
            except KeyError:
                node_key = self._model.vars[key].value_node.name
                position[key] = model_state[node_key].value

        return Position(position)

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        """
        Updates and returns a model state given a position.

        Parameters
        ----------
        position
            A dictionary of variable or node names and values.
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.

        Warnings
        --------
        The ``model_state`` must be up-to-date, i.e. it must *not* contain any outdated
        nodes. Updates can only be triggered through new variable or node values in the
        ``position``. If you supply a ``model_state`` with outdated nodes, these nodes
        and their outputs will not be updated.
        """

        # sets all outdated flags in the model state to false
        # this is required to make the function jittable

        self._model.state = model_state

        for node in self._model.nodes.values():
            node._outdated = False

        for key, value in position.items():
            try:
                self._model.nodes[key].value = value  # type: ignore  # data node
            except KeyError:
                self._model.vars[key].value = value

        self._model.update()
        return self._model.state

    def log_prob(self, model_state: ModelState) -> float:
        """
        Returns the log-probability from a model state.

        Parameters
        ----------
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.
        """
        return model_state["_model_log_prob"].value


def create_categorical_gibbs_kernel(
    name: str, outcomes: Sequence[Array], model: Model
) -> GibbsKernel:
    """
    Creates a categorical Gibbs kernel.

    The prior distribution of the variable to sample must be a categorical distribution,
    usually implemented via :class:`tfd.FiniteDiscrete`.

    This kernel evaluates the full conditional log probability of the model for each
    possible value of the variable to sample. It then draws a new value for the variable
    from the categorical distribution defined by the full conditional log probabilities.

    Usually, you can define more efficient specialized kernels if you know the actual
    full conditional distribution of the variable to sample.

    Parameters
    ----------
    name
        The name of the variable to sample.
    outcomes
        List of possible outcomes.
    model
        The model to sample from.

    Examples
    --------

    In the following example, we create a categorical Gibbs kernel for a variable
    with three possible values. The prior distribution of the variable is a categorical
    distribution with probabilities ``[0.1, 0.2, 0.7]``.

    You can then use the kernel to sample from the model.

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    >>> values = [0, 1, 2]
    >>> prior_probs = [0.1, 0.2, 0.7]
    >>> value_grid = lsl.Var(values, name="value_grid")

    >>> prior = lsl.Dist(tfd.FiniteDiscrete, outcomes=value_grid, probs=prior_probs)
    >>> categorical_var = lsl.Var(
    ...     value=values[0],
    ...     distribution=prior,
    ...     name="categorical_var",
    ... )

    >>> model = lsl.GraphBuilder().add(categorical_var).build_model()
    >>> kernel = create_categorical_gibbs_kernel("categorical_var", values, model)
    >>> type(kernel)
    <class 'liesel.goose.gibbs.GibbsKernel'>

    """

    model = model._copy_computational_model()

    def transition_fn(prng_key, model_state):

        model.state = model_state
        for node in model.nodes.values():
            node._outdated = False

        def conditional_log_prob_fn(value: Array):
            """
            Evaluates the full conditional log probability of the model
            given the input value.
            """
            model.vars[name].value = value
            return model.log_prob

        conditional_log_probs = jax.tree_map(conditional_log_prob_fn, outcomes)

        draw_index = jax.random.categorical(
            prng_key, logits=jnp.stack(conditional_log_probs)
        )
        draw = jnp.array(outcomes)[draw_index]

        return {name: draw}

    return GibbsKernel([name], transition_fn)
