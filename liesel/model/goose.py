"""
Goose model interface.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..goose.gibbs import GibbsKernel
from ..goose.types import ModelState, Position
from .model import Model


class GooseModel:
    """
    A :class:`.ModelInterface` for a Liesel :class:`.Model`.

    This is an alias for :class:`~.goose.LieselInterface` provided for backwards
    compatibility.

    .. deprecated:: v0.2.6
        Use :class:`~.goose.LieselInterface` instead. This alias will be removed in
        v0.4.0.

    Parameters
    ----------
    model
        A Liesel :class:`.Model`.
    """

    def __init__(self, model: Model):
        self._model = model._copy_computational_model()
        warnings.warn(
            "lsl.GooseModel is deprecated. Use gs.LieselInterface instead."
            "This alias will be removed in v0.4.0.",
            FutureWarning,
        )

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


def finite_discrete_gibbs_kernel(
    name: str, model: Model, outcomes: Sequence | None = None
) -> GibbsKernel:
    """
    Creates a Gibbs kernel for a parameter with a finite discrete (categorical) prior.

    The prior distribution of the variable to sample must be a categorical distribution,
    usually implemented via :class:`tfd.FiniteDiscrete`.

    This kernel evaluates the full conditional log probability of the model for each
    possible value of the variable to sample. It then draws a new value for the variable
    from the categorical distribution defined by the full conditional log probabilities.

    Parameters
    ----------
    name
        The name of the variable to sample.
    model
        The model to sample from.
    outcomes
        The possible outcomes of the variable to sample. If ``outcomes=None``, the \
        possible outcomes are extracted from the prior distribution of the variable \
        to sample. Note however, that this only works for some prior distributions. \
        If the possible outcomes cannot be extracted from the prior distribution, \
        you must specify them manually via this argument.

    Examples
    --------
    In the following example, we create a categorical Gibbs kernel for a variable with
    three possible values. The prior distribution of the variable is a finite discrete
    (categorical) distribution with the probabilities ``[0.1, 0.2, 0.7]``.

    You can then use the kernel to sample from the model:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    >>> values = [0.0, 1.0, 2.0]
    >>> prior_probs = [0.1, 0.2, 0.7]
    >>> value_grid = lsl.Var(values, name="value_grid")

    >>> prior = lsl.Dist(tfd.FiniteDiscrete, outcomes=value_grid, probs=prior_probs)
    >>> categorical_var = lsl.Var(
    ...     value=values[0],
    ...     distribution=prior,
    ...     name="categorical_var",
    ... )

    >>> model = lsl.GraphBuilder().add(categorical_var).build_model()
    >>> kernel = finite_discrete_gibbs_kernel("categorical_var", model)
    >>> type(kernel)
    <class 'liesel.goose.gibbs.GibbsKernel'>

    Example for a variable with a Bernoulli prior distribution:

    >>> prior = lsl.Dist(tfd.Bernoulli, probs=lsl.Value(0.7))
    >>> dummy_var = lsl.Var(
    ...     value=1,
    ...     distribution=prior,
    ...     name="dummy_var",
    ... )

    >>> model = lsl.GraphBuilder().add(dummy_var).build_model()
    >>> kernel = finite_discrete_gibbs_kernel("dummy_var", model, outcomes=[0, 1])
    >>> type(kernel)
    <class 'liesel.goose.gibbs.GibbsKernel'>

    """
    if outcomes is not None:
        outcomes = jnp.asarray(outcomes)
    else:
        dist = model.vars[name].dist_node.init_dist()  # type: ignore
        assert dist.batch_shape == ()

        match dist:
            case tfd.Bernoulli():
                outcomes = jnp.array([0, 1], dtype=dist.dtype)
            case tfd.FiniteDiscrete():
                outcomes = dist.outcomes
            case _:
                raise ValueError(
                    "Cannot extract outcomes from the distribution of variable "
                    f"'{name}'. Please provide the argument 'outcomes'."
                )

    model = model._copy_computational_model()
    model.auto_update = False

    def transition_fn(prng_key, model_state):
        model.state = model_state

        for node in model.nodes.values():
            node._outdated = False

        def conditional_log_prob_fn(value):
            """
            Evaluates the full conditional log probability of the model
            given the input value.
            """
            model.vars[name].value = value
            model.update("_model_log_prob")
            return model.log_prob

        conditional_log_probs = jax.vmap(conditional_log_prob_fn)(outcomes)
        draw_index = jax.random.categorical(prng_key, logits=conditional_log_probs)
        draw = outcomes[draw_index]

        return {name: draw}

    return GibbsKernel([name], transition_fn)
