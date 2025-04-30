"""
Goose model interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..goose.gibbs import GibbsKernel
from .model import Model


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

    >>> model = lsl.Model([categorical_var])
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

    >>> model = lsl.Model([dummy_var])
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
