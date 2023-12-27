from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

import liesel.goose as gs
import liesel.model as lsl

from .types import Array, ModelState, Position


@dataclass
class OptimResult:
    """Holds the results of model optimization with :func:`.optim`."""

    position: Position
    """Position dictionary of optimized parameters with their final values."""
    model_state: ModelState
    """Final model state after optimization."""
    iteration: int
    """Iteration counter of the last iteration."""
    abs_change: float | Array
    """Absolute change in negative log probability in the last iteration."""
    rel_change: float | Array
    """Relative change in negative log probability in the last iteration."""
    neg_log_prob: float
    """Negative log probability after the last iteration."""


def _find_observed(graph: lsl.Model) -> dict[str, lsl.Var | lsl.Node]:
    obs = {
        var_.name: jnp.array(var_.value)
        for var_ in graph.vars.values()
        if var_.observed
    }
    for node in graph.nodes.values():
        try:
            if node.observed:  # type: ignore
                obs[node.name] = jnp.array(node.value)
        except AttributeError:
            pass
    return obs


def batched_nodes(nodes: dict[str, Array], batch_indices: Array) -> dict[str, Array]:
    """Returns a subset of the graph state using the given batch indices."""
    return jax.tree_util.tree_map(lambda x: x[batch_indices, ...], nodes)


def _find_sample_size(graph: lsl.Model) -> int:
    obs = {var_.name: var_ for var_ in graph.vars.values() if var_.observed}
    n_set = {int(np.array(var_.value.shape)[0, ...]) for var_ in obs.values()}
    if len(n_set) > 1:
        raise ValueError(
            "The observed variables must have the same number of observations."
        )
    return n_set.pop()


def optim(
    model: lsl.Model,
    params: Sequence[str],
    optimizer: optax.GradientTransformation | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-6,
    maxiter: int = 1_000,
    batch_size: int | None = None,
) -> OptimResult:
    """
    Optimize the parameters of a  Liesel :class:`.Model`.

    Approximates maximum a posteriori (MAP) parameter estimates by minimizing the
    negative log probability of the model.

    Parameters
    ----------
    model
        The Liesel model to optimize.
    params
        The parameters to optimize. All other parameters of the model are held fixed.
    optimizer
        An optimizer from the ``optax`` library. If ``None`` , \
        ``optax.adam(learning_rate=1e-1)`` is used.
    atol
        The absolute tolerance for early stopping. If the change in the negative log \
        probability is smaller than this value, optimization stops early.
    rtol
        The relative tolerance for early stopping. If the relative change in the \
        negative log probability is smaller than this value, optimization stops early.
    maxiter
        The maximum number of optimization steps.
    batch_size
        The batch size for stochastic gradient descent. If ``None``, the whole dataset \
        is used for each optimization step.

    Returns
    -------
    A dataclass of type :class:`.OptimResult`, giving access to the results.

    Examples
    --------

    We show a minimal example. First, import ``tfd``.

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    Next, generate some data.

    >>> key = jax.random.PRNGKey(42)
    >>> key, subkey = jax.random.split(key)
    >>> x = jax.random.normal(key, (100,))
    >>> y = 0.5 + 1.2 * x + jax.random.normal(subkey, (100,))

    Next, set up a linear model. For simplicity, we assume the scale to be fixed to the
    true value of 1.

    >>> coef = lsl.param(jnp.zeros(2), name="coef")
    >>> xvar = lsl.obs(jnp.c_[jnp.ones_like(x), x], name="x")
    >>> mu = lsl.Var(lsl.Calc(jnp.dot, xvar, coef), name="mu")
    >>> ydist = lsl.Dist(tfd.Normal, loc=mu, scale=1.0)
    >>> yvar = lsl.obs(y, ydist, name="y")
    >>> model = lsl.GraphBuilder().add(yvar).build_model()

    Now, we are ready to run the optimization.

    >>> result = gs.optim(model, params=["coef"])
    >>> result.position
    {'coef': Array([0.5227438, 1.2980561], dtype=float32)}

    We can now, for example, use ``result.model_state`` in
    :meth:`.EngineBuilder.set_initial_values` to implement a "warm start" of MCMC
    sampling.
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate=1e-1)

    n = _find_sample_size(model)
    observed = _find_observed(model)
    interface = gs.LieselInterface(model)
    position = interface.extract_position(params, model.state)
    interface._model.auto_update = False

    def _batched_neg_log_prob(position: Position, batch_indices: Array | None = None):
        if batch_indices is not None:
            batched_observed = batched_nodes(observed, batch_indices)
            position = position | batched_observed  # type: ignore

        updated_state = interface.update_state(position, model.state)
        return -updated_state["_model_log_prob"].value

    neg_log_prob_grad = jax.grad(_batched_neg_log_prob, argnums=0)

    opt_state = optimizer.init(position)

    nlp_start = _batched_neg_log_prob(position, batch_indices=None)
    abs_change: Array | float = 100.0
    rel_change: Array | float = 100.0
    i = 0

    if batch_size is None:
        batch_size = n

    n_batches_inside = n // batch_size
    batch_indices = jnp.arange(0, batch_size)
    last_batch_size = n % batch_size
    last_batch_start = n_batches_inside * batch_size
    last_batch_end = last_batch_start + last_batch_size
    last_batch_indices = jnp.arange(last_batch_start, last_batch_end)

    def _batched_update(i, val):
        pos, opt_state = val
        batch_indices_here = batch_indices + i * batch_size

        grad = neg_log_prob_grad(pos, batch_indices=batch_indices_here)
        updates, opt_state = optimizer.update(grad, opt_state)
        pos = optax.apply_updates(pos, updates)
        return pos, opt_state

    def _last_batched_update(i, val):
        pos, opt_state = val
        grad = neg_log_prob_grad(pos, batch_indices=last_batch_indices)
        updates, opt_state = optimizer.update(grad, opt_state)
        pos = optax.apply_updates(pos, updates)
        return pos, opt_state

    def _update_cond(val):
        i, _, _, nlp_old, nlp_new = val

        abs_change = jnp.abs(nlp_old - nlp_new)
        rel_change = jnp.abs(abs_change / nlp_old)

        return (abs_change > atol) & (rel_change > rtol) & (i < maxiter)

    def _update_body(val):
        i, position, opt_state, _, nlp_old = val

        init = (position, opt_state)
        position, opt_state = jax.lax.fori_loop(
            0, n_batches_inside, _batched_update, init
        )

        init = (position, opt_state)
        position, opt_state = jax.lax.fori_loop(
            n_batches_inside, n_batches_inside + 1, _last_batched_update, init
        )

        nlp_new = _batched_neg_log_prob(position, batch_indices=None)

        i += 1
        return (i, position, opt_state, nlp_old, nlp_new)

    i, position, _, nlp_old, nlp_new = jax.lax.while_loop(
        _update_cond, _update_body, (i, position, opt_state, 0.0, nlp_start)
    )

    abs_change = jnp.abs(nlp_old - nlp_new)
    rel_change = jnp.abs(abs_change / nlp_old)

    final_state = interface.update_state(position, model.state)
    return OptimResult(position, final_state, i, abs_change, rel_change, nlp_new)
