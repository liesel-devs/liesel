from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from tqdm import tqdm

from ..model import Model, Node, Var
from .interface import LieselInterface
from .types import Array, KeyArray, ModelState, Position


def array_to_dict(
    x: Array, names_prefix: str = "x", prefix_1d: bool = False
) -> dict[str, Array]:
    """Turns a 2d-array into a dict."""

    if isinstance(x, float) or x.ndim == 1:
        if prefix_1d:
            return {f"{names_prefix}0": x}
        else:
            return {names_prefix: x}
    elif x.ndim == 2:
        return {f"{names_prefix}{i}": x[:, i] for i in range(x.shape[-1])}
    else:
        raise ValueError(f"x should have ndim <= 2, but it has x.ndim={x.ndim}")


@dataclass
class OptimResult:
    """Holds the results of model optimization with :func:`.optim_flat`."""

    model_state: ModelState
    """Final model state after optimization."""
    position: Position
    """Position dictionary of optimized parameters with their final values."""
    iteration: int
    """Iteration counter of the last iteration."""
    iteration_best: int
    """Iteration counter of the iteration with lowest loss."""
    history: dict[str, dict[str, Array] | Array]
    """History of loss evaluations and, if applicable, intermediate position values."""
    max_iter: int
    """Maximum number of iterations."""
    n_train: int
    """Number of training observations."""
    n_validation: int
    """Number of test observations."""


def _find_observed(model: Model) -> dict[str, Var | Node]:
    obs = {
        var_.name: jnp.array(var_.value)
        for var_ in model.vars.values()
        if var_.observed
    }
    return obs


def batched_nodes(nodes: dict[str, Array], batch_indices: Array) -> dict[str, Array]:
    """Returns a subset of the model state using the given batch indices."""
    return jax.tree_util.tree_map(lambda x: x[batch_indices, ...], nodes)


def _generate_batch_indices(key: KeyArray, n: int, batch_size: int) -> Array:
    n_full_batches = n // batch_size
    shuffled_indices = jax.random.permutation(key, n)
    shuffled_indices_subset = shuffled_indices[0 : n_full_batches * batch_size]
    list_of_batch_indices = jnp.array_split(shuffled_indices_subset, n_full_batches)
    return jnp.asarray(list_of_batch_indices)


def _find_sample_size(model: Model) -> int:
    obs = {var_.name: var_ for var_ in model.vars.values() if var_.observed}
    n_set = {int(np.array(var_.value.shape)[0, ...]) for var_ in obs.values()}
    if len(n_set) > 1:
        raise ValueError(
            "The observed variables must have the same number of observations."
        )
    return n_set.pop()


@dataclass
class Stopper:
    """
    Handles (early) stopping for :func:`.optim_flat`.

    Parameters
    ----------
    max_iter
        The maximum number of optimization steps.
    patience
        Early stopping happens only, if there was no improvement for the number of\
        patience iterations, and there were at least as many iterations as the length\
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
    The relative magnitude of the difference is calculaterd with respect to the best
    lost within patience. A simplified pseudo-implementation is:

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

    max_iter: int
    patience: int
    atol: float = 1e-3
    rtol: float = 0.0

    def stop_early(self, i: int | Array, loss_history: Array):
        p = self.patience
        lower = jnp.max(jnp.array([i - p + 1, 0]))
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
        stop_max_iter = i >= (self.max_iter - 1)

        return stop_early | stop_max_iter

    def continue_(self, i: int | Array, loss_history: Array):
        """Whether optimization should continue (inverse of :meth:`.stop_now`)."""
        return ~self.stop_now(i=i, loss_history=loss_history)

    def which_best_in_recent_history(self, i: int, loss_history: Array):
        """
        Identifies the index of the best observation in recent history.

        Recent history includes the last ``p`` iterations looking backwards from the
        current iteration `ì``., where ``p`` is the patience.
        """
        p = self.patience
        recent_history = jax.lax.dynamic_slice(
            loss_history, start_indices=(i - p + 1,), slice_sizes=(p,)
        )
        imin = jnp.argmin(recent_history)
        return i - self.patience + imin + 1


def _validate_log_prob_decomposition(
    interface: LieselInterface, position: Position, state: ModelState
) -> bool:
    updated_state = interface.update_state(position, state)
    log_prob = updated_state["_model_log_prob"].value

    log_lik = updated_state["_model_log_lik"].value
    log_prior = updated_state["_model_log_prior"].value

    if not jnp.allclose(log_prob, log_lik + log_prior):
        raise ValueError(
            f"You model's {log_prob=} cannot correctly be decomposed into the"
            f" {log_prior=} and the {log_lik=}. Check whether your observed variables"
            " have the attribute observed=True and whether your parameter variables"
            " have the attribute parameter=True."
        )

    return True


def optim_flat(
    model_train: Model,
    params: Sequence[str],
    optimizer: optax.GradientTransformation | None = None,
    stopper: Stopper | None = None,
    batch_size: int | None = None,
    batch_seed: int | None = None,
    save_position_history: bool = True,
    model_validation: Model | None = None,
    restore_best_position: bool = True,
    prune_history: bool = True,
    progress_bar: bool = True,
) -> OptimResult:
    """
    Optimize the parameters of a  Liesel :class:`.Model`.

    Approximates maximum a posteriori (MAP) parameter estimates by minimizing the
    negative log posterior probability of the model. If you use batching, be aware that
    the batching functionality implemented here assumes a "flat" model structure.
    See below for details.

    .. warning::
        This function is experimental.
        The API may change more quickly than in other parts of the library.
        Check your results carefully. If you encounter puzzling results, try to disable
        batching.

    Params
    ------
    model_train
        The Liesel model to optimize.
    params
        List of parameter names to optimize. All other parameters of the model are held\
        fixed.
    optimizer
        An optimizer from the ``optax`` library. If ``None`` , \
        ``optax.adam(learning_rate=1e-2)`` is used.
    stopper
        A :class:`.Stopper` that carries information about the maximum number of\
        iterations and early stopping.
    batch_size
        The batch size. If ``None``, the whole dataset\
        is used for each optimization step.
    batch-seed
        Batches are assembled randomly in each iteration. This is the seed used for \
        shuffling in this step.
    save_position_history
        If ``True``, the position history is saved to the results object.
    model_validation
        If supplied, this model serves as a validation model, which means that early\
        stopping is based on the negative log likelihood evaluated using the observed\
        data in this model. If ``None``, no early stopping is conducted.
    restore_best_position
        If ``True``, the position with the lowest loss within the patience defined\
        by the supplied :class:`.Stopper` is restored as the final postion. If \
        ``False``, the last iteration's position is used.
    prune_history
        If ``True``, the history is pruned to the length of the final iteration. This\
        means, the history can be shorter than the maximum number of iterations defined\
        by the supplied :class:`.Stopper`. If ``False``, unused history entries are set\
        to ``jax.numpy.nan`` if optimization stops early.
    progress_bar
        Whether to use a progress bar.

    Returns
    -------
    A dataclass of type :class:`.OptimResult`, giving access to the results.

    See Also
    --------
    .history_to_df : A helper function to turn the :attr:`.OptimResult.history` into
        a ``pandas.DataFrame`` - nice for quickly plotting results.

    Notes
    -----

    If you use batching, be aware that the
    batching functionality implemented here assumes a "flat" model structure. This means
    that this function assumes that, for all :class:`.Var` objects in your model, it
    is valid to index their values like this::

        var_object.value[batch_indices, ...]

    The batching functionality also assumes that all objects that should be batched
    are included as :class:`.Var` objects with ``Var.observed`` set to ``True``.

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

    >>> stopper = gs.Stopper(max_iter=1000, patience=10, atol=0.01)
    >>> result = gs.optim_flat(model, params=["coef"], stopper=stopper)
    >>> {name: jnp.round(value, 2) for name, value in result.position.items()}
    {'coef': Array([0.52, 1.29], dtype=float32)}

    We can now, for example, use ``result.model_state`` in
    :meth:`.EngineBuilder.set_initial_values` to implement a "warm start" of MCMC
    sampling.

    """
    # ---------------------------------------------------------------------------------
    # Validation input
    if restore_best_position:
        assert (
            save_position_history
        ), "Cannot restore best position if history is not saved."

    # ---------------------------------------------------------------------------------
    # Pre-process inputs

    batch_seed = (
        batch_seed if batch_seed is not None else np.random.randint(low=1, high=1000)
    )

    if stopper is None:
        stopper = Stopper(max_iter=10_000, patience=10)

    user_patience = stopper.patience
    if model_validation is None:
        model_validation = model_train
        stopper.patience = stopper.max_iter

    if optimizer is None:
        optimizer = optax.adam(learning_rate=1e-2)

    n_train = _find_sample_size(model_train)
    n_validation = _find_sample_size(model_validation)
    observed = _find_observed(model_train)

    batch_size = batch_size if batch_size is not None else n_train

    interface_train = LieselInterface(model_train)
    position = interface_train.extract_position(params, model_train.state)
    interface_train._model.auto_update = False

    interface_validation = LieselInterface(model_validation)
    interface_validation._model.auto_update = False

    # ---------------------------------------------------------------------------------
    # Validate model log prob decomposition
    _validate_log_prob_decomposition(
        interface_train, position=position, state=model_train.state
    )
    _validate_log_prob_decomposition(
        interface_validation, position=position, state=model_validation.state
    )

    # ---------------------------------------------------------------------------------
    # Define loss function(s)

    likelihood_scalar = n_train / batch_size
    likelihood_scalar_validation = n_train / n_validation

    def _batched_neg_log_prob(
        position: Position, model_state: ModelState, batch_indices: Array | None = None
    ):
        if batch_indices is not None:
            batched_observed = batched_nodes(observed, batch_indices)
            position = position | batched_observed  # type: ignore

        updated_state = interface_train.update_state(position, model_state)
        log_lik = likelihood_scalar * updated_state["_model_log_lik"].value
        log_prior = updated_state["_model_log_prior"].value
        log_prob = log_lik + log_prior
        return -log_prob

    def _neg_log_prob_train(position: Position, model_state: ModelState):
        updated_state = interface_train.update_state(position, model_state)
        return -updated_state["_model_log_prob"].value

    def _neg_log_prob_validation(position: Position, model_state: ModelState):
        updated_state = interface_validation.update_state(position, model_state)
        log_lik = likelihood_scalar_validation * updated_state["_model_log_lik"].value
        log_prior = updated_state["_model_log_prior"].value
        log_prob = log_lik + log_prior
        return -log_prob

    neg_log_prob_grad = jax.grad(_batched_neg_log_prob, argnums=0)

    # ---------------------------------------------------------------------------------
    # Initialize history

    history: dict[str, Any] = dict()
    history["loss_train"] = jnp.zeros(shape=stopper.max_iter)
    history["loss_validation"] = jnp.zeros(shape=stopper.max_iter)

    if save_position_history:
        history["position"] = {
            name: jnp.zeros((stopper.max_iter,) + jnp.shape(value))
            for name, value in position.items()
        }
        history["position"] = jax.tree.map(
            lambda d, pos: d.at[0].set(pos), history["position"], position
        )
    else:
        history["position"] = None

    loss_train_start = _neg_log_prob_train(
        position=position, model_state=model_train.state
    )
    loss_validation_start = _neg_log_prob_validation(
        position=position, model_state=model_validation.state
    )
    history["loss_train"] = history["loss_train"].at[0].set(loss_train_start)
    history["loss_validation"] = (
        history["loss_validation"].at[0].set(loss_validation_start)
    )

    # ---------------------------------------------------------------------------------
    # Initialize while loop carry dictionary

    init_val: dict[str, Any] = dict()
    init_val["while_i"] = 0
    init_val["history"] = history
    init_val["position"] = position
    init_val["opt_state"] = optimizer.init(position)
    init_val["key"] = jax.random.PRNGKey(batch_seed)
    init_val["model_state_train"] = model_train.state
    init_val["model_state_validation"] = model_validation.state

    # ---------------------------------------------------------------------------------
    # Initialize while loop carry dictionary

    progress_bar = tqdm(
        total=stopper.max_iter - 1,
        desc=(
            f"Training loss: {loss_train_start:.3f}, Validation loss:"
            f" {loss_validation_start:.3f}"
        ),
        position=0,
        leave=True,
    )

    def tqdm_callback(val):
        i = val["while_i"]
        loss_train = val["history"]["loss_train"][i]
        loss_validation = val["history"]["loss_validation"][i]
        desc = (
            f"Training loss: {loss_train:.3f}, Validation loss: {loss_validation:.3f}"
        )
        progress_bar.update(1)
        progress_bar.set_description(desc)

    # ---------------------------------------------------------------------------------
    # Define while loop body

    def body_fun(val: dict):
        _, subkey = jax.random.split(val["key"])
        batches = _generate_batch_indices(key=subkey, n=n_train, batch_size=batch_size)

        # -----------------------------------------------------------------------------
        # Loop over batches

        def _fori_body(i, val):
            batch = batches[i]
            pos = val["position"]

            grad = neg_log_prob_grad(
                pos, batch_indices=batch, model_state=val["model_state_train"]
            )
            updates, opt_state = optimizer.update(grad, val["opt_state"], params=pos)

            val["position"] = optax.apply_updates(pos, updates)
            val["opt_state"] = opt_state

            return val

        val = jax.lax.fori_loop(
            body_fun=_fori_body, init_val=val, lower=0, upper=len(batches)
        )

        # -----------------------------------------------------------------------------
        # Save values and increase counter

        val["while_i"] += 1

        loss_train = _neg_log_prob_train(
            val["position"], model_state=val["model_state_train"]
        )

        val["history"]["loss_train"] = (
            val["history"]["loss_train"].at[val["while_i"]].set(loss_train)
        )

        loss_validation = _neg_log_prob_validation(
            val["position"], model_state=val["model_state_validation"]
        )

        val["history"]["loss_validation"] = (
            val["history"]["loss_validation"].at[val["while_i"]].set(loss_validation)
        )

        if save_position_history:
            pos_hist = val["history"]["position"]
            val["history"]["position"] = jax.tree.map(
                lambda d, pos: d.at[val["while_i"]].set(pos), pos_hist, val["position"]
            )

        if progress_bar:
            jax.debug.callback(tqdm_callback, val)

        return val

    # ---------------------------------------------------------------------------------
    # Run while loop

    val = jax.lax.while_loop(
        cond_fun=lambda val: stopper.continue_(
            val["while_i"], val["history"]["loss_validation"]
        ),
        body_fun=body_fun,
        init_val=init_val,
    )

    max_iter = val["while_i"]

    # ---------------------------------------------------------------------------------
    # Set final position and model state
    stopper.patience = user_patience
    ibest = stopper.which_best_in_recent_history(
        i=max_iter, loss_history=val["history"]["loss_validation"]
    )

    if restore_best_position:
        final_position: Position = {
            name: pos[ibest] for name, pos in val["history"]["position"].items()
        }  # type: ignore
    else:
        final_position = val["position"]

    final_state = interface_train.update_state(final_position, model_train.state)

    # ---------------------------------------------------------------------------------
    # Set unused values in history to nan

    val["history"]["loss_train"] = (
        val["history"]["loss_train"].at[(max_iter + 1) :].set(jnp.nan)
    )
    val["history"]["loss_validation"] = (
        val["history"]["loss_validation"].at[(max_iter + 1) :].set(jnp.nan)
    )
    if save_position_history:
        for name, value in val["history"]["position"].items():
            val["history"]["position"][name] = value.at[(max_iter + 1) :, ...].set(
                jnp.nan
            )

    # ---------------------------------------------------------------------------------
    # Remove unused values in history, if applicable

    if prune_history:
        val["history"]["loss_train"] = val["history"]["loss_train"][: (max_iter + 1)]
        val["history"]["loss_validation"] = val["history"]["loss_validation"][
            : (max_iter + 1)
        ]
        if save_position_history:
            for name, value in val["history"]["position"].items():
                val["history"]["position"][name] = value[: (max_iter + 1), ...]

    # ---------------------------------------------------------------------------------
    # Initialize results object and return

    result = OptimResult(
        model_state=final_state,
        position=final_position,
        iteration=max_iter,
        iteration_best=ibest,
        history=val["history"],
        max_iter=stopper.max_iter,
        n_train=n_train,
        n_validation=n_validation,
    )

    return result


def history_to_df(history: dict[str, Array]) -> pd.DataFrame:
    """
    Turns a :attr:`.OptimResult.history` dictionary into a ``pandas.DataFrame``.
    """
    data: dict[str, Array] = dict()

    position_history = history.get("position", None)

    for name, value in history.items():
        if name == "position":
            continue
        data |= array_to_dict(value, names_prefix=name)

    if position_history is not None:
        for name, value in position_history.items():
            data |= array_to_dict(value, names_prefix=name)

    df = pd.DataFrame(data)
    df["iteration"] = np.arange(value.shape[0])

    return df.astype(float)
