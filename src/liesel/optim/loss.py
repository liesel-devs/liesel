"""Loss protocols and concrete losses for experimental optimizers.

This module defines the interface consumed by :class:`.OptimEngine` and provides
the default negative log-probability loss for Liesel models.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import networkx as nx
from jax.flatten_util import ravel_pytree

from ..model import Calc, Model
from ..model.model import _reduced_sum
from ._log_lik import validate_likelihood_groups
from ._model_utils import continuous_coordinate_nodes, validate_model_data_keys
from .approximation import (
    LaplaceApproximation,
    _positive_definite,
    _prepare_approximation,
)
from .split import PositionSplit, PositionSplitManager
from .types import Position

if TYPE_CHECKING:
    from .state import OptimCarry, OptimResult

SplitConfig = PositionSplit | PositionSplitManager


def _all_finite(tree):
    return jnp.all(jnp.array([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


def _check_evaluation(loss, carry, value, proposal, gradient=None):
    """Classify stateful evaluations without interpreting their opaque state."""
    if carry.loss_state is None:
        return carry
    reason = getattr(loss, "_evaluation_failure", lambda *_: 0)(
        value, proposal, gradient
    )
    reason = jnp.where(
        reason != 0,
        reason,
        jnp.where(~jnp.isfinite(value), -1, jnp.where(_all_finite(gradient), 0, -2)),
    )
    return carry._record_failure(reason, proposal)


def _training_loss_scalar(split: SplitConfig) -> float:
    if isinstance(split, PositionSplitManager):
        return sum(split.train_sample_sizes)

    return split.train_sample_size


def _validate_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(  # noqa: TRY004
            f"{name} must be True or False, but got {value!r}."
        )


def _validate_model_decomposition(model: Model) -> None:
    """Require the factorization used by the built-in split/batch loss."""
    likelihood = Counter(
        var.dist_node.name for var in model.observed.values() if var.has_dist
    )
    prior = Counter(
        var.dist_node.name for var in model.parameters.values() if var.has_dist
    )
    for name, expected in (
        ("_model_log_lik", likelihood),
        ("_model_log_prior", prior),
        ("_model_log_prob", likelihood + prior),
    ):
        node = model.nodes[name]
        actual = Counter(parent.name for parent in node.inputs)
        standard_sum = (
            isinstance(node, Calc)
            and node.function is _reduced_sum
            and not node.kwinputs
        )
        if not standard_sum or actual != expected:
            hint = ""
            if name == "_model_log_prob" and standard_sum:
                unexpected = actual - expected
                observed_values = {var.value_node for var in model.observed.values()}
                for var in model.vars.values():
                    if (
                        var.weak
                        and not var.observed
                        and not var.parameter
                        and var.dist_node is not None
                        and var.dist_node.name in unexpected
                        and observed_values
                        & nx.ancestors(model.node_graph, var.value_node)
                    ):
                        hint += (
                            f" {var.name!r} is a weak variable with a distribution "
                            "that is neither observed nor a parameter and depends "
                            "on observed data. If it represents part of the "
                            f"likelihood, set model.vars[{var.name!r}].observed = True."
                        )
            raise ValueError(
                f"NegLogProbLoss cannot decompose {name!r}: expected the standard "
                "sum of observed likelihoods and parameter priors. "
                f"Unexpected inputs: {list((actual - expected).elements())}; "
                f"missing inputs: {list((expected - actual).elements())}. "
                "Custom aggregate objectives require a custom Loss. Unclassified "
                "factors must be assigned their intended role or handled by a "
                "custom Loss. A manual split does not change the objective "
                f"decomposition.{hint}"
            )


class Loss(Protocol):
    """
    Protocol for optimizer losses.

    ``OptimEngine`` is intentionally agnostic about the concrete loss type. Any
    object satisfying this protocol can be optimized: it must expose the split used
    for training and validation, provide initial parameter positions, compute
    training and validation losses, and provide gradients for optimizer updates.

    Notes
    -----
    The standard built-in optimizer calls :meth:`value_and_grad` so the objective
    value used for monitoring and its gradient share one stochastic evaluation.
    Custom losses can inherit from :class:`LossMixin` to get :meth:`value_and_grad`
    and :meth:`grad` automatically.
    """

    @property
    def split(self) -> SplitConfig:
        """Train/validation/test split used by this loss."""
        ...

    def position(self, position_keys: Sequence[str]) -> Position:
        """
        Extracts an initial optimizer position.

        Parameters
        ----------
        position_keys
            Parameter names requested by the engine's optimizers.
        """
        ...

    @property
    def default_position_keys(self) -> Sequence[str] | None:
        """Optional default optimizer keys; None uses the model's parameters."""
        ...

    def init_state(self, params: Position, carry: "OptimCarry") -> Any:
        """Initial loss-state PyTree, or None for a stateless loss."""
        ...

    def loss_train_batched(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, Any]:
        """
        Computes the training loss for the current mini-batch.

        ``carry.batch`` contains the observed mini-batch and ``carry.fixed_position``
        contains parameters currently owned by other optimizers. Returns
        ``(value, proposed_state)``; stateless losses return ``(value, None)``.
        """
        ...

    def loss_train(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, Any]:
        """
        Computes the full-data training loss at ``params``.

        The engine calls this method for ``loss_monitor="train_full_data"`` after
        every epoch, at the final post-update position. It commits the proposed
        state only when this value is finite. All other proposals are discarded.
        """
        ...

    def loss_monitor(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, Any]:
        """Computes the complete validation monitoring loss at ``params``."""
        ...

    def value_and_grad(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[tuple[jax.Array, Any], Position]:
        """Returns ``(loss_train_batched(params, carry), grad)``."""
        ...

    def grad(self, params: Position, carry: "OptimCarry") -> Position:
        """Returns the gradient of :meth:`loss_train_batched` with respect to params."""
        ...


class LossMixin:
    """
    Shared convenience implementation for differentiable losses.

    Subclasses must define :attr:`split` and :meth:`loss_train_batched`. They should
    also define :meth:`loss_train` if they support exact full-data monitoring. The
    mixin provides validation-position helpers and JAX gradient methods used by
    :class:`.Optimizer`.

    Examples
    --------
    A minimal quadratic loss can inherit from ``LossMixin`` and immediately use the
    gradient helpers:

    >>> import jax.numpy as jnp
    >>> from liesel.optim import LossMixin, PositionSplit
    >>> from liesel.optim.types import Position
    >>> class Quadratic(LossMixin):
    ...     def __init__(self):
    ...         self.split = PositionSplit(
    ...             Position({"y": jnp.array([0.0])}),
    ...             Position({}),
    ...             Position({}),
    ...             1,
    ...             0,
    ...             0,
    ...         )
    ...
    ...     def loss_train_batched(self, params, carry):
    ...         del carry
    ...         return params["x"] ** 2, None
    >>> loss = Quadratic()
    >>> loss.grad(Position({"x": jnp.array(3.0)}), carry=None)["x"]
    Array(6., dtype=float32, weak_type=True)
    >>> loss.obs_validate["y"].tolist()
    [0.0]
    """

    split: SplitConfig
    """Train/validation/test split used by the loss."""

    default_position_keys: Sequence[str] | None = None
    """Optional default optimizer keys; None preserves the model-based default."""

    loss_train_batched: Callable[[Position, "OptimCarry"], tuple[jax.Array, Any]]
    """Training objective differentiated by :meth:`grad` and :meth:`value_and_grad`."""

    def init_state(self, params: Position, carry: "OptimCarry") -> Any:
        """Returns None for a stateless loss; override to initialize loss state.

        Evaluations return a proposed state without mutating ``carry.loss_state``.
        Its PyTree structure, shapes, and dtypes must match this initial state.
        Stateful losses require full-data batches and full-training monitoring.
        """
        return None

    def _evaluation_failure(self, value, proposed_state, gradient):
        """Optional state-specific failure code; gradient is None at monitoring."""
        return 0

    def _failure_message(self, reason, failed_state) -> str | None:
        """Optional explanation for a positive state-specific failure code."""
        return None

    def _checkpoint_configuration(self) -> tuple | None:
        """Optional configuration metadata checked before checkpoint recovery."""
        return None

    def loss_train(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, Any]:
        """
        Computes the full-data training loss.

        The base mixin does not know how to assemble full training data for arbitrary
        custom losses. Subclasses can implement this method to support
        ``OptimEngine(loss_monitor="train_full_data")``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement loss_train(). Pass "
            "EmaTrainLossMonitor(effective_window=...) or implement loss_train() "
            "on the custom loss."
        )

    @property
    def obs_validate(self) -> Position:
        """
        Observed position used for validation.

        If the split has no validation part, the training position is returned.
        """
        if not self.split.has_validation:
            return self.split.train

        return self.split.validate

    @property
    def validate_sample_scale(self) -> float:
        """
        Scalar validation likelihood scale.

        Returns ``1.0`` when no validation split exists. For multi-branch splits with
        unequal branch scales, this property can raise ``ValueError`` through the
        split object; use split-aware scaled likelihood methods in concrete losses.
        """
        if not self.split.has_validation:
            return 1.0

        return self.split.validate_sample_scale

    @property
    def validate_axis_size(self) -> int:
        """
        Number of observations used by validation.

        If no validation split exists, this returns the training axis size.
        """
        if not self.split.has_validation:
            return self.split.train_axis_size

        return self.split.validate_axis_size

    def value_and_grad(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[tuple[jax.Array, Any], Position]:
        """
        Evaluates :meth:`loss_train_batched` and its gradient.

        Parameters
        ----------
        params
            Optimized parameter subset.
        carry
            Current optimizer carry.

        Returns
        -------
        tuple
            ``((value, proposed_state), grad_tree)``. Auxiliary state is not
            differentiated or committed by this method.
        """
        grad_ = jax.value_and_grad(self.loss_train_batched, argnums=0, has_aux=True)
        value, grad_tree = grad_(params, carry)
        return value, Position(grad_tree)

    def grad(self, params: Position, carry: "OptimCarry") -> Position:
        """
        Computes the gradient of :meth:`loss_train_batched`.

        Parameters
        ----------
        params
            Optimized parameter subset.
        carry
            Current optimizer carry.

        Returns
        -------
        Position
            Gradient tree with the same keys as ``params``.
        """
        grad_ = jax.grad(self.loss_train_batched, argnums=0, has_aux=True)
        grad_tree, _ = grad_(params, carry)
        return Position(grad_tree)


class NegLogProbLoss(LossMixin):
    """
    Negative log-probability loss for Liesel models.

    The training objective is the negative sum of the model log-likelihood and
    log-prior. During mini-batch optimization, likelihood terms are scaled through
    ``carry.batches.scaled_log_lik(...)`` so :class:`.BatchManager` can apply
    branch-specific scaling for multi-size observed data. Validation loss uses
    ``split.scaled_log_lik(...)`` for the same reason.

    The model must use the standard sums of observed distribution factors and
    parameter priors. Weak observed variables and weak parameters contribute
    their likelihoods and priors like strong ones.
    Custom aggregate nodes or additional unclassified distribution factors
    require a custom :class:`Loss`; supplying a manual split is not sufficient.

    Parameters
    ----------
    model
        Liesel model evaluated by the loss.
    split
        Train/validation/test split. Use :class:`.PositionSplitManager` for models
        with observed branches of different sample sizes.
    validation_strategy
        Validation objective. ``"log_lik"`` uses the scaled log-likelihood only.
        ``"log_prob"`` also includes the model log-prior.
    scale
        If ``True``, divide losses by the training sample size. For
        :class:`.PositionSplitManager`, the scalar is the sum of all branch-specific
        training sizes.

    Examples
    --------
    Construct a default loss for a simple observed model:

    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.optim import NegLogProbLoss, PositionSplit
    >>> y = lsl.Var.new_obs(
    ...     jnp.arange(3.0),
    ...     lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
    ...     name="y",
    ... )
    >>> model = lsl.Model([y])
    >>> split = PositionSplit.from_model(model, position_keys=["y"])
    >>> loss = NegLogProbLoss(model, split)
    >>> loss.position([]) == {}
    True
    >>> repr(loss)
    'NegLogProbLoss(validation_strategy=log_lik)'
    """

    def __init__(
        self,
        model: Model,
        split: SplitConfig,
        validation_strategy: Literal["log_lik", "log_prob"] = "log_lik",
        scale: bool = False,
    ):
        _validate_model_decomposition(model)
        validate_model_data_keys(model, split.position_keys)
        splits = split.splits if isinstance(split, PositionSplitManager) else (split,)
        validate_likelihood_groups(model, [part.split_position_keys for part in splits])
        self._model = model
        self.split = split
        if validation_strategy not in ("log_lik", "log_prob"):
            raise ValueError(
                "validation_strategy must be 'log_lik' or 'log_prob', but got "
                f"{validation_strategy!r}."
            )
        _validate_bool(scale, "scale")
        self.validation_strategy = validation_strategy
        self.scale = scale
        self.scalar = _training_loss_scalar(self.split) if self.scale else 1.0

    @property
    def model(self) -> Model:
        """
        Liesel model evaluated by the loss.

        Returns
        -------
        Model
            Model passed to :class:`NegLogProbLoss`.
        """
        return self._model

    def position(self, position_keys: Sequence[str]) -> Position:
        """
        Extracts an initial optimizer position from the model.

        Parameters
        ----------
        position_keys
            Model position keys requested by optimizers.

        Returns
        -------
        Position
            Model position restricted to ``position_keys``.
        """
        for key in position_keys:
            if key in self.model.vars and self.model.vars[key].weak:
                raise RuntimeError(
                    f"Cannot optimize weak variable {key!r}; name its strong source "
                    "instead."
                )
        return self.model.extract_position(position_keys)

    def loss_train_batched(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, None]:
        """
        Computes mini-batch negative log posterior.

        Parameters
        ----------
        params
            Optimized parameter subset.
        carry
            Current optimizer carry. ``carry.batch`` supplies observed mini-batch
            values, ``carry.fixed_position`` supplies other parameters, and
            ``carry.batches`` supplies likelihood scaling.

        Returns
        -------
        tuple
            ``(value, None)``: negative scaled log-likelihood plus log-prior,
            optionally normalized by ``self.scalar``.
        """
        position = Position(params | carry.batch | carry.fixed_position)
        states = getattr(carry, "_data_states", {})
        state = states.get("train", carry.model_state)
        new_state = self.model.update_state(position, state, allow_weak_vars=True)

        log_lik = carry.batches.scaled_log_lik(
            self.model, new_state, batch_index=carry.i_batch
        )
        log_prior = new_state["_model_log_prior"].value
        return (-(log_lik + log_prior) / self.scalar), None

    def loss_train(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, None]:
        """
        Computes full-data negative log posterior.

        Parameters
        ----------
        params
            Optimized parameter subset.
        carry
            Current optimizer carry.

        Returns
        -------
        tuple
            ``(value, None)``: negative full-data log-likelihood plus log-prior,
            optionally normalized by ``self.scalar``.
        """
        states = getattr(carry, "_data_states", {})
        data = {} if "train" in states else self.split.train
        position = Position(params | data | carry.fixed_position)
        state = states.get("train", carry.model_state)
        new_state = self.model.update_state(position, state, allow_weak_vars=True)

        log_lik = self.split.scaled_log_lik(self.model, new_state, part="train")
        log_prior = new_state["_model_log_prior"].value
        return (-(log_lik + log_prior) / self.scalar), None

    def loss_monitor(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, None]:
        """
        Computes validation loss.

        Parameters
        ----------
        params
            Optimized parameter subset.
        carry
            Current optimizer carry.

        Returns
        -------
        tuple
            ``(value, None)``: negative scaled validation log-likelihood. If
            ``validation_strategy="log_prob"``, the log-prior is included as well.
        """
        part = "validate" if self.split.has_validation else "train"
        states = getattr(carry, "_data_states", {})
        data = {} if part in states else self.obs_validate
        position = Position(params | data | carry.fixed_position)
        state = states.get(part, carry.model_state)
        new_state = self.model.update_state(position, state, allow_weak_vars=True)
        loss = -self.split.scaled_log_lik(self.model, new_state, part=part)
        if self.validation_strategy == "log_prob":
            loss -= new_state["_model_log_prior"].value

        return (loss / self.scalar), None

    def approximate_joint_posterior(
        self,
        result: "OptimResult",
        *,
        at: str = "min_monitor",
        raise_on_failure: bool = True,
        stationarity_tol: float = 1e-4,
    ) -> LaplaceApproximation:
        """Construct a joint Gaussian approximation at the selected fitted mode.

        Curvature comes from the full training log-likelihood plus log-prior,
        including transformation Jacobians. Validation and test data are excluded,
        and loss normalization (``scale`` or ``LieselOptim.scale_loss``) does not
        affect the approximation. Only optimized parameters are included;
        omitted parameters retain their model values.

        Parameters
        ----------
        result
            Fit result belonging to this model, loss, and training data. Keep
            the model, split, and fixed parameters unchanged after fitting.
        at
            ``"min_monitor"`` (default) uses ``result.position_min_monitor``;
            ``"final"`` uses ``result.position_final``. A minimum validation or
            EMA monitoring loss need not identify a stationary posterior mode.
        raise_on_failure
            Raise RuntimeError on an invalid approximation (default True).
            False returns an inspectable object with ``valid=False`` whose
            sampling, covariance, and block methods raise. Invalid argument
            values always raise ValueError.
        stationarity_tol
            Positive finite bound on half the squared Newton decrement for the
            unscaled full-training objective. Defaults to 1e-4.

        Notes
        -----
        This optional calculation evaluates a dense Hessian once per call;
        it does not refit the model. Finite values and derivatives, positive
        definite curvature, and stationarity are required. No jitter or
        eigenvalue clipping is used. Diagnostics retain ``value``, ``gradient``,
        ``joint_precision``, and ``newton_decrement_squared`` when available;
        ``reason`` explains a failure. Names are sorted, with parameter entries
        flattened within each name. Use ``Model.predict`` to transform draws
        back to the original parameter scales.

        In hierarchical models, joint MAP can favor vanishing scale parameters.
        :class:`~liesel.optim.LaplaceLoss` instead integrates selected effects
        before optimizing the remaining parameters.
        """
        approximation = _prepare_approximation(
            result, at, raise_on_failure, stationarity_tol
        )
        if approximation.diagnostics["reason"] is not None:
            return approximation
        try:
            data_nodes = {
                self.model._node_for_position_key(k) for k in self.split.train
            }
            continuous_coordinate_nodes(self.model, approximation.names, data_nodes)
            expected = self.position(approximation.names)
        except ValueError as error:
            return approximation._failed(
                f"Incompatible optimized coordinates: {error}", raise_on_failure
            )
        if any(
            jnp.shape(value) != jnp.shape(expected[name])
            or jnp.asarray(value).dtype != jnp.asarray(expected[name]).dtype
            for name, value in approximation.mean.items()
        ):
            return approximation._failed(
                "Selected coordinates have incompatible shapes or dtypes.",
                raise_on_failure,
            )
        flat, unravel = ravel_pytree(approximation.mean)
        if not flat.size:
            return approximation._failed(
                "No optimized coordinates are available.", raise_on_failure
            )

        training_state = self.model.update_state(self.split.train, self.model.state)

        def joint(flat):
            state = self.model.update_state(unravel(flat), training_state)
            log_lik = self.split.scaled_log_lik(self.model, state, part="train")
            return -(log_lik + state["_model_log_prior"].value)

        @jax.jit
        def derivatives(flat):
            def grad_with_value(flat):
                value, gradient = jax.value_and_grad(joint)(flat)
                return gradient, (value, gradient)

            precision, (value, gradient) = jax.jacfwd(grad_with_value, has_aux=True)(
                flat
            )
            return value, gradient, precision

        value, gradient, precision = derivatives(flat)
        factor = jnp.linalg.cholesky(precision)
        decrement = gradient @ jsp.linalg.cho_solve((factor, True), gradient)
        approximation.diagnostics.update(
            value=value,
            gradient=gradient,
            joint_precision=precision,
            newton_decrement_squared=decrement,
        )
        if not bool(
            jnp.isfinite(value)
            & jnp.isfinite(gradient).all()
            & jnp.isfinite(precision).all()
        ):
            return approximation._failed(
                "Non-finite posterior value or derivatives.", raise_on_failure
            )
        if not bool(_positive_definite(precision, factor)):
            return approximation._failed(
                "Invalid posterior curvature: positive definiteness is required.",
                raise_on_failure,
            )
        if not bool(decrement / 2 <= stationarity_tol):
            return approximation._failed(
                "Stationarity check failed: half the squared Newton decrement "
                "exceeds stationarity_tol.",
                raise_on_failure,
            )
        approximation.precision_cholesky = factor
        approximation.valid = True
        return approximation

    def __repr__(self) -> str:
        """Returns a compact representation showing the validation strategy."""
        name = type(self).__name__
        out = f"{name}(validation_strategy={self.validation_strategy})"
        return out
