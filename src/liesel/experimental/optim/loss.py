"""Loss protocols and concrete losses for experimental optimizers.

This module defines the interface consumed by :class:`.OptimEngine` and provides
the default negative log-probability loss for Liesel models. The same protocol is
also implemented by variational losses such as :class:`.Elbo`.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Protocol

import jax

from ...model import Model
from .split import PositionSplit, PositionSplitManager
from .types import Position

if TYPE_CHECKING:
    from .state import OptimCarry

SplitConfig = PositionSplit | PositionSplitManager


def _training_loss_scalar(split: SplitConfig) -> int:
    if isinstance(split, PositionSplitManager):
        return sum(split.n_trains)

    return split.n_train


class Loss(Protocol):
    """
    Protocol for optimizer losses.

    ``OptimEngine`` is intentionally agnostic about the concrete loss type. Any
    object satisfying this protocol can be optimized: it must expose the split used
    for training and validation, provide initial parameter positions, compute
    training and validation losses, and provide gradients for optimizer updates.

    Attributes
    ----------
    split
        Train/validation/test split used by the loss.

    Notes
    -----
    Built-in optimizers call :meth:`grad` or :meth:`value_and_grad` on
    ``loss_train_batched``. Custom losses can inherit from :class:`LossMixin` to get
    these gradient methods automatically.
    """

    split: SplitConfig

    def position(self, position_keys: Sequence[str]) -> Position:
        """
        Extracts an initial optimizer position.

        Parameters
        ----------
        position_keys
            Parameter names requested by the engine's optimizers.
        """
        ...

    def loss_train_batched(self, params: Position, carry: "OptimCarry") -> jax.Array:
        """
        Computes the training loss for the current mini-batch.

        ``carry.batch`` contains the observed mini-batch and ``carry.fixed_position``
        contains parameters currently owned by other optimizers.
        """
        ...

    def loss_train(self, params: Position, carry: "OptimCarry") -> jax.Array:
        """
        Computes the full-data training loss at ``params``.

        The engine calls this method only for exact training-data monitoring when no
        validation split is available.
        """
        ...

    def loss_validate(self, params: Position, carry: "OptimCarry") -> jax.Array:
        """Computes the validation loss at ``params``."""
        ...

    def value_and_grad(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, Position]:
        """Returns ``(loss_train_batched(params, carry), grad)``."""
        ...

    def grad(self, params: Position, carry: "OptimCarry") -> Position:
        """Returns the gradient of :meth:`loss_train_batched` with respect to params."""
        ...


class LossMixin:
    """
    Shared convenience implementation for differentiable losses.

    Subclasses must define :attr:`split` and :meth:`loss_train_batched`. They should
    also define :meth:`loss_train` if they support exact full-data monitoring when
    no validation split is available. The mixin provides validation-position helpers
    and JAX gradient methods used by :class:`.Optimizer`.

    Attributes
    ----------
    split
        Train/validation/test split used by the loss.
    loss_train_batched
        Callable training objective differentiated by :meth:`grad` and
        :meth:`value_and_grad`.

    Examples
    --------
    A minimal quadratic loss can inherit from ``LossMixin`` and immediately use the
    gradient helpers:

    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim import PositionSplit
    >>> from liesel.experimental.optim.loss import LossMixin
    >>> from liesel.experimental.optim.types import Position
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
    ...         return params["x"] ** 2
    >>> loss = Quadratic()
    >>> loss.grad(Position({"x": jnp.array(3.0)}), carry=None)["x"]
    Array(6., dtype=float32, weak_type=True)
    >>> loss.obs_validate["y"].tolist()
    [0.0]
    """

    split: SplitConfig
    loss_train_batched: Callable[[Position, "OptimCarry"], jax.Array]

    def loss_train(self, params: Position, carry: "OptimCarry") -> jax.Array:
        """
        Computes the full-data training loss.

        The base mixin does not know how to assemble full training data for arbitrary
        custom losses. Subclasses can implement this method to support
        ``OptimEngine(train_monitor="full_data")`` without a validation split.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement loss_train(). Use "
            "OptimEngine(train_monitor='epoch_average') or implement loss_train() "
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
    def scale_validate(self) -> float:
        """
        Scalar validation likelihood scale.

        Returns ``1.0`` when no validation split exists. For multi-branch splits with
        unequal branch scales, this property can raise ``ValueError`` through the
        split object; use split-aware scaled likelihood methods in concrete losses.
        """
        if not self.split.has_validation:
            return 1.0

        return self.split.scale_validate

    @property
    def n_validate(self) -> int:
        """
        Number of observations used by validation.

        If no validation split exists, this returns the training sample size.
        """
        if not self.split.has_validation:
            return self.split.n_train

        return self.split.n_validate

    def value_and_grad(
        self, params: Position, carry: "OptimCarry"
    ) -> tuple[jax.Array, Position]:
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
            Pair ``(value, grad_tree)`` as returned by :func:`jax.value_and_grad`.
        """
        grad_ = jax.value_and_grad(self.loss_train_batched, argnums=0)
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
        grad_ = jax.grad(self.loss_train_batched, argnums=0)
        grad_tree = grad_(params, carry)
        return Position(grad_tree)


class NegLogProbLoss(LossMixin):
    """
    Negative log-probability loss for Liesel models.

    The training objective is the negative sum of the model log-likelihood and
    log-prior. During mini-batch optimization, likelihood terms are scaled through
    ``carry.batches.scaled_log_lik(...)`` so :class:`.BatchManager` can apply
    branch-specific scaling for multi-size observed data. Validation loss uses
    ``split.scaled_log_lik(...)`` for the same reason.

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
    >>> from liesel.experimental.optim import NegLogProbLoss, PositionSplit
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
        self._model = model
        self.split = split
        if validation_strategy not in ("log_lik", "log_prob"):
            raise ValueError(
                "validation_strategy must be 'log_lik' or 'log_prob', but got "
                f"{validation_strategy!r}."
            )
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
        return self.model.extract_position(position_keys)

    def loss_train_batched(self, params: Position, carry: "OptimCarry") -> jax.Array:
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
        jax.Array
            Negative scaled log-likelihood plus log-prior, optionally normalized by
            ``self.scalar``.
        """
        position = Position(params | carry.batch | carry.fixed_position)
        new_state = self.model.update_state(position, carry.model_state)

        log_lik = carry.batches.scaled_log_lik(self.model, new_state)
        log_prior = new_state["_model_log_prior"].value
        return -(log_lik + log_prior) / self.scalar

    def loss_train(self, params: Position, carry: "OptimCarry") -> jax.Array:
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
        jax.Array
            Negative full-data log-likelihood plus log-prior, optionally normalized
            by ``self.scalar``.
        """
        position = Position(params | self.split.train | carry.fixed_position)
        new_state = self.model.update_state(position, carry.model_state)

        log_lik = self.split.scaled_log_lik(self.model, new_state, part="train")
        log_prior = new_state["_model_log_prior"].value
        return -(log_lik + log_prior) / self.scalar

    def loss_validate(self, params: Position, carry: "OptimCarry") -> jax.Array:
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
        jax.Array
            Negative scaled validation log-likelihood. If
            ``validation_strategy="log_prob"``, the log-prior is included as well.
        """
        position = Position(params | self.obs_validate | carry.fixed_position)
        new_state = self.model.update_state(position, carry.model_state)
        loss = -self.split.scaled_log_lik(self.model, new_state, part="validate")
        if self.validation_strategy == "log_prob":
            loss -= new_state["_model_log_prior"].value

        return loss / self.scalar

    def __repr__(self) -> str:
        """Returns a compact representation showing the validation strategy."""
        name = type(self).__name__
        out = f"{name}(validation_strategy={self.validation_strategy})"
        return out
