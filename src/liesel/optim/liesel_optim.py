"""Opinionated optimization setup for Liesel models."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import optax

from ..model import Model
from ._engine_utils import (
    BatchConfig,
    SplitConfig,
    _validate_optimizer_batches,
    _validate_positive_int,
)
from .batch import Batches
from .engine import EmaTrainLossMonitor, LossMonitor
from .loss import Loss, NegLogProbLoss, _validate_bool
from .optimizer import LBFGS, Optimizer, OptimizerLike
from .split import PositionSplit
from .stop import Stopper

if TYPE_CHECKING:
    from .engine import OptimEngine
    from .state import OptimResult


class LieselOptim[LossType: Loss = NegLogProbLoss]:
    """
    Builds an :class:`.OptimEngine` for a Liesel model using sensible defaults.

    ``LieselOptim`` is the quick-start wrapper for regular model optimization. It
    creates a negative log-posterior loss, full-data training batches,
    and wraps a supplied Optax transformation over all model parameters. Call
    :meth:`build_engine` to inspect or modify the low-level engine
    before fitting, or :meth:`fit` for the direct path.

    Parameters
    ----------
    model
        Liesel model to optimize.
    loss_monitor
        Source for the epoch-level stopping and progress loss. Pass
        :class:`.EmaTrainLossMonitor` for a continuous EMA of pre-update losses,
        ``"validation"`` for one complete validation-loss evaluation after each
        epoch, or ``"train_full_data"`` for one complete training-loss evaluation
        after each epoch. Exact monitors use the post-update epoch position.
        ``"train_full_data"`` always adds the full-data evaluation, including with
        one full-data optimization batch.
    optimizers
        Required optimizer choice. An Optax transformation such as
        ``optax.adam(0.01)`` applies to all model parameters. Pass ``"lbfgs"``
        for the built-in full-data L-BFGS optimizer. A custom loss can override
        the automatic keys through ``default_position_keys``. Pass a sequence of
        explicit optimizers for selected parameters. If a parameter is weak (computed),
        automatic selection raises an error. Explicitly name its strong source
        variables with :class:`.Optimizer` or :class:`.LBFGS`; those sources need
        not be marked as parameters. Priors on weak parameters remain in the loss.
        Pass a configured transformation, not an optimizer factory.
        Transformations must support updates from
        gradients, state and parameters without extra objective arguments.
    stopper
        Maximum-epoch and early-stopping configuration. ``None`` creates a new
        :class:`.Stopper` with ``epochs=1000``, ``patience=10``, and ``rtol=1e-6``.
    seed
        Integer seed for batch shuffling or random batch sampling, and for custom
        losses or optimizers that use ``carry.key``. Starting parameter values
        come from ``model``. Set the seed for a random training/validation/test
        split separately when creating that split. Defaults to ``0``. Explicit
        ``None`` uses the current Unix time in whole seconds.
    split
        Optional split. If neither ``split`` nor ``loss`` is supplied, all observed
        data is used for training; weak observations are recomputed from their
        strong inputs. Multi-size observed data automatically uses
        :class:`.PositionSplitManager`. With a custom loss, an explicit split must
        be the same object as ``loss.split``. Models with ``per_obs=False``
        require an explicitly constructed split: use
        :meth:`.PositionSplit.from_model` with ``infer_sample_sizes=False`` for
        axis counts, or supply effective ``sample_sizes`` there. Custom aggregate
        likelihood or probability nodes require a custom loss as well as an
        explicit split.
    batch_size
        Rows per batch in each training group. Uses :meth:`.Batches.from_split`
        with otherwise default settings. ``None`` uses all training data.
        Pass ``batches`` instead for custom settings, such as non-leading axes.
    batches
        Explicit batch configuration. Cannot be combined with a non-``None``
        ``batch_size``.
    loss
        Custom loss. Uses ``loss.split`` and overrides ``validation_strategy`` and
        ``scale_loss``.
    validation_strategy
        Validation strategy passed to :class:`.NegLogProbLoss` when ``loss`` is not
        supplied.
    scale_loss
        Whether the default :class:`.NegLogProbLoss` should divide losses by the
        training sample size. Defaults to ``True``.
        This setting has no effect when ``loss`` is supplied.
    save_position_history
        Whether to save parameter values at every epoch. Defaults to ``True``.
        History is allocated for the maximum epoch budget before fitting: its
        size is approximately epochs times the total parameter bytes (4 GB for
        1,000 epochs and one million float32 parameters). Set ``False`` to skip
        this allocation; losses and final/best positions remain available.
    show_progress
        Whether the built engine should show ``tqdm`` progress bars.
    show_step_progress
        Whether to show an additional progress bar for batches within each epoch
        when ``show_progress`` is enabled.
    progress_update_every
        Update the epoch progress bar after this many completed epochs. When batch
        progress is active, the epoch bar advances after every epoch.
    step_progress_update_every
        Update the batch progress bar after this many completed batches.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import optax
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.optim import EmaTrainLossMonitor, LieselOptim
    >>> loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    >>> y = lsl.Var.new_obs(
    ...     jnp.array([0.0, 1.0]),
    ...     lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
    ...     name="y",
    ... )
    >>> model = lsl.Model([y])
    >>> engine = LieselOptim(
    ...     model,
    ...     optimizers=optax.adam(0.01),
    ...     loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
    ...     seed=1,
    ... ).build_engine()
    >>> type(engine).__name__
    'OptimEngine'
    """

    def __init__(
        self,
        model: Model,
        *,
        loss_monitor: LossMonitor,
        optimizers: optax.GradientTransformation
        | Sequence[OptimizerLike]
        | Literal["lbfgs"],
        stopper: Stopper | None = None,
        seed: int | None = 0,
        split: SplitConfig | None = None,
        batch_size: int | None = None,
        batches: BatchConfig | None = None,
        loss: LossType | None = None,
        validation_strategy: Literal["log_lik", "log_prob"] = "log_lik",
        scale_loss: bool = True,
        save_position_history: bool = True,
        show_progress: bool = True,
        show_step_progress: bool = False,
        progress_update_every: int = 10,
        step_progress_update_every: int = 10,
    ) -> None:
        if batches is not None and batch_size is not None:
            raise ValueError("Pass either batch_size or batches, not both.")
        self.model = model
        self.seed = int(time.time()) if seed is None else seed
        self.stopper = (
            Stopper(epochs=1000, patience=10, rtol=1e-6) if stopper is None else stopper
        )
        self.split = self._resolve_split(loss, split)
        self.loss_monitor = loss_monitor
        if not isinstance(loss_monitor, EmaTrainLossMonitor) and loss_monitor not in (
            "validation",
            "train_full_data",
        ):
            raise ValueError(
                "loss_monitor must be EmaTrainLossMonitor(effective_window=...), "
                "'validation', or "
                f"'train_full_data', but got {loss_monitor!r}."
            )
        if loss_monitor == "validation" and not self.split.has_validation:
            raise ValueError(
                "loss_monitor='validation' requires a split with validation data."
            )
        self.loss = self._resolve_loss(
            loss, validation_strategy=validation_strategy, scale_loss=scale_loss
        )
        self.batches = (
            Batches.from_split(self.split, batch_size=batch_size)
            if batches is None
            else batches
        )
        self.optimizers = self._resolve_optimizers(optimizers)
        self.save_position_history = save_position_history
        _validate_optimizer_batches(self.optimizers, self.batches)
        self.show_progress = show_progress
        self.progress_update_every = progress_update_every
        self.show_step_progress = show_step_progress
        self.step_progress_update_every = step_progress_update_every
        _validate_positive_int(self.progress_update_every, "progress_update_every")
        _validate_positive_int(
            self.step_progress_update_every, "step_progress_update_every"
        )

    def _resolve_split(
        self,
        loss: Loss | None,
        split: SplitConfig | None,
    ) -> SplitConfig:
        if loss is not None:
            if split is not None and split is not loss.split:
                raise ValueError(
                    "When both loss and split are provided, split must be loss.split."
                )

            return loss.split

        if split is not None:
            return split

        return PositionSplit.from_model(
            self.model,
            multi_size="manager",
            shuffle=False,
        )

    def _resolve_loss(
        self,
        loss: LossType | None,
        validation_strategy: Literal["log_lik", "log_prob"],
        scale_loss: bool,
    ) -> LossType | NegLogProbLoss:
        if loss is not None:
            return loss

        _validate_bool(scale_loss, "scale_loss")
        return NegLogProbLoss(
            self.model,
            self.split,
            validation_strategy=validation_strategy,
            scale=scale_loss,
        )

    def _resolve_optimizers(
        self,
        optimizers: optax.GradientTransformation
        | Sequence[OptimizerLike]
        | Literal["lbfgs"],
    ) -> Sequence[OptimizerLike]:
        default_keys = self.loss.default_position_keys
        position_keys = list(
            self.model.parameters if default_keys is None else default_keys
        )
        if isinstance(optimizers, optax.GradientTransformation) or (
            isinstance(optimizers, str) and optimizers == "lbfgs"
        ):
            weak = [
                name
                for name in position_keys
                if name in self.model.vars and self.model.vars[name].weak
            ]
            if weak:
                raise ValueError(
                    f"Cannot automatically select weak parameters {weak}: their "
                    "values are computed from other variables. Explicitly name the "
                    "strong variables to estimate with "
                    "optimizers=[Optimizer(['strong_name'], optax.adam(0.01))] "
                    "or optimizers=[LBFGS(['strong_name'])]. Those strong variables "
                    "need not be marked as parameters; priors on weak parameters "
                    "remain part of the loss."
                )
        if isinstance(optimizers, optax.GradientTransformation):
            return [Optimizer(position_keys, optimizers)]
        if isinstance(optimizers, str):
            if optimizers == "lbfgs":
                return [LBFGS(position_keys)]
            raise ValueError(
                "The only optimizer string is 'lbfgs'. For Adam, pass a configured "
                "Optax transformation as optimizers=optax.adam(learning_rate=...)."
            )
        if isinstance(optimizers, Sequence):
            for index, optimizer in enumerate(optimizers):
                if isinstance(optimizer, optax.GradientTransformation):
                    raise TypeError(
                        f"optimizers[{index}] is a bare Optax transformation. Pass "
                        "a single transformation directly for all parameters, or "
                        "wrap each one in Optimizer(keys, transformation) for "
                        "separate parameter blocks."
                    )
            return optimizers
        raise TypeError(
            "optimizers must be a configured Optax transformation such as "
            "optax.adam(learning_rate=...), 'lbfgs', or a sequence of optimizers. "
            "Pass the transformation, not the optimizer factory."
        )

    def build_engine(self) -> OptimEngine:
        """
        Builds the low-level optimization engine.

        Returns
        -------
        OptimEngine
            Configured engine. Users may modify engine attributes before calling
            :meth:`OptimEngine.fit`.
        """
        from .engine import OptimEngine

        return OptimEngine(
            loss=self.loss,
            batches=self.batches,
            optimizers=self.optimizers,
            stopper=self.stopper,
            initial_state=self.model.state,
            seed=self.seed,
            loss_monitor=self.loss_monitor,
            save_position_history=self.save_position_history,
            show_progress=self.show_progress,
            progress_update_every=self.progress_update_every,
            show_step_progress=self.show_step_progress,
            step_progress_update_every=self.step_progress_update_every,
        )

    def fit(self) -> OptimResult:
        """
        Builds an engine and runs optimization immediately.

        Returns
        -------
        OptimResult
            Result returned by :meth:`OptimEngine.fit`.
        """
        return self.build_engine().fit()
