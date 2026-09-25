"""Low-level optimization engine for experimental optimizers.

The :class:`OptimEngine` class coordinates losses, optimizers, mini-batches,
train/validation/test splits, early stopping, and optimizer history recording. Most
users will usually construct it through :class:`.LieselOptim`, but direct construction
is useful for custom losses or optimizer schedules.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from ._engine_utils import (
    BatchConfig,
    SplitConfig,
    _validate_optimizer_batches,
    _validate_positive_int,
)
from .batch import Batches, BatchManager
from .laplace import LaplaceLoss
from .loss import Loss, LossMixin, NegLogProbLoss, _check_evaluation
from .optimizer import LBFGS, Optimizer, OptimizerLike
from .split import PositionSplitManager
from .state import (
    _NAN_DEBUG_KIND_LOSS,
    _NAN_DEBUG_KIND_NAMES,
    _NAN_DEBUG_KIND_POSITION_AFTER,
    _NAN_DEBUG_KIND_POSITION_BEFORE,
    OptimCarry,
    OptimCheckpoint,
    OptimHistory,
    OptimNaNDebugInfo,
    OptimNaNDebugState,
    OptimResult,
    _checkpoint_versions,
)
from .stop import Stopper
from .types import ModelState, Position

__all__ = ["EmaTrainLossMonitor", "LossMonitor", "OptimEngine"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmaTrainLossMonitor:
    r"""Configures EMA monitoring of the pre-update mini-batch training loss.

    The pre-update mini-batch training loss is the scalar objective evaluated for
    an optimizer's gradient at its supplied position, before that optimizer applies
    its update. The value and gradient share the same mini-batch, parameter position,
    PRNG key, and stochastic objective draw; monitoring is not an independent loss
    evaluation. The first active optimizer supplies this observation. If none is
    active, the engine evaluates the loss once at the unchanged position.

    Let :math:`e` be ``effective_window`` and :math:`N` the configured number of
    full batches per epoch. The span :math:`W` and smoothing coefficients are

    .. math::

       W = \max(1, eN), \qquad
       \alpha = \frac{2}{W + 1}, \qquad
       \beta = 1 - \alpha.

    Starting from :math:`m_0 = w_0 = 0`, mini-batch observation :math:`t` updates

    .. math::

       m_t = \beta m_{t-1} + \alpha L_t, \qquad
       w_t = \beta w_{t-1} + \alpha, \qquad
       \operatorname{EMA}_t = \frac{m_t}{w_t}.

    Here :math:`L_t` is the pre-update loss, :math:`m_t` its unnormalized
    exponentially weighted sum, :math:`w_t` the accumulated weight,
    :math:`\alpha` the newest-loss weight, and :math:`\beta` the decay applied to
    earlier losses. The index :math:`t` continues across epoch boundaries. Since
    :math:`w_t = 1 - \beta^t`, the bias-correction factor :math:`1 / w_t`
    automatically approaches one without changing the update rule.

    Numerically, the engine stores the normalized EMA directly and uses the
    equivalent update

    .. math::

       w_t = -\operatorname{expm1}(t\operatorname{log1p}(-\alpha)), \qquad
       \operatorname{EMA}_t = \operatorname{EMA}_{t-1}
           + \frac{\alpha}{w_t}(L_t - \operatorname{EMA}_{t-1}).

    The update uses compensated summation to retain small increments. This avoids
    drift from repeatedly accumulating the normalization weight and reduces
    rounding error for long windows, including with float32 and JAX 64-bit mode
    disabled. Ordinary floating-point rounding still applies.

    ``effective_window`` is an EMA span measured in epoch equivalents, not a hard
    inclusion window or half-life. With a typical multi-batch epoch, a span of one
    epoch equivalent has a half-life of roughly 0.35 epoch equivalents, so recent
    batches receive substantially more weight than early batches from the same
    epoch. The :meth:`from_half_life` constructor uses the large-span approximation
    :math:`e = 2h / \log(2)` for a requested half-life :math:`h`. The exact
    finite-step relationship also depends on the number of batches per epoch.
    """

    effective_window: float

    def __post_init__(self) -> None:
        try:
            valid = math.isfinite(self.effective_window) and self.effective_window > 0
        except (TypeError, ValueError):
            valid = False

        if isinstance(self.effective_window, bool) or not valid:
            raise ValueError(
                "effective_window must be finite and positive, but got "
                f"{self.effective_window!r}."
            )

    @classmethod
    def from_half_life(cls, half_life: float) -> EmaTrainLossMonitor:
        """Constructs a monitor from an approximate half-life.

        ``half_life`` is measured in configured epoch equivalents. The conversion
        uses ``effective_window = 2 * half_life / log(2)`` and is approximate because
        the exact finite-step relationship depends on the number of batches per
        epoch.
        """
        try:
            valid = math.isfinite(half_life) and half_life > 0
        except (TypeError, ValueError):
            valid = False

        if isinstance(half_life, bool) or not valid:
            raise ValueError(
                f"half_life must be finite and positive, but got {half_life!r}."
            )

        return cls(effective_window=2.0 * half_life / math.log(2.0))


type LossMonitor = EmaTrainLossMonitor | Literal["validation", "train_full_data"]


def _tree_has_nan(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    has_nan = jnp.asarray(False)

    for leaf in leaves:
        has_nan = has_nan | jnp.any(jnp.isnan(jnp.asarray(leaf)))

    return has_nan


def _tree_where(condition: jax.Array, true_tree, false_tree):
    return jax.tree_util.tree_map(
        lambda true_leaf, false_leaf: jnp.where(condition, true_leaf, false_leaf),
        true_tree,
        false_tree,
    )


def _position_where(
    condition: jax.Array, true_position: Position, false_position: Position
) -> Position:
    return Position(_tree_where(condition, true_position, false_position))


@dataclass(init=False)
class OptimEngine:
    """
    Runs an optimization loop over epochs, batches, and optimizers.

    ``OptimEngine`` runs the fit configured by :class:`.LieselOptim`.
    Each epoch starts by asking ``batches`` for fresh batch indices,
    then iterates over all full batches. For each batch, each optimizer gets a turn
    to update the subset of parameters named in its ``position_keys``. The first
    active optimizer's pre-update loss supplies the batch observation; if none is
    active, the loss is evaluated once at the unchanged position. At the end of the
    epoch, the engine records training and monitoring losses, updates the
    minimum-monitor position, and asks ``stopper`` whether to continue.

    Parameters
    ----------
    loss
        Loss object implementing the :class:`.loss.Loss` protocol.
    batches
        Batch configuration used for the training data. Use :class:`.Batches` for a
        single observation size and :class:`.BatchManager` for multi-branch models
        with different observation sizes.
    optimizers
        Sequence of optimizers. Each optimizer must claim a disjoint set of position
        keys. Individual optimizers may delay activation with
        :attr:`.Optimizer.activate_after_epochs`.
    stopper
        Early-stopping and maximum-epoch configuration.
    seed
        Integer seed or JAX PRNG key for batch shuffling or random batch sampling,
        and for losses or optimizers that use ``carry.key``. Starting parameters
        come from ``loss.position()``; the data split is already defined. Resuming
        a checkpoint uses its saved random key.
    initial_state
        Initial model state passed into :class:`.OptimCarry`.
    prune_history
        If ``True``, remove unused history entries after early stopping.
    show_progress
        Whether to show ``tqdm`` progress bars. This is the master switch for both
        epoch and optional batch progress.
    save_position_history
        Whether to store the full position history. The minimum-monitor position is
        tracked independently of this setting.
    loss_monitor
        Source for the epoch-level stopping and progress loss. Pass
        :class:`EmaTrainLossMonitor` for a continuous EMA of pre-update losses,
        ``"validation"`` for one complete validation-loss evaluation after each
        epoch, or ``"train_full_data"`` for one complete training-loss evaluation
        after each epoch. Exact monitors are evaluated at the final post-update
        epoch position, and ``"train_full_data"`` incurs this additional full-data
        evaluation even when optimization itself uses one full-data batch.
    debug_nans
        Whether to capture first-NaN reproduction data during batch updates. Every
        active optimizer's returned pre-update loss and the updated position are
        checked; when no optimizer is active, the explicit batched loss is checked.
    progress_update_every
        Update the epoch progress bar after this many completed epochs. Defaults to
        10. The final state is always rendered. When batch progress is active, the
        epoch bar advances after every epoch to keep the nested display consistent.
    show_step_progress
        Whether to show an additional progress bar for batches within each epoch
        when ``show_progress`` is enabled.
    step_progress_update_every
        Update the batch progress bar after this many completed batches. Defaults to
        10. The final state of an interrupted epoch is always rendered.

    Notes
    -----
    ``OptimEngine`` uses ``carry.epoch`` as the number of completed epochs and as the
    next history index to be written. This matches :class:`.Stopper`'s
    indexing convention. Built-in :class:`.LBFGS` is accepted only with full-data
    batches and also requires a deterministic objective, which the engine cannot
    validate. Exact monitor minima retain the post-update position used for the
    evaluation. An EMA minimum instead retains the associated parameter snapshot;
    because an EMA combines losses from several positions, it is not an exact
    loss-position pairing.

    Examples
    --------
    Start with :class:`.LieselOptim` for ordinary fits. To assemble the pieces
    yourself:

    >>> import jax.numpy as jnp
    >>> import optax
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> import liesel.model as lsl
    >>> import liesel.optim as opt
    >>> loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    >>> y = lsl.Var.new_obs(
    ...     jnp.array([1.0, 2.0, 3.0]),
    ...     lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
    ...     name="y",
    ... )
    >>> model = lsl.Model([y])
    >>> split = opt.PositionSplit.from_model(model)
    >>> engine = opt.OptimEngine(
    ...     loss=opt.NegLogProbLoss(model, split, scale=True),
    ...     batches=opt.Batches.from_split(split, batch_size=None),
    ...     optimizers=[opt.Optimizer(["loc"], optax.adam(0.01))],
    ...     stopper=opt.Stopper(epochs=5, patience=5),
    ...     initial_state=model.state,
    ...     loss_monitor="train_full_data",
    ...     seed=42,
    ...     show_progress=False,
    ... )
    >>> result = engine.fit()
    >>> result.n_epochs
    5
    """

    loss: Loss
    batches: BatchConfig
    optimizers: Sequence[OptimizerLike]
    stopper: Stopper
    seed: jax.Array
    initial_state: ModelState
    loss_monitor: LossMonitor
    prune_history: bool = True
    show_progress: bool = True
    save_position_history: bool = True
    progress_update_every: int = 10
    debug_nans: bool = False
    show_step_progress: bool = False
    step_progress_update_every: int = 10

    def __init__(
        self,
        loss: Loss,
        batches: BatchConfig,
        optimizers: Sequence[OptimizerLike],
        stopper: Stopper,
        seed: int | jax.Array,
        initial_state: ModelState,
        prune_history: bool = True,
        show_progress: bool = True,
        save_position_history: bool = True,
        *,
        debug_nans: bool = False,
        loss_monitor: LossMonitor,
        progress_update_every: int = 10,
        show_step_progress: bool = False,
        step_progress_update_every: int = 10,
    ) -> None:
        """Initializes an optimization engine."""
        self.loss = loss
        self.batches = batches
        self.optimizers = optimizers
        self.stopper = stopper
        self.seed = (
            jax.random.key(int(seed))
            if isinstance(seed, Integral)
            else cast(jax.Array, seed)
        )
        self.initial_state = initial_state
        self.prune_history = prune_history
        self.show_progress = show_progress
        self.save_position_history = save_position_history
        self.progress_update_every = progress_update_every
        self.loss_monitor = loss_monitor
        self.debug_nans = debug_nans
        self.show_step_progress = show_step_progress
        self.step_progress_update_every = step_progress_update_every

        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Validates engine configuration and normalizes integer seeds.

        Raises
        ------
        ValueError
            If optimizer ownership, batching, split, or progress settings are
            invalid.
        """
        self.optimizers = tuple(self.optimizers)

        if len(self.optimizers) == 0:
            raise ValueError("OptimEngine requires at least one optimizer.")

        self._name_optimizers()
        self._validate_optimizer_identifiers()
        self._validate_position_keys()
        self._validate_optimizer_activation_delays()
        self._validate_progress_settings()
        self._validate_loss_monitor()
        self._validate_debug_nans()
        self._validate_batch_split_compatibility()

    @property
    def split(self) -> SplitConfig:
        """
        Train/validation/test split supplied by :attr:`loss`.

        Returns
        -------
        PositionSplit | PositionSplitManager
            The split object stored on ``self.loss.split``.
        """
        return self.loss.split

    @property
    def position_keys(self) -> list[str]:
        """
        Position keys claimed by all optimizers.

        Returns
        -------
        list[str]
            Concatenated optimizer position keys in optimizer order.
        """
        keys: list[str] = []
        for optim in self.optimizers:
            keys += optim.position_keys
        return keys

    def _validate_position_keys(self) -> None:
        """
        Validates that each optimized position key is owned by one optimizer.

        Raises
        ------
        ValueError
            If two or more optimizers claim the same position key.
        """
        counts = {}
        for key in self.position_keys:
            if key not in counts:
                counts[key] = 1
            else:
                counts[key] += 1

        duplicates = {k: v for k, v in counts.items() if v > 1}
        if len(duplicates) >= 1:
            raise ValueError(
                f"Position keys claimed by multiple optimizers: {list(duplicates)}"
            )

    def _validate_optimizer_activation_delays(self) -> None:
        invalid = [
            opt.identifier
            for opt in self.optimizers
            if opt.activate_after_epochs >= self.stopper.epochs
        ]
        if invalid:
            raise ValueError(
                "activate_after_epochs must be less than stopper.epochs for "
                f"optimizers: {invalid}."
            )

    def _validate_optimizer_identifiers(self) -> None:
        """
        Validates that optimizer identifiers are unique.

        Raises
        ------
        ValueError
            If two or more optimizers have the same identifier.
        """
        identifiers = [opt.identifier for opt in self.optimizers]
        duplicates = sorted(
            {
                identifier
                for identifier in identifiers
                if identifiers.count(identifier) > 1
            }
        )
        if duplicates:
            raise ValueError(
                "Optimizer identifiers must be unique, but got duplicates: "
                f"{duplicates}."
            )

    def _validate_progress_settings(self) -> None:
        """
        Validates progress-bar configuration.

        Raises
        ------
        ValueError
            If either progress update interval is not a positive integer.
        """
        _validate_positive_int(self.progress_update_every, "progress_update_every")
        _validate_positive_int(
            self.step_progress_update_every, "step_progress_update_every"
        )

    def _validate_loss_monitor(self) -> None:
        """Validates the configured monitoring source.

        Raises
        ------
        ValueError
            If ``loss_monitor`` is unsupported, validation data is missing, or
            full-data monitoring uses the unimplemented :class:`.LossMixin` stub.
        """
        if not isinstance(self.loss_monitor, EmaTrainLossMonitor) and (
            self.loss_monitor not in ("validation", "train_full_data")
        ):
            raise ValueError(
                "loss_monitor must be EmaTrainLossMonitor(effective_window=...), "
                "'validation', or "
                f"'train_full_data', but got {self.loss_monitor!r}."
            )

        if self.loss_monitor == "validation" and not self.split.has_validation:
            raise ValueError(
                "loss_monitor='validation' requires a split with validation data."
            )

        if self.loss_monitor == "train_full_data" and (
            getattr(self.loss.loss_train, "__func__", None) is LossMixin.loss_train
        ):
            raise ValueError(
                "loss_monitor='train_full_data' requires the custom loss to "
                "implement loss_train(). Implement it or use "
                "EmaTrainLossMonitor(effective_window=...)."
            )

    def _validate_debug_nans(self) -> None:
        """
        Validates the NaN debugging switch.

        Raises
        ------
        ValueError
            If ``debug_nans`` is not a boolean.
        """
        if not isinstance(self.debug_nans, bool):
            raise ValueError("debug_nans must be a boolean.")  # noqa: TRY004

    def _validate_batch_split_compatibility(self) -> None:
        """
        Validates batch, split, and built-in L-BFGS compatibility.

        L-BFGS must be the sole optimizer and requires full-data batches and a
        deterministic objective. Stochastic objective evaluations cannot be
        detected here.

        Raises
        ------
        ValueError
            If a multi-size split is paired with single-size batches, or if batches
            reference missing keys or incompatible array shapes in the training
            split, or if built-in L-BFGS is paired with mini-batches or another
            optimizer.
        """
        if isinstance(self.split, PositionSplitManager) and isinstance(
            self.batches, Batches
        ):
            raise ValueError(  # noqa: TRY004
                "OptimEngine requires a BatchManager when used with a "
                "PositionSplitManager."
            )

        _validate_optimizer_batches(self.optimizers, self.batches)

        missing = sorted(
            key for key in self.batches.position_keys if key not in self.split.train
        )
        if missing:
            raise ValueError(
                "Batch position keys must be present in split.train, but these keys "
                f"are missing: {missing}."
            )

        batches = (
            self.batches.batches
            if isinstance(self.batches, BatchManager)
            else (self.batches,)
        )
        getattr(self.loss, "_validate_data_keys", lambda *_: None)(
            self.split, self.position_keys
        )
        getattr(self.loss, "_validate_batch_keys", lambda *_: None)(
            [batch.position_keys for batch in batches]
        )
        for batch in batches:
            batch._validate_position(self.split.train)

    def _name_optimizers(self) -> Sequence[OptimizerLike]:
        """
        Fills missing optimizer identifiers with stable numeric names.

        Optimizer states are stored by identifier in :class:`.OptimCarry`. This
        method mutates optimizers whose ``identifier`` is empty and leaves existing
        identifiers unchanged.

        Returns
        -------
        collections.abc.Sequence[OptimizerLike]
            The optimizer sequence attached to the engine.
        """
        for i, opt in enumerate(self.optimizers):
            if not opt.identifier:
                opt.identifier = f"{i:03}"
        return self.optimizers

    def fit(
        self,
        *,
        checkpoint: OptimCheckpoint | str | os.PathLike[str] | None = None,
        pause_after: int | None = None,
        checkpoint_every: int = 10,
        allow_version_mismatch: bool = False,
    ) -> OptimResult:
        """
        Runs optimization and returns processed results.

        Parameters
        ----------
        checkpoint
            ``None`` starts fresh in memory. An :class:`.OptimCheckpoint` resumes
            in memory. A path selects a persistent run: load it if present, or
            start fresh if absent, and save subsequent checkpoints there. The
            parent directory must exist. Only load trusted checkpoint files.
        pause_after
            Maximum additional epochs for this call. Pausing preserves optimizer
            state in ``result.checkpoint``. The stopper still owns the total epoch
            budget; change ``engine.stopper.epochs`` to extend it explicitly.
        checkpoint_every
            Save every this many completed epochs when a path is supplied, by
            default 10. Also save at a deliberate pause or normal completion.
            A NaN failure leaves the last saved file intact.
        allow_version_mismatch
            Attempt recovery despite differing Liesel, JAX, jaxlib, Optax, or
            NumPy versions, issuing a warning. Structural checks still apply.

        Returns
        -------
        OptimResult
            Processed history, final and best-monitor positions,
            monitoring source, cumulative active runtime, status, and an
            independent checkpoint for continuation (``None`` on NaN failure).

        Notes
        -----
        Reconstruct the same model, data, and optimizer settings before resuming.
        Parameter and observed-variable names, shapes, and dtypes must agree.
        Compatibility checks cannot detect changed data or learning rates.
        Resuming a checkpoint that permits no further epochs under the current
        stopper settings issues a warning and returns its result. Use a new path
        or call ``fit()`` without a checkpoint to start a fresh run. Extending the
        epoch budget permits continuation only if early stopping does not apply.
        A failed checkpoint write raises and preserves the previous file.
        Interruptions recover from the last successful periodic save.
        """
        self.stopper.__post_init__()
        self.__post_init__()
        _validate_positive_int(checkpoint_every, "checkpoint_every")
        start = time.monotonic()
        checkpoint_path = (
            Path(cast(str | os.PathLike[str], checkpoint))
            if isinstance(checkpoint, (str, os.PathLike))
            else None
        )
        if checkpoint_path is not None:
            try:
                checkpoint_path.lstat()
            except FileNotFoundError:
                checkpoint = None
            else:
                checkpoint = OptimCheckpoint.load(checkpoint_path)
        if checkpoint is not None and not isinstance(checkpoint, OptimCheckpoint):
            raise TypeError("checkpoint must be an OptimCheckpoint, path, or None.")
        if not isinstance(allow_version_mismatch, bool):
            raise TypeError("allow_version_mismatch must be a boolean.")
        if checkpoint is not None:
            versions = _checkpoint_versions()
            differences = [
                f"{name}: saved={checkpoint.versions.get(name)!r}, current={current!r}"
                for name, current in versions.items()
                if checkpoint.versions.get(name) != current
            ]
            if differences:
                message = "Checkpoint version mismatch: " + "; ".join(differences)
                if not allow_version_mismatch:
                    raise ValueError(
                        message
                        + ". Set allow_version_mismatch=True to attempt recovery."
                    )
                warnings.warn(message, UserWarning, stacklevel=2)
        if pause_after is not None:
            _validate_positive_int(pause_after, "pause_after")
        carry = (
            self._init_carry(self.stopper.epochs)
            if checkpoint is None
            else self._restore_carry(checkpoint)
        )
        if carry.loss_state is not None and (
            not self.batches.is_full_data or self.loss_monitor != "train_full_data"
        ):
            raise ValueError(
                "Stateful losses require full-data batches and "
                "loss_monitor='train_full_data'."
            )
        if checkpoint is not None:
            status = self._fit_status(carry)
            if status in ("max_epochs", "early_stopping"):
                location = (
                    f" at {checkpoint_path}" if checkpoint_path is not None else ""
                )
                warnings.warn(
                    f"Checkpoint{location} is already complete under the current "
                    f"stopper settings ({int(carry.epoch)} epochs, {status}); "
                    "returning its result without running additional epochs. "
                    "Use a new path or call fit() without a checkpoint to start "
                    "a fresh run.",
                    UserWarning,
                    stacklevel=2,
                )
        end_epoch = self.stopper.epochs
        if pause_after is not None:
            end_epoch = min(end_epoch, int(carry.epoch) + pause_after)
        logger.info(
            "%s optimization at epoch %s%s",
            "Initializing" if checkpoint is None else "Resuming",
            int(carry.epoch),
            f" ({checkpoint_path})" if checkpoint_path is not None else "",
        )
        previous_duration = checkpoint.duration if checkpoint is not None else 0.0

        def save_checkpoint(carry: OptimCarry) -> None:
            assert checkpoint_path is not None
            jax.block_until_ready(carry)
            snapshot = jax.tree.map(lambda x: x, carry)
            snapshot.history = self._process_history(snapshot.epoch, snapshot.history)
            self._make_checkpoint(
                snapshot, previous_duration + time.monotonic() - start
            ).save(checkpoint_path)

        carry = self._fit(
            carry,
            end_epoch,
            checkpoint_every,
            save_checkpoint if checkpoint_path is not None else None,
        )
        if bool(carry._numerical_failure):
            carry = self._rollback_epoch(carry)
        jax.block_until_ready(carry)
        duration = previous_duration + time.monotonic() - start
        status = self._fit_status(carry)
        nan_debug = self._nan_debug_info(carry)
        history = self._process_history(carry.epoch, carry.history)
        n_epochs = int(carry.epoch)
        position_final = carry.position

        if n_epochs == 0 or not bool(jnp.isfinite(carry.min_monitor_loss)):
            position_min_monitor = None
            min_monitor_epoch = None
        else:
            position_min_monitor = carry.position_min_monitor
            min_monitor_epoch = int(carry.min_monitor_epoch)

        monitor_source: Literal["train_ema", "validation", "train_full_data"]
        if isinstance(self.loss_monitor, EmaTrainLossMonitor):
            monitor_source = "train_ema"
        else:
            monitor_source = self.loss_monitor

        result = OptimResult(
            history=history,
            position_final=position_final,
            position_min_monitor=position_min_monitor,
            n_epochs=n_epochs,
            min_monitor_epoch=min_monitor_epoch,
            monitor_source=monitor_source,
            patience=self.stopper.patience,
            duration=duration,
            nan_debug=nan_debug,
            status=status,
            loss_state_final=(
                carry.loss_state if bool(carry._loss_state_valid) else None
            ),
            loss_state_min_monitor=(
                carry.loss_state_min_monitor if min_monitor_epoch is not None else None
            ),
            failed_loss_state=(
                carry.failed_loss_state if status == "numerical_failure" else None
            ),
            failure_reason=self._failure_reason(carry),
            checkpoint=(
                None
                if status in ("nan", "numerical_failure")
                else self._make_checkpoint(carry, duration)
            ),
        )
        result.duration = previous_duration + time.monotonic() - start
        if result.checkpoint is not None:
            result.checkpoint = replace(result.checkpoint, duration=result.duration)
        if checkpoint_path is not None and result.checkpoint is not None:
            result.checkpoint.save(checkpoint_path)
            result.duration = previous_duration + time.monotonic() - start
            result.checkpoint = replace(result.checkpoint, duration=result.duration)
        return result

    def _can_rebuild_model_state(self) -> bool:
        # Only these concrete implementations leave the evaluation template unchanged.
        return type(self.loss) in (NegLogProbLoss, LaplaceLoss) and all(
            type(opt) in (Optimizer, LBFGS) for opt in self.optimizers
        )

    def _prepare_data_states(self, carry: OptimCarry) -> None:
        # Custom losses/optimizers may evolve model_state. Only reuse templates
        # for the same concrete implementations that permit checkpoint rebuilding.
        if not self._can_rebuild_model_state():
            return
        model = getattr(self.loss, "model", None)
        assert model is not None
        carry._data_states = {}
        if self.batches.is_full_data or self.loss_monitor == "train_full_data":
            carry._data_states["train"] = model.update_state(
                self.split.train, carry.model_state, allow_weak_vars=True
            )
        if self.loss_monitor == "validation":
            carry._data_states["validate"] = model.update_state(
                self.split.validate, carry.model_state, allow_weak_vars=True
            )

    def _data_structure(self) -> tuple:
        return tuple(
            {
                name: (jnp.shape(value), str(jnp.asarray(value).dtype))
                for name, value in part.items()
            }
            for part in (self.split.train, self.split.validate, self.split.test)
        )

    def _loss_configuration(self) -> tuple | None:
        return getattr(self.loss, "_checkpoint_configuration", lambda: None)()

    def _make_checkpoint(self, carry: OptimCarry, duration: float) -> OptimCheckpoint:
        snapshot = jax.tree.map(lambda x: x, carry)
        rebuild_model_state = self._can_rebuild_model_state()
        if rebuild_model_state:
            # Reconstruct this static template from the caller's model. Internal
            # anonymous node names may differ between otherwise identical models.
            snapshot.model_state = {}
            snapshot._data_states = {}
            if snapshot.nan_debug_state is not None:
                snapshot.nan_debug_state.reproduction_model_state = {}
        return OptimCheckpoint(
            snapshot,
            duration,
            _rebuild_model_state=rebuild_model_state,
            _data_structure=self._data_structure(),
            _loss_configuration=self._loss_configuration(),
        )

    def _restore_carry(self, checkpoint: OptimCheckpoint) -> OptimCarry:
        """Copies snapshot containers and restores the working history capacity."""
        if checkpoint._loss_configuration != self._loss_configuration():
            raise ValueError("Checkpoint loss configuration is incompatible.")
        position = self.loss.position(self.position_keys)
        self._validate_checkpoint_tree(checkpoint._carry.position, position, "position")
        if checkpoint._data_structure != self._data_structure():
            raise ValueError(
                "Checkpoint data structure (names, shapes or dtypes) differs."
            )
        states = {opt.identifier: opt.init(position) for opt in self.optimizers}
        self._validate_checkpoint_tree(
            checkpoint._carry.optimizer_states, states, "optimizer state"
        )
        if checkpoint._rebuild_model_state:
            if not self._can_rebuild_model_state():
                raise ValueError(
                    "Checkpoint requires the built-in loss and optimizers."
                )
        else:
            self._validate_checkpoint_tree(
                checkpoint._carry.model_state, self.initial_state, "model state"
            )
        if jax.tree.structure(checkpoint._carry.batches) != jax.tree.structure(
            self.batches
        ):
            raise ValueError("Checkpoint batch configuration is incompatible.")
        if (checkpoint._carry.nan_debug_state is not None) != self.debug_nans:
            raise ValueError("Checkpoint debug_nans setting is incompatible.")
        if (checkpoint.history.position is not None) != self.save_position_history:
            raise ValueError(
                "Checkpoint save_position_history setting is incompatible."
            )
        carry = jax.tree.map(lambda x: x, checkpoint._carry)
        if checkpoint._rebuild_model_state:
            carry.model_state = jax.tree.map(lambda x: x, self.initial_state)
            self._prepare_data_states(carry)
            if carry.nan_debug_state is not None:
                carry.nan_debug_state.reproduction_model_state = jax.tree.map(
                    lambda x: x, self.initial_state
                )
        self._validate_checkpoint_tree(
            carry.loss_state, self.loss.init_state(position, carry), "loss state"
        )
        n = int(carry.epoch)
        capacity = max(n, self.stopper.epochs)
        history = OptimHistory.from_epochs(
            capacity,
            carry.position if carry.history.position is not None else None,
            carry.history.loss_train.dtype,
        )
        carry.history = jax.tree.map(
            lambda empty, saved: empty.at[:n].set(saved[:n]), history, carry.history
        )
        return carry

    @staticmethod
    def _validate_checkpoint_tree(saved, expected, name: str) -> None:
        if jax.tree.structure(saved) != jax.tree.structure(expected):
            raise ValueError(f"Checkpoint {name} structure is incompatible.")
        for a, b in zip(jax.tree.leaves(saved), jax.tree.leaves(expected), strict=True):
            if (
                jnp.shape(a) != jnp.shape(b)
                or jnp.asarray(a).dtype != jnp.asarray(b).dtype
            ):
                raise ValueError(
                    f"Checkpoint {name} shapes or dtypes are incompatible."
                )

    def _fit_status(self, carry: OptimCarry):
        if int(carry._numerical_failure):
            return "numerical_failure"
        if bool(jnp.isnan(carry.loss_train) | jnp.isnan(carry.loss_monitor)) or (
            carry.nan_debug_state is not None and bool(carry.nan_debug_state.has_nan)
        ):
            return "nan"
        if int(carry.epoch) >= self.stopper.epochs:
            return "max_epochs"
        if not bool(self._continue_fit(carry)):
            return "early_stopping"
        return "paused"

    def _failure_reason(self, carry: OptimCarry) -> str | None:
        reason = int(carry._numerical_failure)
        if reason == 0:
            return None
        message = getattr(self.loss, "_failure_message", lambda *_: None)(
            reason, carry.failed_loss_state
        )
        if message is not None:
            return message
        return {
            -1: "Non-finite loss evaluation.",
            -2: "Non-finite outer gradient.",
            -3: "Outer line search failed to find a valid step.",
            -4: "Non-finite outer parameter update.",
        }.get(reason, f"Loss evaluation failed (code {reason}).")

    def _nan_debug_info(self, carry: OptimCarry) -> OptimNaNDebugInfo | None:
        if not self.debug_nans:
            return None

        debug_state = carry.nan_debug_state
        if debug_state is None or not bool(debug_state.has_nan):
            return None

        kind_code = int(debug_state.kind_code)
        kind = _NAN_DEBUG_KIND_NAMES[kind_code]

        optimizer_index_raw = int(debug_state.optimizer_index)
        if optimizer_index_raw < 0:
            optimizer_index = None
            optimizer_identifier = None
            optimizer_position_keys = None
            fixed_position = Position({})
        else:
            optimizer_index = optimizer_index_raw
            optimizer = self.optimizers[optimizer_index]
            optimizer_identifier = optimizer.identifier
            optimizer_position_keys = tuple(optimizer.position_keys)
            fixed_position = optimizer.not_position(debug_state.reproduction_position)

        nan_position = (
            debug_state.nan_position
            if kind in ("position_before", "position_after")
            else None
        )
        loss = debug_state.loss if kind == "loss" else None

        reproduction_carry = OptimCarry(
            key=debug_state.reproduction_key,
            position=debug_state.reproduction_position,
            history=carry.history,
            batches=debug_state.reproduction_batches,
            optimizer_states=debug_state.reproduction_optimizer_states,
            model_state=debug_state.reproduction_model_state,
            loss_state=debug_state.reproduction_loss_state,
            loss_state_min_monitor=carry.loss_state_min_monitor,
            failed_loss_state=(
                debug_state.reproduction_loss_state
                if carry.loss_state is not None
                else None
            ),
            _data_states=carry._data_states,
            batch=debug_state.obs_batch,
            fixed_position=fixed_position,
            position_min_monitor=carry.position_min_monitor,
            min_monitor_loss=carry.min_monitor_loss,
            min_monitor_epoch=carry.min_monitor_epoch,
            loss_train=carry.loss_train,
            loss_monitor=carry.loss_monitor,
            epoch=int(debug_state.epoch),
            i_batch=int(debug_state.batch),
            nan_debug_state=None,
        )

        return OptimNaNDebugInfo(
            kind=kind,
            epoch=int(debug_state.epoch),
            batch=int(debug_state.batch),
            obs_batch=debug_state.obs_batch,
            last_non_nan_position=debug_state.last_non_nan_position,
            nan_position=nan_position,
            loss=loss,
            optimizer_index=optimizer_index,
            optimizer_identifier=optimizer_identifier,
            optimizer_position_keys=optimizer_position_keys,
            reproduction_position=debug_state.reproduction_position,
            reproduction_carry=reproduction_carry,
        )

    def _process_history(self, i: int, history: OptimHistory) -> OptimHistory:
        """
        Marks unused history entries and optionally prunes them.

        Parameters
        ----------
        i
            Number of completed epochs. Entries at indices ``i:`` are unused.
        history
            Raw history allocated for ``stopper.epochs`` epochs.

        Returns
        -------
        OptimHistory
            History with unused entries set to ``nan`` and, if ``prune_history`` is
            ``True``, removed from the arrays.
        """
        # Set unused values in history to nan
        history.loss_train = history.loss_train.at[i:].set(jnp.nan)
        history.loss_monitor = history.loss_monitor.at[i:].set(jnp.nan)
        if self.save_position_history:
            assert history.position is not None
            for name, value in history.position.items():
                history.position[name] = value.at[i:, ...].set(jnp.nan)

        if not self.prune_history:
            return history

        # Remove unused values in history, if applicable
        history.loss_train = history.loss_train[:i]
        history.loss_monitor = history.loss_monitor[:i]
        if self.save_position_history:
            assert history.position is not None
            for name, value in history.position.items():
                history.position[name] = value[:i, ...]

        return history

    def _run_optimizer_step(
        self, opt: OptimizerLike, carry: OptimCarry
    ) -> tuple[OptimCarry, jax.Array]:
        """
        Runs one optimizer update for the current batch.

        Parameters
        ----------
        opt
            Optimizer to apply.
        carry
            Current optimizer carry.

        Returns
        -------
        tuple[OptimCarry, jax.Array]
            Updated carry and the optimizer's pre-update scalar loss.
        """
        # subset of the position handled by this optimizer
        pos = opt.position(carry.position)

        # parameters handled by other optimizers
        carry.fixed_position = opt.not_position(carry.position)

        key, subkey = jax.random.split(carry.key)
        carry.key = subkey
        carry, loss = opt.step(pos, self.loss, carry)
        carry.key = key
        carry.fixed_position = Position({})  # reset fixed position

        return carry, loss

    def _observed_batch(
        self,
        batches: BatchConfig,
        batch_index: int | jax.Array = 0,
        *,
        prepared: bool = False,
    ) -> Position:
        """Keep omitted keys on training rows, either via a template or an overlay."""
        has_holdout = self.split.has_validation or self.split.has_test
        children = batches.batches if isinstance(batches, BatchManager) else [batches]
        if prepared and batches.is_full_data and not any(b.shuffle for b in children):
            return Position({})
        if not batches.is_full_data or has_holdout:
            batch = batches.get_batched_position(self.split.train, batch_index)
            if has_holdout and not prepared:
                return Position(self.split.train | batch)
            return batch

        return Position({})

    def _init_nan_debug_state(self, carry: OptimCarry) -> OptimNaNDebugState:
        obs_batch = self._observed_batch(
            carry.batches, prepared="train" in carry._data_states
        )
        loss_dtype = jnp.asarray(carry.loss_train).dtype
        return OptimNaNDebugState.new(
            key=carry.key,
            position=carry.position,
            obs_batch=obs_batch,
            optimizer_states=dict(carry.optimizer_states),
            batches=carry.batches,
            model_state=carry.model_state,
            loss_dtype=loss_dtype,
            loss_state=carry.loss_state,
        )

    def _debug_state(self, carry: OptimCarry) -> OptimNaNDebugState:
        debug_state = carry.nan_debug_state
        assert debug_state is not None
        return debug_state

    def _debug_update_last_non_nan_position(
        self, carry: OptimCarry, position: Position
    ) -> OptimCarry:
        debug_state = self._debug_state(carry)
        position_has_nan = _tree_has_nan(position)
        should_update = (~debug_state.has_nan) & (~position_has_nan)
        debug_state.last_non_nan_position = _position_where(
            should_update,
            position,
            debug_state.last_non_nan_position,
        )
        carry.nan_debug_state = debug_state
        return carry

    def _debug_capture_nan(
        self,
        carry: OptimCarry,
        *,
        kind_code: int,
        obs_batch: Position,
        nan_position: Position,
        loss: jax.Array,
        reproduction_position: Position,
        reproduction_key: jax.Array,
        reproduction_optimizer_states: dict[str, optax.OptState],
        reproduction_batches: BatchConfig,
        reproduction_model_state: ModelState,
        optimizer_index: int = -1,
    ) -> OptimCarry:
        debug_state = self._debug_state(carry)
        should_capture = ~debug_state.has_nan

        debug_state.has_nan = debug_state.has_nan | should_capture
        debug_state.kind_code = jnp.where(
            should_capture,
            jnp.asarray(kind_code, dtype=debug_state.kind_code.dtype),
            debug_state.kind_code,
        )
        debug_state.epoch = jnp.where(
            should_capture,
            jnp.asarray(carry.epoch, dtype=debug_state.epoch.dtype),
            debug_state.epoch,
        )
        debug_state.batch = jnp.where(
            should_capture,
            jnp.asarray(carry.i_batch, dtype=debug_state.batch.dtype),
            debug_state.batch,
        )
        debug_state.optimizer_index = jnp.where(
            should_capture,
            jnp.asarray(optimizer_index, dtype=debug_state.optimizer_index.dtype),
            debug_state.optimizer_index,
        )
        debug_state.obs_batch = Position(
            _tree_where(should_capture, obs_batch, debug_state.obs_batch)
        )
        debug_state.nan_position = _position_where(
            should_capture, nan_position, debug_state.nan_position
        )
        debug_state.loss = jnp.where(
            should_capture,
            jnp.asarray(loss, dtype=debug_state.loss.dtype),
            debug_state.loss,
        )
        debug_state.reproduction_position = _position_where(
            should_capture,
            reproduction_position,
            debug_state.reproduction_position,
        )
        debug_state.reproduction_key = _tree_where(
            should_capture, reproduction_key, debug_state.reproduction_key
        )
        debug_state.reproduction_optimizer_states = _tree_where(
            should_capture,
            reproduction_optimizer_states,
            debug_state.reproduction_optimizer_states,
        )
        debug_state.reproduction_batches = _tree_where(
            should_capture,
            reproduction_batches,
            debug_state.reproduction_batches,
        )
        debug_state.reproduction_model_state = _tree_where(
            should_capture,
            reproduction_model_state,
            debug_state.reproduction_model_state,
        )
        debug_state.reproduction_loss_state = _tree_where(
            should_capture, carry.loss_state, debug_state.reproduction_loss_state
        )

        carry.nan_debug_state = debug_state
        return carry

    def _run_batch_unchecked(self, j: int | jax.Array, carry: OptimCarry) -> OptimCarry:
        """
        Runs all optimizer updates and records training loss for one batch.

        Parameters
        ----------
        j
            Batch index within the current epoch.
        carry
            Current optimizer carry.

        Returns
        -------
        OptimCarry
            Updated carry with accumulated epoch training loss.
        """
        obs_batch = self._observed_batch(
            carry.batches, j, prepared="train" in carry._data_states
        )
        carry.batch = obs_batch
        carry.i_batch = j

        loss = jnp.zeros_like(carry.loss_train)
        has_active_optimizer = jnp.asarray(False)
        optimizer_loss_has_nan = jnp.asarray(False)
        for opt in self.optimizers:
            is_active = (carry.epoch >= opt.activate_after_epochs) & (
                carry._numerical_failure == 0
            )
            carry, optimizer_loss = jax.lax.cond(
                is_active,
                lambda carry, opt=opt: self._run_optimizer_step(opt, carry),
                lambda carry, loss=loss: (carry, loss),
                carry,
            )
            loss = jnp.where(is_active & ~has_active_optimizer, optimizer_loss, loss)
            optimizer_loss_has_nan = optimizer_loss_has_nan | (
                is_active & _tree_has_nan(optimizer_loss)
            )
            has_active_optimizer = has_active_optimizer | is_active

        loss = jax.lax.cond(
            has_active_optimizer,
            lambda carry: loss,
            lambda carry: self.loss.loss_train_batched(carry.position, carry)[0],
            carry,
        )
        batch_has_nan = (
            optimizer_loss_has_nan | _tree_has_nan(loss) | _tree_has_nan(carry.position)
        )
        loss = jnp.where(batch_has_nan, jnp.nan, loss)
        carry = self._accumulate_loss(loss, carry)

        carry.batch = Position({})

        return carry

    def _run_optimizer_step_debug(
        self,
        opt: OptimizerLike,
        opt_index: int,
        obs_batch: Position,
        carry: OptimCarry,
    ) -> tuple[OptimCarry, jax.Array]:
        pre_position = Position(dict(carry.position))
        pre_optimizer_states = dict(carry.optimizer_states)
        pre_model_state = carry.model_state
        pre_position_has_nan = _tree_has_nan(pre_position)
        empty_loss = jnp.zeros_like(carry.loss_train)

        def capture_pre_position(
            carry: OptimCarry,
        ) -> tuple[OptimCarry, jax.Array]:
            return (
                self._debug_capture_nan(
                    carry,
                    kind_code=_NAN_DEBUG_KIND_POSITION_BEFORE,
                    obs_batch=obs_batch,
                    nan_position=pre_position,
                    loss=self._debug_state(carry).loss,
                    reproduction_position=pre_position,
                    reproduction_key=carry.key,
                    reproduction_optimizer_states=pre_optimizer_states,
                    reproduction_batches=carry.batches,
                    reproduction_model_state=pre_model_state,
                    optimizer_index=opt_index,
                ),
                empty_loss,
            )

        def run_step(carry: OptimCarry) -> tuple[OptimCarry, jax.Array]:
            carry = self._debug_update_last_non_nan_position(carry, pre_position)

            pos = opt.position(pre_position)
            carry.fixed_position = opt.not_position(pre_position)

            key, subkey = jax.random.split(carry.key)
            carry.key = subkey
            carry, loss = opt.step(pos, self.loss, carry)
            carry.key = key
            carry.fixed_position = Position({})

            loss_has_nan = _tree_has_nan(loss)
            position_has_nan = _tree_has_nan(carry.position)

            def capture_loss(carry: OptimCarry) -> OptimCarry:
                return self._debug_capture_nan(
                    carry,
                    kind_code=_NAN_DEBUG_KIND_LOSS,
                    obs_batch=obs_batch,
                    nan_position=pre_position,
                    loss=loss,
                    reproduction_position=pre_position,
                    reproduction_key=subkey,
                    reproduction_optimizer_states=pre_optimizer_states,
                    reproduction_batches=carry.batches,
                    reproduction_model_state=pre_model_state,
                    optimizer_index=opt_index,
                )

            def capture_post_position(carry: OptimCarry) -> OptimCarry:
                return self._debug_capture_nan(
                    carry,
                    kind_code=_NAN_DEBUG_KIND_POSITION_AFTER,
                    obs_batch=obs_batch,
                    nan_position=carry.position,
                    loss=self._debug_state(carry).loss,
                    reproduction_position=pre_position,
                    reproduction_key=subkey,
                    reproduction_optimizer_states=pre_optimizer_states,
                    reproduction_batches=carry.batches,
                    reproduction_model_state=pre_model_state,
                    optimizer_index=opt_index,
                )

            def update_last_non_nan(carry: OptimCarry) -> OptimCarry:
                return self._debug_update_last_non_nan_position(carry, carry.position)

            carry = jax.lax.cond(
                loss_has_nan,
                capture_loss,
                lambda carry: jax.lax.cond(
                    position_has_nan,
                    capture_post_position,
                    update_last_non_nan,
                    carry,
                ),
                carry,
            )
            return carry, loss

        return jax.lax.cond(
            pre_position_has_nan,
            capture_pre_position,
            run_step,
            carry,
        )

    def _run_batch_debug_body(
        self, j: int | jax.Array, carry: OptimCarry
    ) -> OptimCarry:
        obs_batch = self._observed_batch(
            carry.batches, j, prepared="train" in carry._data_states
        )
        carry.batch = obs_batch
        carry.i_batch = j

        position_has_nan = _tree_has_nan(carry.position)

        def capture_initial_position(carry: OptimCarry) -> OptimCarry:
            return self._debug_capture_nan(
                carry,
                kind_code=_NAN_DEBUG_KIND_POSITION_BEFORE,
                obs_batch=obs_batch,
                nan_position=carry.position,
                loss=self._debug_state(carry).loss,
                reproduction_position=carry.position,
                reproduction_key=carry.key,
                reproduction_optimizer_states=dict(carry.optimizer_states),
                reproduction_batches=carry.batches,
                reproduction_model_state=carry.model_state,
            )

        def keep_initial_position(carry: OptimCarry) -> OptimCarry:
            return self._debug_update_last_non_nan_position(carry, carry.position)

        carry = jax.lax.cond(
            position_has_nan,
            capture_initial_position,
            keep_initial_position,
            carry,
        )

        loss = jnp.zeros_like(carry.loss_train)
        has_active_optimizer = jnp.asarray(False)
        for opt_index, opt in enumerate(self.optimizers):

            def run_optimizer_step(
                carry: OptimCarry, opt=opt, opt_index=opt_index
            ) -> tuple[OptimCarry, jax.Array]:
                return self._run_optimizer_step_debug(opt, opt_index, obs_batch, carry)

            is_active = (carry.epoch >= opt.activate_after_epochs) & (
                carry._numerical_failure == 0
            )
            carry, optimizer_loss = jax.lax.cond(
                jnp.logical_or(
                    self._debug_state(carry).has_nan,
                    ~is_active,
                ),
                lambda carry, loss=loss: (carry, loss),
                run_optimizer_step,
                carry,
            )
            loss = jnp.where(is_active & ~has_active_optimizer, optimizer_loss, loss)
            has_active_optimizer = has_active_optimizer | is_active

        def skip_loss(carry: OptimCarry) -> OptimCarry:
            return carry

        def record_loss(carry: OptimCarry) -> OptimCarry:
            def accumulate_optimizer_loss(carry: OptimCarry) -> OptimCarry:
                return self._accumulate_loss(loss, carry)

            def evaluate_loss(carry: OptimCarry) -> OptimCarry:
                evaluated_loss, _ = self.loss.loss_train_batched(carry.position, carry)
                loss_has_nan = _tree_has_nan(evaluated_loss)

                def capture_loss(carry: OptimCarry) -> OptimCarry:
                    return self._debug_capture_nan(
                        carry,
                        kind_code=_NAN_DEBUG_KIND_LOSS,
                        obs_batch=obs_batch,
                        nan_position=carry.position,
                        loss=evaluated_loss,
                        reproduction_position=carry.position,
                        reproduction_key=carry.key,
                        reproduction_optimizer_states=dict(carry.optimizer_states),
                        reproduction_batches=carry.batches,
                        reproduction_model_state=carry.model_state,
                    )

                def accumulate_loss(carry: OptimCarry) -> OptimCarry:
                    return self._accumulate_loss(evaluated_loss, carry)

                return jax.lax.cond(loss_has_nan, capture_loss, accumulate_loss, carry)

            return jax.lax.cond(
                has_active_optimizer,
                accumulate_optimizer_loss,
                evaluate_loss,
                carry,
            )

        return jax.lax.cond(
            self._debug_state(carry).has_nan,
            skip_loss,
            record_loss,
            carry,
        )

    def _run_batch_debug(self, j: int | jax.Array, carry: OptimCarry) -> OptimCarry:
        return jax.lax.cond(
            self._debug_state(carry).has_nan,
            lambda carry: carry,
            lambda carry: self._run_batch_debug_body(j, carry),
            carry,
        )

    def _run_batch(self, j: int | jax.Array, carry: OptimCarry) -> OptimCarry:
        return jax.lax.cond(
            carry._numerical_failure != 0,
            lambda c: c,
            lambda c: self._run_batch_body(j, c),
            carry,
        )

    def _run_batch_body(self, j: int | jax.Array, carry: OptimCarry) -> OptimCarry:
        if self.debug_nans:
            return self._run_batch_debug(j, carry)

        return self._run_batch_unchecked(j, carry)

    def _accumulate_loss(self, loss: jax.Array, carry: OptimCarry) -> OptimCarry:
        loss_dtype = jnp.asarray(loss).dtype
        n_batches = jnp.asarray(carry.batches.n_full_batches, dtype=loss_dtype)
        carry.loss_train += loss / n_batches

        if isinstance(self.loss_monitor, EmaTrainLossMonitor):
            one = jnp.asarray(1.0, dtype=loss_dtype)
            two = jnp.asarray(2.0, dtype=loss_dtype)
            effective_window = jnp.maximum(
                one,
                jnp.asarray(self.loss_monitor.effective_window, dtype=loss_dtype)
                * n_batches,
            )
            alpha = two / (effective_window + one)
            # Cast before multiplying to avoid overflowing an integer step count.
            t = (
                jnp.asarray(carry.epoch, dtype=loss_dtype) * n_batches
                + jnp.asarray(carry.i_batch, dtype=loss_dtype)
                + one
            )
            # Compute bias correction directly; accumulating this weight in
            # float32 can stall and make a flat loss appear to worsen.
            weight = -jnp.expm1(t * jnp.log1p(-alpha))
            # Kahan summation retains updates smaller than one float32 ULP.
            update = (alpha / weight) * (loss - carry._ema_mean)
            update -= carry._ema_compensation
            mean = carry._ema_mean + update
            compensation = (mean - carry._ema_mean) - update
            # These cases equal the observation exactly, without subtraction.
            use_loss = (t == one) | (alpha == one)
            carry._ema_compensation = jnp.where(use_loss, 0, compensation)
            carry._ema_mean = jnp.where(use_loss, loss, mean)

        return carry

    def _start_epoch(self, carry: OptimCarry) -> OptimCarry:
        """Starts a batch epoch and resets its accumulated losses."""
        if carry.loss_state is not None:
            carry._epoch_start = jax.tree.map(
                lambda x: x,
                (
                    carry.position,
                    carry.optimizer_states,
                    carry.loss_train,
                    carry.loss_monitor,
                    carry._loss_state_valid,
                ),
            )
        key, subkey = jax.random.split(carry.key)
        carry.key = key
        carry.batches = carry.batches.start_epoch(subkey)
        carry.loss_train = jnp.zeros_like(carry.loss_train)
        carry._loss_state_valid = jnp.asarray(False)
        return carry

    def _run_batch_range(
        self,
        lower: int | jax.Array,
        upper: int | jax.Array,
        carry: OptimCarry,
    ) -> OptimCarry:
        """Runs a contiguous range of batches within the current epoch."""
        return jax.lax.fori_loop(
            lower=lower,
            upper=upper,
            body_fun=self._run_batch,
            init_val=carry,
        )

    def _run_epoch(self, carry: OptimCarry) -> OptimCarry:
        """
        Runs one full epoch over the configured batches.

        The method starts a new batch epoch, runs the batch loop, records train and
        monitoring losses, updates position history, updates the global best
        position, and increments ``carry.epoch``.

        Parameters
        ----------
        carry
            Current optimizer carry.

        Returns
        -------
        OptimCarry
            Carry advanced by one completed epoch.
        """
        carry = self._start_epoch(carry)
        carry = self._run_batch_range(
            lower=0,
            upper=carry.batches.n_full_batches,
            carry=carry,
        )

        if self.debug_nans:
            return jax.lax.cond(
                self._debug_state(carry).has_nan & (carry._numerical_failure == 0),
                lambda carry: carry,
                self._finish_epoch,
                carry,
            )

        return self._finish_epoch(carry)

    def _finish_epoch(self, carry: OptimCarry) -> OptimCarry:
        if carry.loss_state is not None:
            return jax.lax.cond(
                carry._numerical_failure != 0,
                self._rollback_epoch,
                self._finish_epoch_body,
                carry,
            )
        return self._finish_epoch_body(carry)

    @staticmethod
    def _rollback_epoch(carry: OptimCarry) -> OptimCarry:
        (
            carry.position,
            carry.optimizer_states,
            carry.loss_train,
            carry.loss_monitor,
            carry._loss_state_valid,
        ) = carry._epoch_start
        return carry

    def _finish_epoch_body(self, carry: OptimCarry) -> OptimCarry:
        """
        Records losses and histories after a full epoch completed without debug stop.
        """
        i = carry.epoch
        loss_i = carry.loss_train
        carry.history.loss_train = carry.history.loss_train.at[i].set(loss_i)

        if isinstance(self.loss_monitor, EmaTrainLossMonitor):
            loss_monitor_i = carry._ema_mean
            carry.loss_monitor = loss_monitor_i
            carry.history.loss_monitor = carry.history.loss_monitor.at[i].set(
                loss_monitor_i
            )
        elif self.loss_monitor == "validation":
            key, subkey = jax.random.split(carry.key)
            carry.key = subkey

            loss_monitor_i, _ = self.loss.loss_monitor(carry.position, carry)
            carry.key = key

            carry.loss_monitor = loss_monitor_i
            carry.history.loss_monitor = carry.history.loss_monitor.at[i].set(
                loss_monitor_i
            )
        else:
            loss_monitor_i, proposed_state = self.loss.loss_train(carry.position, carry)
            carry = _check_evaluation(self.loss, carry, loss_monitor_i, proposed_state)
            carry._loss_state_valid = jnp.isfinite(loss_monitor_i)
            carry._loss_state_valid &= carry._numerical_failure == 0
            carry.loss_state = jax.lax.cond(
                carry._loss_state_valid,
                lambda: proposed_state,
                lambda: carry.loss_state,
            )
            carry.loss_monitor = loss_monitor_i
            carry.history.loss_monitor = carry.history.loss_monitor.at[i].set(
                loss_monitor_i
            )

        if self.save_position_history:
            assert carry.history.position is not None
            carry.history.position = carry.history.update_position_history(
                carry.epoch, carry.history.position, carry.position
            )

        def update_carry(carry: OptimCarry):
            carry.min_monitor_loss = carry.loss_monitor
            carry.position_min_monitor = carry.position
            carry.loss_state_min_monitor = carry.loss_state
            carry.min_monitor_epoch = carry.epoch
            return carry

        carry = jax.lax.cond(
            jnp.isfinite(carry.loss_monitor)
            & (carry._numerical_failure == 0)
            & (carry.loss_monitor < carry.min_monitor_loss),
            update_carry,
            lambda carry: carry,
            carry,
        )

        carry.epoch += 1

        if carry.loss_state is not None:

            def rollback(carry):
                carry.epoch -= 1
                return self._rollback_epoch(carry)

            carry = jax.lax.cond(
                carry._numerical_failure != 0, rollback, lambda c: c, carry
            )

        return carry

    def _continue_fit(self, carry: OptimCarry) -> jax.Array:
        """Returns whether another epoch should be run."""
        loss_train_is_nan = jnp.isnan(carry.loss_train)
        loss_monitor_is_nan = jnp.isnan(carry.loss_monitor)
        no_nan_loss = ~jnp.logical_or(loss_train_is_nan, loss_monitor_is_nan)
        continue_ = self.stopper.continue_(carry.epoch, carry.history.loss_monitor)
        should_continue = jnp.logical_and(no_nan_loss, continue_)
        should_continue &= carry._numerical_failure == 0

        if self.debug_nans:
            should_continue = jnp.logical_and(
                should_continue, ~self._debug_state(carry).has_nan
            )

        return should_continue

    @staticmethod
    def _completed_epoch_losses(carry: OptimCarry) -> tuple[jax.Array, jax.Array]:
        """Returns losses from the latest completed epoch, if one exists."""
        index = jnp.maximum(carry.epoch - 1, 0)
        has_completed_epoch = carry.epoch > 0
        loss_train = jnp.where(
            has_completed_epoch,
            carry.history.loss_train[index],
            carry.loss_train,
        )
        loss_monitor = jnp.where(
            has_completed_epoch,
            carry.history.loss_monitor[index],
            carry.loss_monitor,
        )
        return loss_train, loss_monitor

    def _init_carry(self, epochs: int) -> OptimCarry:
        """
        Creates the initial :class:`.OptimCarry` for a fit.

        Parameters
        ----------
        epochs
            Maximum number of epochs used to allocate history.

        Returns
        -------
        OptimCarry
            Initialized carry with model position, optimizer states, and history.
        """
        key = self.seed

        initial_position = self.loss.position(self.position_keys)

        carry = OptimCarry.new(
            batches=self.batches,
            key=key,
            epochs=epochs,
            position=initial_position,
            optimizers=self.optimizers,
            model_state=self.initial_state,
            save_position_history=self.save_position_history,
        )
        self._prepare_data_states(carry)
        carry.loss_state = self.loss.init_state(initial_position, carry)
        carry.loss_state_min_monitor = jax.tree.map(
            lambda value: value, carry.loss_state
        )
        if carry.loss_state is not None:
            carry.failed_loss_state = jax.tree.map(lambda x: x, carry.loss_state)
            carry._epoch_start = jax.tree.map(
                lambda x: x,
                (
                    carry.position,
                    carry.optimizer_states,
                    carry.loss_train,
                    carry.loss_monitor,
                    carry._loss_state_valid,
                ),
            )
        if self.debug_nans:
            carry.batch = self._observed_batch(
                carry.batches, prepared="train" in carry._data_states
            )
            carry.nan_debug_state = self._init_nan_debug_state(carry)

        return carry

    def _fit_monolithic(self, carry: OptimCarry, end_epoch: int) -> OptimCarry:
        """Runs the full fit as one JAX loop without host synchronization."""
        return jax.lax.while_loop(
            cond_fun=lambda c: self._continue_fit(c) & (c.epoch < end_epoch),
            body_fun=self._run_epoch,
            init_val=carry,
        )

    @staticmethod
    def _progress_description(loss_train, loss_monitor) -> str:
        return f"Train={float(loss_train):.3f}, Monitor={float(loss_monitor):.3f}"

    @staticmethod
    def _shared_progress_description(
        epoch: int,
        max_epochs: int,
        batch: int,
        n_batches: int,
        loss_train,
        loss_monitor,
    ) -> str:
        return (
            f"Train={float(loss_train):.3f}, Monitor={float(loss_monitor):.3f} "
            f"[E {epoch:>{len(str(max_epochs))}}/{max_epochs}, "
            f"B {batch:>{len(str(n_batches))}}/{n_batches}]"
        )

    def _update_outer_progress(
        self,
        progress_bar,
        rendered_epochs: int,
        completed_epochs: int,
        loss_train,
        loss_monitor,
    ) -> int:
        """Updates the outer bar once and returns its new rendered position."""
        update = completed_epochs - rendered_epochs
        if progress_bar is not None and update > 0:
            progress_bar.set_description(
                self._progress_description(loss_train, loss_monitor), refresh=False
            )
            progress_bar.update(update)
        return max(rendered_epochs, completed_epochs)

    @staticmethod
    def _close_progress_bar(progress_bar) -> None:
        """Closes a display without masking an optimization exception."""
        if progress_bar is None:
            return
        try:
            progress_bar.close()
        except Exception:  # noqa: BLE001, S110
            # Progress display cleanup must not replace an optimization error.
            pass

    def _fit_epoch_chunks(
        self,
        carry: OptimCarry,
        progress_bar,
        end_epoch: int,
        checkpoint_every: int,
        save_checkpoint: Callable[[OptimCarry], None] | None,
    ) -> OptimCarry:
        """Runs dynamic epoch chunks and updates progress on the host."""
        update_every = (
            self.progress_update_every if progress_bar is not None else end_epoch
        )
        max_epochs = end_epoch

        @jax.jit
        def run_chunk(carry: OptimCarry):
            target_epoch = jnp.minimum(
                (carry.epoch // update_every + 1) * update_every, max_epochs
            )
            if save_checkpoint is not None:
                target_epoch = jnp.minimum(
                    target_epoch,
                    (carry.epoch // checkpoint_every + 1) * checkpoint_every,
                )

            def continue_chunk(carry: OptimCarry) -> jax.Array:
                return jnp.logical_and(
                    self._continue_fit(carry), carry.epoch < target_epoch
                )

            carry = jax.lax.while_loop(continue_chunk, self._run_epoch, carry)
            loss_train, loss_monitor = self._completed_epoch_losses(carry)
            status = (
                carry.epoch,
                loss_train,
                loss_monitor,
                self._continue_fit(carry) & (carry.epoch < end_epoch),
            )
            return carry, status

        rendered_epochs = 0
        should_continue = True

        while should_continue:
            carry, status = run_chunk(carry)
            completed, loss_train, loss_monitor, continue_value = jax.device_get(status)
            completed_epochs = int(completed)
            should_continue = bool(continue_value)
            if not should_continue or completed_epochs % update_every == 0:
                rendered_epochs = self._update_outer_progress(
                    progress_bar,
                    rendered_epochs,
                    completed_epochs,
                    loss_train,
                    loss_monitor,
                )
            if (
                should_continue
                and save_checkpoint is not None
                and completed_epochs % checkpoint_every == 0
            ):
                save_checkpoint(carry)

            # A zero-length chunk can only occur when the initial carry should stop.
            if completed_epochs == 0:
                break

        return carry

    def _fit_nested_progress(
        self,
        carry: OptimCarry,
        outer_progress_bar,
        use_nested_bars: bool,
        end_epoch: int,
        checkpoint_every: int,
        save_checkpoint: Callable[[OptimCarry], None] | None,
    ) -> OptimCarry:
        """Runs batch chunks and updates step progress on the host."""
        n_batches = self.batches.n_full_batches
        step_update_every = self.step_progress_update_every
        max_epochs = self.stopper.epochs

        @jax.jit
        def run_batch_chunk(
            carry: OptimCarry, lower: int | jax.Array, upper: int | jax.Array
        ):
            carry = jax.lax.cond(
                lower == 0,
                self._start_epoch,
                lambda carry: carry,
                carry,
            )
            carry = self._run_batch_range(lower, upper, carry)

            if self.debug_nans:
                debug_has_nan = self._debug_state(carry).has_nan
            else:
                debug_has_nan = jnp.asarray(False)

            batch_failed = debug_has_nan | (carry._numerical_failure != 0)
            completed_batches = jnp.where(batch_failed, carry.i_batch + 1, upper)
            loss_train, loss_monitor = self._completed_epoch_losses(carry)
            status = (
                carry.epoch,
                completed_batches,
                loss_train,
                loss_monitor,
                batch_failed,
            )
            return carry, status

        @jax.jit
        def finish_epoch(carry: OptimCarry):
            carry = self._finish_epoch(carry)
            loss_train, loss_monitor = self._completed_epoch_losses(carry)
            status = (
                carry.epoch,
                loss_train,
                loss_monitor,
                self._continue_fit(carry) & (carry.epoch < end_epoch),
            )
            return carry, status

        inner_progress_bar = None
        rendered_epochs = 0
        completed_epochs = int(carry.epoch)
        should_continue = (
            bool(self._continue_fit(carry)) and completed_epochs < end_epoch
        )

        try:
            while should_continue:
                current_epoch = completed_epochs + 1
                if (
                    use_nested_bars
                    and inner_progress_bar is None
                    and outer_progress_bar is not None
                ):
                    inner_progress_bar = tqdm(
                        total=n_batches,
                        desc=f"Epoch {current_epoch}/{max_epochs}",
                        position=1,
                        leave=False,
                    )
                elif use_nested_bars and inner_progress_bar is not None:
                    inner_progress_bar.reset(total=n_batches)
                    inner_progress_bar.set_description(
                        f"Epoch {current_epoch}/{max_epochs}", refresh=False
                    )

                batch_progress_bar = (
                    inner_progress_bar if use_nested_bars else outer_progress_bar
                )
                rendered_batches = 0
                finished_epoch = False
                loss_train = carry.loss_train
                loss_monitor = carry.loss_monitor

                for lower in range(0, n_batches, step_update_every):
                    upper = min(lower + step_update_every, n_batches)
                    carry, status = run_batch_chunk(carry, lower, upper)
                    (
                        completed,
                        completed_batches_value,
                        loss_train,
                        loss_monitor,
                        batch_failed_value,
                    ) = jax.device_get(status)

                    batch_failed = bool(batch_failed_value)
                    completed_batches = int(completed_batches_value)
                    completed_epochs = int(completed)
                    finished_epoch = upper == n_batches and not batch_failed
                    if batch_progress_bar is not None:
                        if not use_nested_bars:
                            batch_progress_bar.set_description_str(
                                self._shared_progress_description(
                                    current_epoch,
                                    max_epochs,
                                    completed_batches,
                                    n_batches,
                                    loss_train,
                                    loss_monitor,
                                ),
                                refresh=False,
                            )
                        update = completed_batches - rendered_batches
                        if update > 0:
                            batch_progress_bar.update(update)
                        if finished_epoch:
                            batch_progress_bar.refresh()
                    rendered_batches = max(rendered_batches, completed_batches)

                    if batch_failed:
                        should_continue = False
                        break

                    if finished_epoch:
                        carry, status = finish_epoch(carry)
                        (
                            completed,
                            loss_train,
                            loss_monitor,
                            continue_value,
                        ) = jax.device_get(status)
                        completed_epochs = int(completed)
                        should_continue = bool(continue_value)
                        if (
                            should_continue
                            and save_checkpoint is not None
                            and completed_epochs % checkpoint_every == 0
                        ):
                            save_checkpoint(carry)
                        if batch_progress_bar is not None and not use_nested_bars:
                            batch_progress_bar.set_description_str(
                                self._shared_progress_description(
                                    current_epoch,
                                    max_epochs,
                                    completed_batches,
                                    n_batches,
                                    loss_train,
                                    loss_monitor,
                                ),
                                refresh=False,
                            )
                        break

                if finished_epoch and use_nested_bars:
                    rendered_epochs = self._update_outer_progress(
                        outer_progress_bar,
                        rendered_epochs,
                        completed_epochs,
                        loss_train,
                        loss_monitor,
                    )

                if not finished_epoch:
                    break
        finally:
            self._close_progress_bar(inner_progress_bar)

        return carry

    def _fit(
        self,
        carry: OptimCarry,
        end_epoch: int,
        checkpoint_every: int,
        save_checkpoint: Callable[[OptimCarry], None] | None,
    ) -> OptimCarry:
        """
        Runs optimization with host-controlled progress updates.

        The numerical carry remains on the device. Progress-enabled modes return
        only small status tuples to Python at configured display boundaries, so
        notebook output is never written from a JAX callback thread.
        """
        self._validate_progress_settings()
        if not self.show_progress:
            if save_checkpoint is None:
                return self._fit_monolithic(carry, end_epoch)
            return self._fit_epoch_chunks(
                carry, None, end_epoch, checkpoint_every, save_checkpoint
            )

        use_nested_progress = (
            self.show_step_progress
            and self.step_progress_update_every < self.batches.n_full_batches
        )
        use_nested_bars = sys.stderr.isatty()
        render_progress = jax.process_index() == 0
        outer_progress_bar = None
        if render_progress:
            if use_nested_progress and not use_nested_bars:
                outer_progress_bar = tqdm(
                    total=self.stopper.epochs * self.batches.n_full_batches,
                    desc=self._shared_progress_description(
                        1,
                        self.stopper.epochs,
                        0,
                        self.batches.n_full_batches,
                        carry.loss_train,
                        carry.loss_monitor,
                    ),
                    leave=True,
                    ncols=88,
                    bar_format="{l_bar}{bar}| [{elapsed}, {rate_fmt}]",
                )
            else:
                outer_progress_bar = tqdm(
                    total=self.stopper.epochs,
                    desc="Initializing",
                    position=0,
                    leave=True,
                )

        rendered_epochs = 0
        try:
            if use_nested_progress:
                carry = self._fit_nested_progress(
                    carry,
                    outer_progress_bar,
                    use_nested_bars,
                    end_epoch,
                    checkpoint_every,
                    save_checkpoint,
                )
            elif (
                save_checkpoint is not None
                or self.progress_update_every < self.stopper.epochs
            ):
                carry = self._fit_epoch_chunks(
                    carry,
                    outer_progress_bar,
                    end_epoch,
                    checkpoint_every,
                    save_checkpoint,
                )
            else:
                carry = self._fit_monolithic(carry, end_epoch)

            final_loss_train, final_loss_monitor = self._completed_epoch_losses(carry)
            completed, loss_train, loss_monitor = jax.device_get(
                (carry.epoch, final_loss_train, final_loss_monitor)
            )
            completed_epochs = int(completed)
            if not use_nested_progress or use_nested_bars:
                if outer_progress_bar is not None:
                    rendered_epochs = int(outer_progress_bar.n)
                self._update_outer_progress(
                    outer_progress_bar,
                    rendered_epochs,
                    completed_epochs,
                    loss_train,
                    loss_monitor,
                )
        finally:
            self._close_progress_bar(outer_progress_bar)

        return carry

    def __repr__(self) -> str:
        """Returns a compact representation showing the configured loss."""
        name = type(self).__name__
        return f"{name}(loss={self.loss})"
