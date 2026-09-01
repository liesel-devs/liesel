"""Low-level optimization engine for experimental optimizers.

The :class:`OptimEngine` class coordinates losses, optimizers, mini-batches,
train/validation/test splits, early stopping, and optimizer history recording. Most
users will usually construct it through :class:`.LieselOptim`, but direct construction
is useful for custom losses or optimizer schedules.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from ._engine_utils import (
    BatchConfig,
    SplitConfig,
    _progress_n_updates,
    _progress_print_rate,
    _validate_positive_int,
)
from .batch import Batches
from .loss import Loss
from .optimizer import LBFGS, OptimizerLike
from .split import PositionSplitManager
from .state import (
    _NAN_DEBUG_KIND_LOSS,
    _NAN_DEBUG_KIND_NAMES,
    _NAN_DEBUG_KIND_POSITION_AFTER,
    _NAN_DEBUG_KIND_POSITION_BEFORE,
    OptimCarry,
    OptimHistory,
    OptimNaNDebugInfo,
    OptimNaNDebugState,
    OptimResult,
)
from .stop import Stopper
from .types import ModelState, Position

__all__ = ["EmaTrainLossMonitor", "LossMonitor", "OptimEngine"]


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

    ``effective_window`` is an EMA span measured in epoch equivalents, not a hard
    inclusion window or half-life. With a typical multi-batch epoch, a span of one
    epoch equivalent has a half-life of roughly 0.35 epoch equivalents, so recent
    batches receive substantially more weight than early batches from the same
    epoch. The :meth:`from_half_life` constructor uses the large-span approximation
    :math:`e = 2h / \log(2)` for a requested half-life :math:`h`. The exact
    finite-step relationship also depends on the number of batches per epoch.
    """

    effective_window: float = 1.0

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

    ``OptimEngine`` is the low-level execution object behind the experimental
    optimization API. Each epoch starts by asking ``batches`` for fresh batch indices,
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
        Integer seed or JAX PRNG key used for batching and stochastic losses.
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
    progress_n_updates
        Compatibility alias for configuring an approximate maximum number of epoch
        progress-bar updates. The value is converted to ``progress_update_every``;
        reading it returns the resulting effective number of updates.
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
    step_progress_n_updates
        Compatibility alias for configuring an approximate maximum number of batch
        progress-bar updates per epoch. Reading it returns the resulting effective
        number of updates.

    Attributes
    ----------
    position_keys
        Flattened list of all parameter keys claimed by the optimizers.
    split
        Train/validation/test split provided by ``loss.split``.

    Notes
    -----
    ``OptimEngine`` uses ``carry.epoch`` as the number of completed epochs and as the
    next history index to be written. This matches :class:`.Stopper`'s experimental
    indexing convention. Built-in :class:`.LBFGS` is accepted only with full-data
    batches and also requires a deterministic objective, which the engine cannot
    validate. Exact monitor minima retain the post-update position used for the
    evaluation. An EMA minimum instead retains the associated epoch-end checkpoint;
    because an EMA combines losses from several positions, it is not an exact
    loss-position pairing.

    Examples
    --------
    ``OptimEngine`` is usually constructed through a convenience wrapper:

    >>> from liesel.optim import LieselOptim
    >>> LieselOptim.__name__
    'LieselOptim'
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
        progress_n_updates: int | None = None,
        debug_nans: bool = False,
        *,
        loss_monitor: LossMonitor,
        progress_update_every: int = 10,
        show_step_progress: bool = False,
        step_progress_update_every: int = 10,
        step_progress_n_updates: int | None = None,
    ) -> None:
        """Initializes an optimization engine.

        ``progress_n_updates`` retains its historical positional and keyword slot.
        When supplied, it takes precedence over ``progress_update_every``. The batch
        aliases follow the same rule.
        """
        self.loss = loss
        self.batches = batches
        self.optimizers = optimizers
        self.stopper = stopper
        self.seed = jax.random.key(seed) if isinstance(seed, int) else seed
        self.initial_state = initial_state
        self.prune_history = prune_history
        self.show_progress = show_progress
        self.save_position_history = save_position_history
        self.progress_update_every = progress_update_every
        self.loss_monitor = loss_monitor
        self.debug_nans = debug_nans
        self.show_step_progress = show_step_progress
        self.step_progress_update_every = step_progress_update_every

        if progress_n_updates is not None:
            self.progress_n_updates = progress_n_updates
        if step_progress_n_updates is not None:
            self.step_progress_n_updates = step_progress_n_updates

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
    def progress_n_updates(self) -> int:
        """Effective number of epoch updates implied by the update interval."""
        _validate_positive_int(self.progress_update_every, "progress_update_every")
        return _progress_n_updates(self.stopper.epochs, self.progress_update_every)

    @progress_n_updates.setter
    def progress_n_updates(self, value: int) -> None:
        _validate_positive_int(value, "progress_n_updates")
        self.progress_update_every = _progress_print_rate(self.stopper.epochs, value)

    @property
    def step_progress_n_updates(self) -> int:
        """Effective number of batch updates implied by the update interval."""
        _validate_positive_int(
            self.step_progress_update_every, "step_progress_update_every"
        )
        return _progress_n_updates(
            self.batches.n_full_batches, self.step_progress_update_every
        )

    @step_progress_n_updates.setter
    def step_progress_n_updates(self, value: int) -> None:
        _validate_positive_int(value, "step_progress_n_updates")
        self.step_progress_update_every = _progress_print_rate(
            self.batches.n_full_batches, value
        )

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
            If ``loss_monitor`` is not one of the supported sources.
        """
        if not isinstance(self.loss_monitor, EmaTrainLossMonitor) and (
            self.loss_monitor not in ("validation", "train_full_data")
        ):
            raise ValueError(
                "loss_monitor must be EmaTrainLossMonitor(), 'validation', or "
                f"'train_full_data', but got {self.loss_monitor!r}."
            )

        if self.loss_monitor == "validation" and not self.split.has_validation:
            raise ValueError(
                "loss_monitor='validation' requires a split with validation data."
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

        L-BFGS requires full-data batches and a deterministic objective. Only the
        batch requirement can be validated here.

        Raises
        ------
        ValueError
            If a multi-size split is paired with single-size batches, or if batches
            reference keys missing from the training split, or if built-in L-BFGS
            is paired with mini-batches.
        """
        if isinstance(self.split, PositionSplitManager) and isinstance(
            self.batches, Batches
        ):
            raise ValueError(  # noqa: TRY004
                "OptimEngine requires a BatchManager when used with a "
                "PositionSplitManager."
            )

        if not self.batches.is_full_data and any(
            isinstance(opt, LBFGS) for opt in self.optimizers
        ):
            raise ValueError(
                "LBFGS requires full-data batches and a deterministic objective; "
                "configure full-data batches or use another optimizer."
            )

        missing = sorted(
            key for key in self.batches.position_keys if key not in self.split.train
        )
        if missing:
            raise ValueError(
                "Batch position keys must be present in split.train, but these keys "
                f"are missing: {missing}."
            )

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

    def fit(self) -> OptimResult:
        """
        Runs optimization and returns processed results.

        Returns
        -------
        OptimResult
            Processed optimizer history, recommended and diagnostic positions,
            monitoring provenance, and wall-clock runtime.
        """
        start = time.time()
        carry = self._fit()
        end = time.time()
        nan_debug = self._nan_debug_info(carry)
        history = self._process_history(carry.epoch, carry.history)
        n_epochs = int(carry.epoch)
        position_final = carry.position

        if n_epochs == 0:
            position = None
            position_min_monitor = None
            min_monitor_epoch = None
        else:
            position_min_monitor = carry.position_min_monitor
            min_monitor_epoch = int(carry.min_monitor_epoch)
            if isinstance(self.loss_monitor, EmaTrainLossMonitor):
                position = position_final
            else:
                position = position_min_monitor

        monitor_source: Literal["train_ema", "validation", "train_full_data"]
        if isinstance(self.loss_monitor, EmaTrainLossMonitor):
            monitor_source = "train_ema"
        else:
            monitor_source = self.loss_monitor

        result = OptimResult(
            history=history,
            position=position,
            position_final=position_final,
            position_min_monitor=position_min_monitor,
            n_epochs=n_epochs,
            min_monitor_epoch=min_monitor_epoch,
            monitor_source=monitor_source,
            patience=self.stopper.patience,
            duration=end - start,
            nan_debug=nan_debug,
        )
        return result

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
            tracked=carry.tracked,
            history=carry.history,
            batches=debug_state.reproduction_batches,
            optimizer_states=debug_state.reproduction_optimizer_states,
            model_state=debug_state.reproduction_model_state,
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

            if history.tracked is not None:
                for name, value in history.tracked.items():
                    history.tracked[name] = value.at[i:, ...].set(jnp.nan)

        if not self.prune_history:
            return history

        # Remove unused values in history, if applicable
        history.loss_train = history.loss_train[:i]
        history.loss_monitor = history.loss_monitor[:i]
        if self.save_position_history:
            assert history.position is not None
            for name, value in history.position.items():
                history.position[name] = value[:i, ...]

            if history.tracked is not None:
                for name, value in history.tracked.items():
                    history.tracked[name] = value[:i, ...]

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

    def _debug_obs_batch_template(self, batches: BatchConfig) -> Position:
        if not batches.is_full_data or self.split.has_validation or self.split.has_test:
            return batches.get_batched_position(self.split.train, batch_index=0)

        return Position({})

    def _init_nan_debug_state(self, carry: OptimCarry) -> OptimNaNDebugState:
        obs_batch = self._debug_obs_batch_template(carry.batches)
        loss_dtype = jnp.asarray(carry.loss_train).dtype
        return OptimNaNDebugState.new(
            key=carry.key,
            position=carry.position,
            obs_batch=obs_batch,
            optimizer_states=dict(carry.optimizer_states),
            batches=carry.batches,
            model_state=carry.model_state,
            loss_dtype=loss_dtype,
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
        Bi = carry.batches

        if not Bi.is_full_data or self.split.has_validation or self.split.has_test:
            obs_batch = Bi.get_batched_position(self.split.train, batch_index=j)
        else:
            obs_batch = Position({})
        carry.batch = obs_batch

        loss = jnp.zeros_like(carry.loss_train)
        has_active_optimizer = jnp.asarray(False)
        optimizer_loss_has_nan = jnp.asarray(False)
        for opt in self.optimizers:
            is_active = carry.epoch >= opt.activate_after_epochs
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
            lambda carry: self.loss.loss_train_batched(carry.position, carry),
            carry,
        )
        batch_has_nan = (
            optimizer_loss_has_nan | _tree_has_nan(loss) | _tree_has_nan(carry.position)
        )
        loss = jnp.where(batch_has_nan, jnp.nan, loss)
        carry = self._accumulate_loss(loss, carry)

        carry.i_batch = j
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
        Bi = carry.batches

        if not Bi.is_full_data or self.split.has_validation or self.split.has_test:
            obs_batch = Bi.get_batched_position(self.split.train, batch_index=j)
        else:
            obs_batch = Position({})
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

            is_active = carry.epoch >= opt.activate_after_epochs
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
                evaluated_loss = self.loss.loss_train_batched(carry.position, carry)
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
            beta = one - alpha
            carry._ema_numerator = beta * carry._ema_numerator + alpha * loss
            carry._ema_weight = beta * carry._ema_weight + alpha

        return carry

    def _start_epoch(self, carry: OptimCarry) -> OptimCarry:
        """Starts a batch epoch and resets its accumulated losses."""
        key, subkey = jax.random.split(carry.key)
        carry.key = key
        carry.batches = carry.batches.start_epoch(subkey)
        carry.loss_train = jnp.zeros_like(carry.loss_train)
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
        monitoring losses, updates position/tracked histories, updates the global best
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
                self._debug_state(carry).has_nan,
                lambda carry: carry,
                self._finish_epoch,
                carry,
            )

        return self._finish_epoch(carry)

    def _finish_epoch(self, carry: OptimCarry) -> OptimCarry:
        """
        Records losses and histories after a full epoch completed without debug stop.
        """
        i = carry.epoch
        loss_i = carry.loss_train
        carry.history.loss_train = carry.history.loss_train.at[i].set(loss_i)

        if isinstance(self.loss_monitor, EmaTrainLossMonitor):
            loss_monitor_i = carry._ema_numerator / carry._ema_weight
            carry.loss_monitor = loss_monitor_i
            carry.history.loss_monitor = carry.history.loss_monitor.at[i].set(
                loss_monitor_i
            )
        elif self.loss_monitor == "validation":
            key, subkey = jax.random.split(carry.key)
            carry.key = subkey

            loss_monitor_i = self.loss.loss_monitor(carry.position, carry)
            carry.key = key

            carry.loss_monitor = loss_monitor_i
            carry.history.loss_monitor = carry.history.loss_monitor.at[i].set(
                loss_monitor_i
            )
        else:
            loss_monitor_i = self.loss.loss_train(carry.position, carry)
            carry.loss_monitor = loss_monitor_i
            carry.history.loss_monitor = carry.history.loss_monitor.at[i].set(
                loss_monitor_i
            )

        if self.save_position_history:
            assert carry.history.position is not None
            carry.history.position = carry.history.update_position_history(
                carry.epoch, carry.history.position, carry.position
            )
            if carry.history.tracked is not None and carry.tracked is not None:
                carry.history.tracked = carry.history.update_position_history(
                    carry.epoch, carry.history.tracked, carry.tracked
                )

        def update_carry(carry: OptimCarry):
            carry.min_monitor_loss = carry.loss_monitor
            carry.position_min_monitor = carry.position
            carry.min_monitor_epoch = carry.epoch
            return carry

        carry = jax.lax.cond(
            carry.loss_monitor < carry.min_monitor_loss,
            update_carry,
            lambda carry: carry,
            carry,
        )

        carry.epoch += 1

        return carry

    def _continue_fit(self, carry: OptimCarry) -> jax.Array:
        """Returns whether another epoch should be run."""
        loss_train_is_nan = jnp.isnan(carry.loss_train)
        loss_monitor_is_nan = jnp.isnan(carry.loss_monitor)
        no_nan_loss = ~jnp.logical_or(loss_train_is_nan, loss_monitor_is_nan)
        continue_ = self.stopper.continue_(carry.epoch, carry.history.loss_monitor)
        should_continue = jnp.logical_and(no_nan_loss, continue_)

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
            tracked=None,
            optimizers=self.optimizers,
            model_state=self.initial_state,
            save_position_history=self.save_position_history,
        )
        if self.debug_nans:
            carry.batch = self._debug_obs_batch_template(carry.batches)
            carry.nan_debug_state = self._init_nan_debug_state(carry)

        return carry

    def _fit_monolithic(self, carry: OptimCarry) -> OptimCarry:
        """Runs the full fit as one JAX loop without host synchronization."""
        return jax.lax.while_loop(
            cond_fun=self._continue_fit,
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

    def _fit_epoch_chunks(self, carry: OptimCarry, progress_bar) -> OptimCarry:
        """Runs dynamic epoch chunks and updates progress on the host."""
        update_every = self.progress_update_every
        max_epochs = self.stopper.epochs

        @jax.jit
        def run_chunk(carry: OptimCarry):
            target_epoch = jnp.minimum(carry.epoch + update_every, max_epochs)

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
                self._continue_fit(carry),
            )
            return carry, status

        rendered_epochs = 0
        should_continue = True

        while should_continue:
            carry, status = run_chunk(carry)
            completed, loss_train, loss_monitor, continue_value = jax.device_get(status)
            completed_epochs = int(completed)
            should_continue = bool(continue_value)
            rendered_epochs = self._update_outer_progress(
                progress_bar,
                rendered_epochs,
                completed_epochs,
                loss_train,
                loss_monitor,
            )

            # A zero-length chunk can only occur when the initial carry should stop.
            if completed_epochs == 0:
                break

        return carry

    def _fit_nested_progress(
        self, carry: OptimCarry, outer_progress_bar, use_nested_bars: bool
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

            completed_batches = jnp.where(debug_has_nan, carry.i_batch + 1, upper)
            loss_train, loss_monitor = self._completed_epoch_losses(carry)
            status = (
                carry.epoch,
                completed_batches,
                loss_train,
                loss_monitor,
                debug_has_nan,
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
                self._continue_fit(carry),
            )
            return carry, status

        inner_progress_bar = None
        rendered_epochs = 0
        completed_epochs = 0
        should_continue = True

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
                        debug_has_nan_value,
                    ) = jax.device_get(status)

                    debug_has_nan = bool(debug_has_nan_value)
                    completed_batches = int(completed_batches_value)
                    completed_epochs = int(completed)
                    finished_epoch = upper == n_batches and not debug_has_nan
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

                    if debug_has_nan:
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

    def _fit(self) -> OptimCarry:
        """
        Runs optimization with host-controlled progress updates.

        The numerical carry remains on the device. Progress-enabled modes return
        only small status tuples to Python at configured display boundaries, so
        notebook output is never written from a JAX callback thread.
        """
        self._validate_progress_settings()
        carry = self._init_carry(self.stopper.epochs)

        if not self.show_progress:
            return self._fit_monolithic(carry)

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
                    carry, outer_progress_bar, use_nested_bars
                )
            elif self.progress_update_every < self.stopper.epochs:
                carry = self._fit_epoch_chunks(carry, outer_progress_bar)
            else:
                carry = self._fit_monolithic(carry)

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
