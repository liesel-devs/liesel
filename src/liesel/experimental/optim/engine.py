"""Low-level optimization engine for experimental optimizers.

The :class:`OptimEngine` class coordinates losses, optimizers, mini-batches,
train/validation/test splits, early stopping, and optimizer history recording. Most
users will usually construct it through :class:`.QuickOptim` or :class:`.LieselVI`,
but direct construction is useful for custom losses or optimizer schedules.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from tqdm import tqdm

from ._engine_utils import BatchConfig, SplitConfig
from .batch import Batches
from .loss import Loss
from .optimizer import Optimizer
from .split import PositionSplitManager
from .state import OptimCarry, OptimHistory, OptimResult
from .stop import Stopper
from .types import ModelState, Position

__all__ = ["OptimEngine"]


def _progress_print_rate(epochs: int, progress_n_updates: int) -> int:
    return max(int(epochs / progress_n_updates), 1)


def _should_update_progress(completed_epochs: int | jax.Array, print_rate: int):
    return completed_epochs % print_rate == 0


def _progress_remainder(completed_epochs: int | jax.Array, print_rate: int) -> int:
    return int(completed_epochs % print_rate)


@dataclass
class OptimEngine:
    """
    Runs an optimization loop over epochs, batches, and optimizers.

    ``OptimEngine`` is the low-level execution object behind the experimental
    optimization API. Each epoch starts by asking ``batches`` for fresh batch indices,
    then iterates over all full batches. For each batch, each optimizer gets a turn
    to update the subset of parameters named in its ``position_keys``. At the end of
    the epoch, the engine records training and validation losses, updates the global
    best position, and asks ``stopper`` whether to continue.

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
        keys.
    stopper
        Early-stopping and maximum-epoch configuration.
    seed
        Integer seed or JAX PRNG key used for batching and stochastic losses.
    initial_state
        Initial model state passed into :class:`.OptimCarry`.
    restore_best_position
        If ``True``, :meth:`fit` returns the global best position found during the
        run. If ``False``, it returns the final position.
    prune_history
        If ``True``, remove unused history entries after early stopping.
    show_progress
        Whether to show a ``tqdm`` progress bar.
    save_position_history
        Whether to store the full position history. The global best position is
        tracked independently of this setting.
    progress_n_updates
        Approximate maximum number of progress-bar updates.

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
    indexing convention.

    Examples
    --------
    ``OptimEngine`` is usually constructed through a convenience wrapper:

    >>> from liesel.experimental.optim import QuickOptim
    >>> QuickOptim.__name__
    'QuickOptim'
    """

    loss: Loss
    batches: BatchConfig
    optimizers: Sequence[Optimizer]
    stopper: Stopper
    seed: int | jax.Array
    initial_state: ModelState
    restore_best_position: bool = True
    prune_history: bool = True
    show_progress: bool = True
    save_position_history: bool = True
    progress_n_updates: int = 100

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
        self._validate_progress_settings()
        self._validate_batch_split_compatibility()

        if isinstance(self.seed, int):
            self.seed = jax.random.key(self.seed)

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
            If ``progress_n_updates`` is not a positive integer.
        """
        if isinstance(self.progress_n_updates, bool) or self.progress_n_updates < 1:
            raise ValueError("progress_n_updates must be a positive integer.")

    def _validate_batch_split_compatibility(self) -> None:
        """
        Validates that batch and split configurations can be used together.

        Raises
        ------
        ValueError
            If a multi-size split is paired with single-size batches, or if batches
            reference keys missing from the training split.
        """
        if isinstance(self.split, PositionSplitManager) and isinstance(
            self.batches, Batches
        ):
            raise ValueError(
                "OptimEngine requires a BatchManager when used with a "
                "PositionSplitManager."
            )

        missing = sorted(
            key for key in self.batches.position_keys if key not in self.split.train
        )
        if missing:
            raise ValueError(
                "Batch position keys must be present in split.train, but these keys "
                f"are missing: {missing}."
            )

    def _name_optimizers(self) -> Sequence[Optimizer]:
        """
        Fills missing optimizer identifiers with stable numeric names.

        Optimizer states are stored by identifier in :class:`.OptimCarry`. This
        method mutates optimizers whose ``identifier`` is empty and leaves existing
        identifiers unchanged.

        Returns
        -------
        collections.abc.Sequence[Optimizer]
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
            Processed optimizer history, selected result position, best epoch, and
            wall-clock runtime.

        Notes
        -----
        ``OptimResult.best_epoch`` always refers to the global best validation loss
        seen during the run. With ``restore_best_position=True``,
        ``OptimResult.best_position`` is the corresponding global best position. With
        ``restore_best_position=False``, ``best_position`` contains the final
        position while ``best_epoch`` still reports the global best epoch.
        """
        start = time.time()
        carry = self._fit()
        end = time.time()
        history = self._process_history(carry.epoch, carry.history)
        best_epoch = int(carry.best_epoch)

        if self.restore_best_position:
            final_position = carry.best_position
        else:
            final_position = carry.position

        result = OptimResult(
            history=history,
            final_epoch=int(carry.epoch),
            best_position=final_position,
            best_epoch=best_epoch,
            duration=end - start,
        )
        return result

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
        history.loss_validate = history.loss_validate.at[i:].set(jnp.nan)
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
        history.loss_validate = history.loss_validate[:i]
        if self.save_position_history:
            assert history.position is not None
            for name, value in history.position.items():
                history.position[name] = value[:i, ...]

            if history.tracked is not None:
                for name, value in history.tracked.items():
                    history.tracked[name] = value[:i, ...]

        return history

    def _get_tqdm_callback(self, stopper: Stopper) -> tuple[Callable, Callable]:
        """
        Creates progress-bar update and close callbacks.

        Parameters
        ----------
        stopper
            Stopper whose ``epochs`` value determines the progress-bar length.

        Returns
        -------
        tuple[Callable, Callable]
            A pair ``(update_progress, close_progress_bar)``. Both callbacks accept an
            :class:`.OptimCarry`.
        """
        print_rate = _progress_print_rate(stopper.epochs, self.progress_n_updates)

        progress_bar_inst = tqdm(
            total=stopper.epochs, desc=("Initializing"), position=0, leave=True
        )

        def tqdm_update(losses, update=print_rate):
            loss_train = float(jnp.squeeze(losses[0]))
            loss_validate = float(jnp.squeeze(losses[1]))
            desc = (
                f"Training loss: {loss_train:.3f}, Validation loss: {loss_validate:.3f}"
            )
            progress_bar_inst.update(update)
            progress_bar_inst.set_description(desc)

        def tqdm_callback(carry: OptimCarry):
            completed_epochs = carry.epoch

            loss_train, loss_validate = carry.loss_train, carry.loss_validate
            losses = (loss_train, loss_validate)

            def true_fn(_):
                jax.debug.callback(tqdm_update, losses, ordered=True)
                return losses

            _ = jax.lax.cond(
                _should_update_progress(completed_epochs, print_rate),
                true_fn,
                lambda _: losses,
                operand=None,
            )

        def close_progress_bar(carry: OptimCarry):
            print_remainder = _progress_remainder(carry.epoch, print_rate)
            loss_train, loss_validate = carry.loss_train, carry.loss_validate
            losses = (loss_train, loss_validate)
            tqdm_update(losses, print_remainder)
            progress_bar_inst.close()

        return tqdm_callback, close_progress_bar

    def _run_optimizer_step(self, opt: Optimizer, carry: OptimCarry) -> OptimCarry:
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
        OptimCarry
            Updated carry with ``carry.position`` modified by ``opt``.
        """
        # subset of the position handled by this optimizer
        pos = Position(opt.position(carry.position))

        # parameters handled by other optimizers
        carry.fixed_position = Position(opt.not_position(carry.position))

        key, subkey = jax.random.split(carry.key)
        carry.key = subkey
        carry = opt.step(pos, self.loss, carry)
        carry.key = key
        carry.fixed_position = Position({})  # reset fixed position

        return carry

    def _run_batch(self, j: int | jax.Array, carry: OptimCarry) -> OptimCarry:
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

        for opt in self.optimizers:
            carry = self._run_optimizer_step(opt, carry)

        loss = self.loss.loss_train_batched(carry.position, carry)
        carry.loss_train += loss / carry.batches.n_full_batches

        carry.i_batch = j
        carry.batch = Position({})

        return carry

    def _run_epoch(self, carry: OptimCarry) -> OptimCarry:
        """
        Runs one full epoch over the configured batches.

        The method starts a new batch epoch, runs the batch loop, records train and
        validation losses, updates position/tracked histories, updates the global best
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
        # permuting the batch indices for each outer iteration
        key, subkey = jax.random.split(carry.key)
        carry.key = key

        carry.batches = carry.batches.start_epoch(subkey)

        carry.loss_train = 0.0
        # run all full batches once
        carry = jax.lax.fori_loop(
            lower=0,
            upper=carry.batches.n_full_batches,
            body_fun=self._run_batch,
            init_val=carry,
        )

        i = carry.epoch
        loss_i = carry.loss_train
        carry.history.loss_train = carry.history.loss_train.at[i].set(loss_i)

        if self.split.has_validation:
            key, subkey = jax.random.split(carry.key)
            carry.key = subkey

            loss_val_i = self.loss.loss_validate(carry.position, carry)
            carry.key = key

            carry.loss_validate = loss_val_i
            carry.history.loss_validate = carry.history.loss_validate.at[i].set(
                loss_val_i
            )
        else:
            carry.loss_validate = loss_i
            carry.history.loss_validate = carry.history.loss_validate.at[i].set(loss_i)

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
            carry.best_loss = carry.loss_validate
            carry.best_position = carry.position
            carry.best_epoch = carry.epoch
            return carry

        carry = jax.lax.cond(
            carry.loss_validate < carry.best_loss,
            update_carry,
            lambda carry: carry,
            carry,
        )

        carry.epoch += 1

        return carry

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
        return carry

    def _fit(self) -> OptimCarry:
        """
        Runs the JAX ``while_loop`` backing :meth:`fit`.

        Returns
        -------
        OptimCarry
            Final carry after early stopping, reaching ``stopper.epochs``, or
            encountering ``nan`` loss values.
        """
        stopper = self.stopper

        if self.show_progress:
            update_progress, close_progress_bar = self._get_tqdm_callback(stopper)

        def while_body(carry: OptimCarry) -> OptimCarry:
            carry = self._run_epoch(carry)

            if self.show_progress:
                update_progress(carry)
            return carry

        def cont(carry: OptimCarry) -> bool:
            loss_train_is_nan = jnp.isnan(carry.loss_train)
            loss_validate_is_nan = jnp.isnan(carry.loss_validate)
            no_nan_loss = ~jnp.logical_or(loss_train_is_nan, loss_validate_is_nan)
            continue_ = stopper.continue_(carry.epoch, carry.history.loss_validate)
            return jnp.logical_and(no_nan_loss, continue_)

        carry = self._init_carry(stopper.epochs)

        result = jax.lax.while_loop(
            cond_fun=cont,
            body_fun=while_body,
            init_val=carry,
        )

        if self.show_progress:
            close_progress_bar(result)

        return result

    def __repr__(self) -> str:
        """Returns a compact representation showing the configured loss."""
        name = type(self).__name__
        return f"{name}(loss={self.loss})"
