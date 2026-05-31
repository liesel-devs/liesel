from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from tqdm import tqdm

from ._engine_utils import BatchConfig, SplitConfig
from .loss import Loss
from .optimizer import Optimizer
from .state import OptimCarry, OptimHistory, OptimResult
from .stop import Stopper
from .types import ModelState, Position

if TYPE_CHECKING:
    from .liesel_vi import LieselVI
    from .quick import QuickOptim

__all__ = ["LieselVI", "OptimEngine", "QuickOptim"]


def __getattr__(name: str) -> object:
    if name == "LieselVI":
        from .liesel_vi import LieselVI

        return LieselVI

    if name == "QuickOptim":
        from .quick import QuickOptim

        return QuickOptim

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class OptimEngine:
    loss: Loss
    batches: BatchConfig
    split: SplitConfig
    optimizers: Sequence[Optimizer]
    stopper: Stopper
    seed: int | jax.Array
    initial_state: ModelState
    restore_best_position: bool = True
    prune_history: bool = True
    show_progress: bool = True
    save_position_history: bool = True
    progress_n_updates: int = 100
    track_keys: Sequence[str] = field(default_factory=list)

    def __post_init__(self):
        self.validate_position_keys()
        self.name_optimizers()

        if isinstance(self.seed, int):
            self.seed = jax.random.key(self.seed)

        if len(self.track_keys) > 0:
            raise NotImplementedError(
                "The argument track_keys=True is currently not operational. "
                "Please set to []."
            )

    @property
    def position_keys(self) -> list[str]:
        keys: list[str] = []
        for optim in self.optimizers:
            keys += optim.position_keys
        return keys

    def validate_position_keys(self) -> None:
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

    def name_optimizers(self) -> Sequence[Optimizer]:
        for i, opt in enumerate(self.optimizers):
            if not opt.identifier:
                opt.identifier = f"{i:03}"
        return self.optimizers

    def fit(self) -> OptimResult:
        start = time.time()
        carry = self._fit()
        end = time.time()
        history = self.process_history(carry.epoch, carry.history)
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

    def process_history(self, i: int, history: OptimHistory) -> OptimHistory:
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

    def get_tqdm_callback(self, stopper: Stopper) -> tuple[Callable, Callable]:
        if stopper.epochs > self.progress_n_updates:
            print_rate = int(stopper.epochs / self.progress_n_updates)
        else:
            print_rate = 1

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
            iter_num = carry.epoch + 1

            loss_train, loss_validate = carry.loss_train, carry.loss_validate
            losses = (loss_train, loss_validate)

            def true_fn(_):
                jax.debug.callback(tqdm_update, losses, ordered=True)
                return losses

            _ = jax.lax.cond(
                (iter_num % print_rate == 0),
                true_fn,
                lambda _: losses,
                operand=None,
            )

        def close_progress_bar(carry: OptimCarry):
            print_remainder = int((carry.epoch + 1) % print_rate)
            loss_train, loss_validate = carry.loss_train, carry.loss_validate
            losses = (loss_train, loss_validate)
            tqdm_update(losses, print_remainder)
            progress_bar_inst.close()

        return tqdm_callback, close_progress_bar

    def inner_loop_over_optimizers(self, opt, carry: OptimCarry):
        # subset of the position handled by this optimizer
        pos = opt.position(carry.position)

        # parameters handled by other optimizers
        carry.fixed_position = opt.not_position(carry.position)

        key, subkey = jax.random.split(carry.key)
        carry.key = subkey
        carry = opt.step(pos, self.loss, carry)
        carry.key = key
        carry.fixed_position = Position({})  # reset fixed position

        return carry

    def inner_loop_over_batches(self, j, carry: OptimCarry):
        Bi = carry.batches

        if not Bi.is_full_data or self.split.has_validation or self.split.has_test:
            obs_batch = Bi.get_batched_position(self.split.train, batch_index=j)
        else:
            obs_batch = Position({})
        carry.batch = obs_batch

        for opt in self.optimizers:
            carry = self.inner_loop_over_optimizers(opt, carry)

        loss = self.loss.loss_train_batched(carry.position, carry)
        carry.loss_train += loss / carry.batches.n_full_batches

        carry.i_batch = j
        carry.batch = Position({})

        return carry

    def outer_loop_over_epochs(self, carry: OptimCarry):
        # permuting the batch indices for each outer iteration
        key, subkey = jax.random.split(carry.key)
        carry.key = key

        carry.batches = carry.batches.start_epoch(subkey)

        carry.loss_train = 0.0
        # run all full batches once
        carry = jax.lax.fori_loop(
            lower=0,
            upper=carry.batches.n_full_batches,
            body_fun=self.inner_loop_over_batches,
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
        key = self.seed

        initial_position = self.loss.position(self.position_keys)
        if self.track_keys:
            initial_tracked = self.loss.model.extract_position(
                self.track_keys, self.initial_state
            )
        else:
            initial_tracked = Position({})

        carry = OptimCarry.new(
            batches=self.batches,
            key=key,
            epochs=epochs,
            position=initial_position,
            tracked=initial_tracked,
            optimizers=self.optimizers,
            model_state=self.initial_state,
            save_position_history=self.save_position_history,
        )
        return carry

    def _fit(self) -> OptimCarry:
        stopper = self.stopper

        if self.show_progress:
            update_progress, close_progress_bar = self.get_tqdm_callback(stopper)

        def while_body(carry: OptimCarry) -> OptimCarry:
            carry = self.outer_loop_over_epochs(carry)

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
        name = type(self).__name__
        return f"{name}(loss={self.loss})"
