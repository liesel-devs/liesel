from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from tqdm import tqdm

from .batch import Batches
from .loss import Loss
from .optimizer import Optimizer
from .split import PositionSplit
from .state import OptimCarry, OptimHistory, OptimResult
from .stop import Stopper
from .types import ModelState, Position

Array = Any


@dataclass
class OptimEngine:
    loss: Loss
    batching_indices: Batches
    data: PositionSplit
    optimizers: Sequence[Optimizer]
    stopper: Stopper
    seed: int | jax.Array
    initial_state: ModelState
    restore_best_position: bool = True
    prune_history: bool = True
    show_progress: bool = True
    save_position_history: bool = True
    progress_n_updates: int = 20
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
        keys = []
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
        ibest = self.stopper.which_best_in_recent_history(
            i=carry.i_it, loss_history=carry.history.loss_validation
        )
        history = self.process_history(carry.i_it, carry.history)

        if self.restore_best_position:
            final_position: Position = Position(
                {name: pos[ibest] for name, pos in carry.history.position.items()}
            )
        else:
            final_position = carry.position

        result = OptimResult(
            history=history,
            final_it=int(carry.i_it),
            best_position=final_position,
            best_it=int(ibest),
            duration=end - start,
        )
        return result

    def process_history(self, i: int, history: OptimHistory) -> OptimHistory:
        # Set unused values in history to nan
        history.loss_train = history.loss_train.at[i:].set(jnp.nan)
        history.loss_validation = history.loss_validation.at[i:].set(jnp.nan)
        if self.save_position_history:
            for name, value in history.position.items():
                history.position[name] = value.at[i:, ...].set(jnp.nan)

            for name, value in history.tracked.items():
                history.tracked[name] = value.at[i:, ...].set(jnp.nan)

        if not self.prune_history:
            return history

        # Remove unused values in history, if applicable
        history.loss_train = history.loss_train[:i]
        history.loss_validation = history.loss_validation[:i]
        if self.save_position_history:
            for name, value in history.position.items():
                history.position[name] = value[:i, ...]
            for name, value in history.tracked.items():
                history.tracked[name] = value[:i, ...]

        return history

    def get_tqdm_callback(self, stopper: Stopper) -> tuple[Callable, Callable]:
        if not self.show_progress:

            def tqdm_callback(carry: OptimCarry):
                return

            def close_progress_bar(carry: OptimCarry):
                return

            return tqdm_callback, close_progress_bar

        if stopper.max_iter > self.progress_n_updates:
            print_rate = int(stopper.max_iter / self.progress_n_updates)
        else:
            print_rate = 1

        progress_bar_inst = tqdm(
            total=stopper.max_iter, desc=("Initializing"), position=0, leave=True
        )

        def tqdm_update(losses, update=print_rate):
            loss_train = float(jnp.squeeze(losses[0]))
            loss_validation = float(jnp.squeeze(losses[1]))
            # if self.data.n_validation > 0:
            desc = (
                f"Training loss: {loss_train:.3f}, "
                f"Validation loss: {loss_validation:.3f}"
            )
            # else:
            #     desc = f"Training loss: {loss_train:.3f}, Validation loss: n/a"
            progress_bar_inst.update(update)
            progress_bar_inst.set_description(desc)
            return losses

        def tqdm_callback(carry: OptimCarry):
            iter_num = carry.i_it + 1

            loss_train, loss_validation = carry.loss_train, carry.loss_validation
            losses = (loss_train, loss_validation)

            _ = jax.lax.cond(
                # update tqdm every multiple of `print_rate` except at the end
                (iter_num % print_rate == 0),
                lambda _: jax.experimental.io_callback(tqdm_update, losses, losses),
                lambda _: losses,
                operand=None,
            )

        def close_progress_bar(carry: OptimCarry):
            print_remainder = int((carry.i_it + 1) % print_rate)
            loss_train, loss_validation = carry.loss_train, carry.loss_validation
            losses = (loss_train, loss_validation)
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
        Bi = carry.batch_indices

        obs_batch = Bi.get_batched_position(self.data.train, batch_index=j)
        carry.batch = obs_batch

        for opt in self.optimizers:
            carry = self.inner_loop_over_optimizers(opt, carry)

        loss = self.loss.loss_train_batched(carry.position, carry)
        carry.loss_train += loss / carry.batch_indices.n_full_batches

        carry.i_batch = j
        carry.batch = Position({})

        return carry

    def outer_loop_over_iterations(self, carry: OptimCarry):
        # permuting the batch indices for each outer iteration
        key, subkey = jax.random.split(carry.key)
        carry.key = key

        carry.batch_indices.indices = carry.batch_indices.permute_indices(subkey)

        carry.loss_train = 0.0
        # run all full batches once
        carry = jax.lax.fori_loop(
            lower=0,
            upper=self.batching_indices.n_full_batches,
            body_fun=self.inner_loop_over_batches,
            init_val=carry,
        )

        i = carry.i_it
        loss_i = carry.loss_train
        carry.history.loss_train = carry.history.loss_train.at[i].set(loss_i)

        if self.data.n_validation > 0:
            key, subkey = jax.random.split(carry.key)
            carry.key = subkey

            loss_val_i = self.loss.loss_validation(carry.position, carry)
            carry.key = key

            carry.loss_validation = loss_val_i
            carry.history.loss_validation = carry.history.loss_validation.at[i].set(
                loss_val_i
            )
        else:
            carry.loss_validation = loss_i
            carry.history.loss_validation = carry.history.loss_validation.at[i].set(
                loss_i
            )

        carry.i_it += 1

        if self.save_position_history:
            carry.history.position = carry.history.update_position_history(
                carry.i_it, carry.history.position, carry.position
            )
            if self.track_keys:
                carry.history.tracked = carry.history.update_position_history(
                    carry.i_it, carry.history.tracked, carry.tracked
                )

        return carry

    def _init_carry(self, niter: int) -> OptimCarry:
        key = self.seed

        initial_position = self.loss.position(self.position_keys)
        if self.track_keys:
            initial_tracked = self.loss.model.extract_position(
                self.track_keys, self.initial_state
            )
        else:
            initial_tracked = Position({})

        carry = OptimCarry.new(
            batch_indices=self.batching_indices,
            key=key,
            niter=niter,
            position=initial_position,
            tracked=initial_tracked,
            optimizers=self.optimizers,
            model_state=self.initial_state,
        )
        return carry

    def _fit(self) -> OptimCarry:
        stopper = self.stopper

        update_progress, close_progress_bar = self.get_tqdm_callback(stopper)

        def while_body(carry: OptimCarry) -> OptimCarry:
            carry = self.outer_loop_over_iterations(carry)

            if self.show_progress:
                update_progress(carry)
            return carry

        def cont(carry: OptimCarry) -> bool:
            loss_train_is_nan = jnp.isnan(carry.loss_train)
            loss_validation_is_nan = jnp.isnan(carry.loss_validation)
            no_nan_loss = ~jnp.logical_or(loss_train_is_nan, loss_validation_is_nan)
            continue_ = stopper.continue_(carry.i_it, carry.history.loss_validation)
            return jnp.logical_and(no_nan_loss, continue_)

        carry = self._init_carry(stopper.max_iter)

        result = jax.lax.while_loop(
            cond_fun=cont,
            body_fun=while_body,
            init_val=carry,
        )

        close_progress_bar(result)

        return result

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(loss={self.loss})"
