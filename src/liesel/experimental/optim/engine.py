from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from ...model import Model
from .batch import Batches
from .loss import Loss, NegLogProbLoss
from .optimizer import LBFGS, Optimizer
from .split import PositionSplit, Split
from .state import OptimCarry, OptimHistory, OptimResult
from .stop import Stopper
from .types import ModelState, Position
from .util import guess_n

Array = Any


class QuickOptim:
    """
    Enables quick optimizer setup by making providing strong defaults.

    1. Default loss: Unnornalized negative log posterior
    2. Default batching: None
    3. Default stopping: ``Stopper(epochs=1000, patience=10, rtol=1e-6)``
    4. Default Train/Validation/Test split: None (use all data for training)
    5. Default optimizer: Adam(learning_rate=1e-3)
    """

    def __init__(
        self,
        model: Model,
        loss: Loss | None = None,
        batches: Batches | None = None,
        split: PositionSplit | None = None,
        optimizers: Sequence[Optimizer] | Literal["adam", "lbfgs"] = "adam",
        stopper: Stopper = Stopper(epochs=1000, patience=10, rtol=1e-6),
        seed: int | None = None,
        n: int | None = None,
    ) -> None:
        self.model = model
        self._n = n
        self._loss = loss
        self.stopper = stopper
        self._batches = batches
        self.seed = int(time.time()) if seed is None else seed
        self._split = split
        self._optimizers = optimizers

    @property
    def n(self) -> int:
        if self._n is not None:
            return self._n

        return guess_n(self.model)

    @property
    def batches(self) -> Batches:
        if self._batches is not None:
            self._batches.n = self.split.n_train
            self._batches.indices = jnp.arange(self.split.n_train)
            return self._batches

        b = Batches(
            position_keys=list(self.model.observed),
            n=self.split.n_train,
            batch_size=None,
            axes=None,
            default_axis=0,
            shuffle=False,
        )

        return b

    @property
    def split(self) -> PositionSplit:
        if self._split:
            return self._split

        splitter = Split(list(self.model.observed), n=self.n)
        pos = self.model.extract_position(list(self.model.observed))
        split = splitter.split_position(pos)
        return split

    @property
    def optimizers(self) -> Sequence[Optimizer]:
        if isinstance(self._optimizers, str):
            match self._optimizers:
                case "lbfgs":
                    return [LBFGS(list(self.model.parameters))]
                case "adam":
                    opt = Optimizer(
                        list(self.model.parameters),
                        optimizer=optax.adam(learning_rate=1e-3),
                    )
                    return [opt]

        return self._optimizers

    @property
    def loss(self) -> Loss:
        if self._loss is not None:
            return self._loss

        return NegLogProbLoss(self.model, self.split)

    def build_engine(self) -> OptimEngine:
        engine = OptimEngine(
            loss=self.loss,
            batches=self.batches,
            split=self.split,
            optimizers=self.optimizers,
            stopper=self.stopper,
            initial_state=self.model.state,
            seed=self.seed,
        )
        return engine


@dataclass
class OptimEngine:
    loss: Loss
    batches: Batches
    split: PositionSplit
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
        ibest = self.stopper.which_best_in_recent_history(
            i=carry.epoch, loss_history=carry.history.loss_validate
        )
        history = self.process_history(carry.epoch, carry.history)

        if self.restore_best_position:
            final_position: Position = Position(
                {name: pos[ibest] for name, pos in carry.history.position.items()}
            )
        else:
            final_position = carry.position

        result = OptimResult(
            history=history,
            final_epoch=int(carry.epoch),
            best_position=final_position,
            best_epoch=int(ibest),
            duration=end - start,
        )
        return result

    def process_history(self, i: int, history: OptimHistory) -> OptimHistory:
        # Set unused values in history to nan
        history.loss_train = history.loss_train.at[i:].set(jnp.nan)
        history.loss_validate = history.loss_validate.at[i:].set(jnp.nan)
        if self.save_position_history:
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
            return losses

        def tqdm_callback(carry: OptimCarry):
            iter_num = carry.epoch + 1

            loss_train, loss_validate = carry.loss_train, carry.loss_validate
            losses = (loss_train, loss_validate)

            _ = jax.lax.cond(
                (iter_num % print_rate == 0),
                lambda _: jax.experimental.io_callback(tqdm_update, losses, losses),
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

        obs_batch = Bi.get_batched_position(self.split.train, batch_index=j)
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

        carry.batches.indices = carry.batches.permute_indices(subkey)

        carry.loss_train = 0.0
        # run all full batches once
        carry = jax.lax.fori_loop(
            lower=0,
            upper=self.batches.n_full_batches,
            body_fun=self.inner_loop_over_batches,
            init_val=carry,
        )

        i = carry.epoch
        loss_i = carry.loss_train
        carry.history.loss_train = carry.history.loss_train.at[i].set(loss_i)

        if self.split.n_validate > 0:
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
            carry.history.position = carry.history.update_position_history(
                carry.epoch, carry.history.position, carry.position
            )
            if carry.history.tracked is not None and carry.tracked is not None:
                carry.history.tracked = carry.history.update_position_history(
                    carry.epoch, carry.history.tracked, carry.tracked
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
