from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from liesel.experimental.optim import Batches, OptimEngine, PositionSplit, Stopper
from liesel.experimental.optim.state import OptimCarry
from liesel.experimental.optim.types import Position


@dataclass
class SequenceOptimizer:
    position_keys: list[str]
    values: jax.Array
    identifier: str = "sequence"

    def position(self, position: Position) -> Position:
        return Position({key: position[key] for key in self.position_keys})

    def not_position(self, position: Position) -> Position:
        return Position(
            {
                key: value
                for key, value in position.items()
                if key not in self.position_keys
            }
        )

    def init(self, position: Position):
        return ()

    def step(self, position: Position, loss, carry: OptimCarry) -> OptimCarry:
        del position, loss
        carry.position = Position(carry.position | {"theta": self.values[carry.epoch]})
        return carry


@dataclass
class SequenceLoss:
    split: PositionSplit

    @property
    def model(self):
        return object()

    def position(self, position_keys) -> Position:
        return Position({key: jnp.array(-1.0) for key in position_keys})

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"]

    def loss_validate(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"]

    def grad(self, params: Position, carry: OptimCarry):
        return {key: jnp.zeros_like(value) for key, value in params.items()}

    def value_and_grad(self, params: Position, carry: OptimCarry):
        return self.loss_train_batched(params, carry), self.grad(params, carry)


@pytest.mark.parametrize("save_position_history", [True, False])
def test_engine_restores_global_best_position(save_position_history):
    split = PositionSplit(
        train=Position({"y": jnp.array([0.0])}),
        validate=Position({}),
        test=Position({}),
        n_train=1,
        n_validate=0,
        n_test=0,
    )
    loss = SequenceLoss(split)
    optimizer = SequenceOptimizer(
        position_keys=["theta"], values=jnp.array([0.0, 5.0, 6.0, 7.0])
    )
    engine = OptimEngine(
        loss=loss,
        batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
        split=split,
        optimizers=[optimizer],
        stopper=Stopper(epochs=4, patience=2),
        seed=1,
        initial_state={},
        restore_best_position=True,
        save_position_history=save_position_history,
        show_progress=False,
    )

    result = engine.fit()

    assert result.best_epoch == 0
    assert result.history.loss_validate.tolist() == [0.0, 5.0, 6.0]
    assert result.best_position["theta"] == pytest.approx(0.0)
