from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

import liesel.experimental.optim as opt
import liesel.experimental.optim.engine as engine_module
from liesel.experimental.optim import (
    Batches,
    BatchManager,
    LieselOptim,
    OptimEngine,
    PositionSplit,
    PositionSplitManager,
    Stopper,
)
from liesel.experimental.optim.engine import (
    _progress_print_rate,
    _progress_remainder,
    _should_update_progress,
)
from liesel.experimental.optim.liesel_vi import LieselVI
from liesel.experimental.optim.liesel_optim import LieselOptim as LieselOptimFromQuick
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
    split: PositionSplit | PositionSplitManager

    @property
    def model(self):
        return object()

    def position(self, position_keys) -> Position:
        return Position({key: jnp.array(-1.0) for key in position_keys})

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"]

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"]

    def loss_validate(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"]

    def grad(self, params: Position, carry: OptimCarry):
        return {key: jnp.zeros_like(value) for key, value in params.items()}

    def value_and_grad(self, params: Position, carry: OptimCarry):
        return self.loss_train_batched(params, carry), self.grad(params, carry)


@dataclass
class BatchSensitiveLoss:
    split: PositionSplit

    def position(self, position_keys) -> Position:
        return Position({key: jnp.array(0.0) for key in position_keys})

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        obs = carry.batch if carry.batch else self.split.train
        return params["theta"] + jnp.sum(obs["y"])

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"] + jnp.sum(self.split.train["y"])

    def loss_validate(self, params: Position, carry: OptimCarry) -> jax.Array:
        del params, carry
        return jnp.array(-999.0)

    def grad(self, params: Position, carry: OptimCarry):
        del carry
        return {key: jnp.zeros_like(value) for key, value in params.items()}

    def value_and_grad(self, params: Position, carry: OptimCarry):
        return self.loss_train_batched(params, carry), self.grad(params, carry)


def _split() -> PositionSplit:
    return PositionSplit(
        train=Position({"y": jnp.array([0.0])}),
        validate=Position({}),
        test=Position({}),
        n_train=1,
        n_validate=0,
        n_test=0,
    )


def _loss() -> SequenceLoss:
    return SequenceLoss(_split())


def _monitor_split() -> PositionSplit:
    return PositionSplit(
        train=Position({"y": jnp.array([1.0, 3.0])}),
        validate=Position({}),
        test=Position({}),
        n_train=2,
        n_validate=0,
        n_test=0,
    )


def _optimizer(
    position_keys: list[str] | None = None, identifier: str = "sequence"
) -> SequenceOptimizer:
    return SequenceOptimizer(
        position_keys=position_keys or ["theta"],
        values=jnp.array([0.0, 5.0, 6.0, 7.0]),
        identifier=identifier,
    )


@pytest.mark.parametrize("save_position_history", [True, False])
def test_engine_restores_global_best_position(save_position_history):
    loss = _loss()
    optimizer = _optimizer()
    engine = OptimEngine(
        loss=loss,
        batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
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


def test_engine_uses_loss_split():
    loss = _loss()
    engine = OptimEngine(
        loss=loss,
        batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=4, patience=2),
        seed=1,
        initial_state={},
        show_progress=False,
    )

    assert engine.split is loss.split


def test_empty_optimizers_raise():
    with pytest.raises(ValueError, match="at least one optimizer"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
            optimizers=[],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


def test_duplicate_optimizer_position_keys_raise():
    with pytest.raises(ValueError, match="Position keys"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
            optimizers=[
                _optimizer(["theta"], identifier="a"),
                _optimizer(["theta"], identifier="b"),
            ],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


def test_duplicate_optimizer_identifiers_after_naming_raise():
    with pytest.raises(ValueError, match="identifiers"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
            optimizers=[
                _optimizer(["theta"], identifier=""),
                _optimizer(["eta"], identifier="000"),
            ],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


def test_invalid_progress_n_updates_raises():
    with pytest.raises(ValueError, match="progress_n_updates"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            progress_n_updates=0,
        )


def test_invalid_train_monitor_raises():
    with pytest.raises(ValueError, match="train_monitor"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            train_monitor="sometimes",  # type: ignore[arg-type]
        )


def test_no_validation_epoch_average_monitor_uses_arithmetic_average():
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], n=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        train_monitor="epoch_average",
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([2.0])
    assert result.history.loss_validate.tolist() == pytest.approx([2.0])


def test_no_validation_full_data_monitor_uses_exact_training_loss():
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], n=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        train_monitor="full_data",
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([2.0])
    assert result.history.loss_validate.tolist() == pytest.approx([4.0])


def test_no_validation_auto_monitor_uses_exact_loss_for_full_data_batches():
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], n=2, batch_size=None, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        train_monitor="auto",
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([4.0])
    assert result.history.loss_validate.tolist() == pytest.approx([4.0])


@pytest.mark.parametrize("train_monitor", ["auto", "weighted_epoch_average"])
def test_no_validation_weighted_epoch_average_weights_later_batches(train_monitor):
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], n=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        train_monitor=train_monitor,
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([2.0])
    assert result.history.loss_validate.tolist() == pytest.approx([7.0 / 3.0])


def test_split_manager_requires_batch_manager():
    split = PositionSplitManager(
        [
            PositionSplit(
                Position({"y": jnp.array([0.0])}), Position({}), Position({}), 1, 0, 0
            ),
            PositionSplit(
                Position({"z": jnp.array([0.0])}), Position({}), Position({}), 1, 0, 0
            ),
        ]
    )

    with pytest.raises(ValueError, match="BatchManager"):
        OptimEngine(
            loss=SequenceLoss(split),
            batches=Batches(["y"], n=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


def test_batch_keys_must_be_present_in_training_split():
    with pytest.raises(ValueError, match="split.train"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["missing"], n=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


def test_batch_manager_keys_must_be_present_in_training_split():
    with pytest.raises(ValueError, match="split.train"):
        OptimEngine(
            loss=_loss(),
            batches=BatchManager(
                [Batches(["missing"], n=1, batch_size=None, shuffle=False)]
            ),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


def test_api_imports_after_engine_refactor():
    assert opt.OptimEngine is OptimEngine
    assert opt.LieselOptim is LieselOptim
    assert LieselOptimFromQuick is LieselOptim
    assert opt.LieselVI is LieselVI
    assert not hasattr(opt, "QuickOptim")
    assert not hasattr(engine_module, "QuickOptim")
    assert not hasattr(engine_module, "LieselVI")


def test_progress_helpers_use_completed_epochs():
    assert _progress_print_rate(100, 10) == 10
    assert bool(_should_update_progress(10, 10))
    assert not bool(_should_update_progress(9, 10))
    assert _progress_remainder(23, 10) == 3
