from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

import liesel.optim as opt
import liesel.optim.engine as engine_module
from liesel.optim import (
    Batches,
    BatchManager,
    LieselOptim,
    OptimEngine,
    PositionSplit,
    PositionSplitManager,
    Stopper,
)
from liesel.optim.engine import _progress_print_rate
from liesel.optim.liesel_optim import LieselOptim as LieselOptimFromQuick
from liesel.optim.loss import Loss
from liesel.optim.state import OptimCarry
from liesel.optim.types import Position


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


@dataclass
class DebugNoOpOptimizer:
    position_keys: list[str]
    identifier: str = "noop"

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
        del position
        return jnp.array(0)

    def step(self, position: Position, loss, carry: OptimCarry) -> OptimCarry:
        del position, loss
        carry.optimizer_states[self.identifier] += 1
        return carry


@dataclass
class AddOneOptimizer(DebugNoOpOptimizer):
    def step(self, position: Position, loss, carry: OptimCarry) -> OptimCarry:
        del loss
        key = self.position_keys[0]
        carry.position = Position(carry.position | {key: position[key] + 1.0})
        carry.optimizer_states[self.identifier] += 1
        return carry


@dataclass
class NanOptimizer(DebugNoOpOptimizer):
    def step(self, position: Position, loss, carry: OptimCarry) -> OptimCarry:
        del loss
        key = self.position_keys[0]
        carry.position = Position(carry.position | {key: position[key] * jnp.nan})
        carry.optimizer_states[self.identifier] += 1
        return carry


@dataclass
class DebugNaNLoss:
    split: PositionSplit
    trigger_batch_value: float | None = None
    trigger_epoch: int | None = None
    initial_nan: bool = False

    def position(self, position_keys) -> Position:
        return Position(
            {
                key: jnp.array(jnp.nan if self.initial_nan else 0.0)
                for key in position_keys
            }
        )

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        param_sum = sum(jnp.sum(value) for value in params.values())
        batch_sum = sum(jnp.sum(value) for value in carry.batch.values())
        loss = param_sum + batch_sum
        if self.trigger_batch_value is None:
            return loss

        trigger = jnp.asarray(False)
        for value in carry.batch.values():
            trigger = trigger | jnp.any(value == self.trigger_batch_value)
        if self.trigger_epoch is not None:
            trigger = trigger & (carry.epoch == self.trigger_epoch)
        return jnp.where(trigger, jnp.nan, loss)

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return sum(jnp.sum(value) for value in params.values())

    def loss_validate(self, params: Position, carry: OptimCarry) -> jax.Array:
        return self.loss_train(params, carry)

    def grad(self, params: Position, carry: OptimCarry):
        del carry
        return Position({key: jnp.zeros_like(value) for key, value in params.items()})

    def value_and_grad(self, params: Position, carry: OptimCarry):
        return self.loss_train_batched(params, carry), self.grad(params, carry)


def _split() -> PositionSplit:
    return PositionSplit(
        train=Position({"y": jnp.array([0.0])}),
        validate=Position({}),
        test=Position({}),
        train_axis_size=1,
        validate_axis_size=0,
        test_axis_size=0,
    )


def _loss() -> SequenceLoss:
    return SequenceLoss(_split())


def _monitor_split() -> PositionSplit:
    return PositionSplit(
        train=Position({"y": jnp.array([1.0, 3.0])}),
        validate=Position({}),
        test=Position({}),
        train_axis_size=2,
        validate_axis_size=0,
        test_axis_size=0,
    )


def _optimizer(
    position_keys: list[str] | None = None, identifier: str = "sequence"
) -> SequenceOptimizer:
    return SequenceOptimizer(
        position_keys=position_keys or ["theta"],
        values=jnp.array([0.0, 5.0, 6.0, 7.0]),
        identifier=identifier,
    )


def _progress_engine(
    *,
    epochs: int = 5,
    n_batches: int = 5,
    show_progress: bool = True,
    progress_update_every: int = 2,
    show_step_progress: bool = False,
    step_progress_update_every: int = 2,
    debug_nans: bool = False,
    loss: Callable[[PositionSplit], Loss] | None = None,
) -> OptimEngine:
    split = PositionSplit(
        train=Position({"y": jnp.arange(float(n_batches))}),
        validate=Position({}),
        test=Position({}),
        train_axis_size=n_batches,
        validate_axis_size=0,
        test_axis_size=0,
    )
    resolved_loss = BatchSensitiveLoss(split) if loss is None else loss(split)
    return OptimEngine(
        loss=resolved_loss,
        batches=Batches(
            ["y"],
            axis_size=n_batches,
            batch_size=1,
            shuffle=False,
        ),
        optimizers=[DebugNoOpOptimizer(["theta"])],
        stopper=Stopper(epochs=epochs, patience=epochs),
        seed=1,
        initial_state={},
        show_progress=show_progress,
        progress_update_every=progress_update_every,
        show_step_progress=show_step_progress,
        step_progress_update_every=step_progress_update_every,
        debug_nans=debug_nans,
    )


class FakeTqdm:
    instances: list[FakeTqdm] = []

    def __init__(self, total, desc, position, leave):
        self.total = total
        self.desc = desc
        self.position = position
        self.leave = leave
        self.n = 0
        self.updates = []
        self.descriptions = [desc]
        self.thread_ids = [threading.get_ident()]
        self.closed = False
        type(self).instances.append(self)

    def update(self, value):
        self.thread_ids.append(threading.get_ident())
        self.updates.append(value)
        self.n += value

    def set_description(self, desc, refresh=True):
        del refresh
        self.thread_ids.append(threading.get_ident())
        self.desc = desc
        self.descriptions.append(desc)

    def reset(self, total=None):
        self.thread_ids.append(threading.get_ident())
        self.n = 0
        if total is not None:
            self.total = total

    def close(self):
        self.thread_ids.append(threading.get_ident())
        self.closed = True


@pytest.mark.parametrize("save_position_history", [True, False])
def test_engine_restores_global_best_position(save_position_history):
    loss = _loss()
    optimizer = _optimizer()
    engine = OptimEngine(
        loss=loss,
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
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
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
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
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
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
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
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
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[
                _optimizer(["theta"], identifier=""),
                _optimizer(["eta"], identifier="000"),
            ],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


@pytest.mark.parametrize("progress_n_updates", [0, True, 1.5, "100"])
def test_invalid_progress_n_updates_raises(progress_n_updates):
    with pytest.raises(ValueError, match="progress_n_updates"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            progress_n_updates=progress_n_updates,
        )


@pytest.mark.parametrize(
    "name", ["progress_update_every", "step_progress_update_every"]
)
@pytest.mark.parametrize("value", [0, True, 1.5, "10"])
def test_invalid_progress_update_interval_raises(name, value):
    kwargs = {name: value}
    with pytest.raises(ValueError, match=name):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            **kwargs,
        )


@pytest.mark.parametrize("name", ["progress_n_updates", "step_progress_n_updates"])
@pytest.mark.parametrize("value", [0, True, 1.5, "10"])
def test_invalid_progress_update_count_raises(name, value):
    kwargs = {name: value}
    with pytest.raises(ValueError, match=name):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            **kwargs,
        )


def test_invalid_train_monitor_raises():
    with pytest.raises(ValueError, match="train_monitor"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            train_monitor="sometimes",  # type: ignore[arg-type]
        )


def test_debug_nans_loss_capture_reproduces_loss():
    split = PositionSplit(
        train=Position({"y": jnp.array([0.0, 1.0, 2.0])}),
        validate=Position({}),
        test=Position({}),
        train_axis_size=3,
        validate_axis_size=0,
        test_axis_size=0,
    )
    engine = OptimEngine(
        loss=DebugNaNLoss(split, trigger_batch_value=1.0),
        batches=Batches(["y"], axis_size=3, batch_size=1, shuffle=False),
        optimizers=[DebugNoOpOptimizer(["theta"])],
        stopper=Stopper(epochs=3, patience=3),
        seed=1,
        initial_state={},
        show_progress=False,
        debug_nans=True,
    )

    result = engine.fit()

    info = result.nan_debug
    assert info is not None
    assert result.final_epoch == 0
    assert info.kind == "loss"
    assert info.batch == 1
    assert info.optimizer_index is None
    assert info.nan_position is None
    assert info.obs_batch["y"].tolist() == pytest.approx([1.0])
    assert info.last_non_nan_position["theta"] == pytest.approx(0.0)
    assert bool(jnp.isnan(info.loss))
    assert bool(jnp.isnan(info.reproduce_loss(engine)))
    assert info.reproduction_carry.batch["y"].tolist() == pytest.approx([1.0])
    assert info.reproduction_carry.fixed_position == {}

    epoch_key, _ = jax.random.split(jax.random.key(1))
    batch0_key, _ = jax.random.split(epoch_key)
    expected_loss_key, _ = jax.random.split(batch0_key)
    assert jnp.array_equal(
        jax.random.key_data(info.reproduction_carry.key),
        jax.random.key_data(expected_loss_key),
    )


def test_debug_nans_position_after_reproduces_second_optimizer_step():
    split = PositionSplit(
        train=Position({"y": jnp.array([0.0, 1.0])}),
        validate=Position({}),
        test=Position({}),
        train_axis_size=2,
        validate_axis_size=0,
        test_axis_size=0,
    )
    engine = OptimEngine(
        loss=DebugNaNLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[
            AddOneOptimizer(["theta"], identifier="add_theta"),
            NanOptimizer(["eta"], identifier="nan_eta"),
        ],
        stopper=Stopper(epochs=3, patience=3),
        seed=1,
        initial_state={},
        show_progress=False,
        debug_nans=True,
    )

    result = engine.fit()

    info = result.nan_debug
    assert info is not None
    assert result.final_epoch == 0
    assert info.kind == "position_after"
    assert info.batch == 0
    assert info.optimizer_index == 1
    assert info.optimizer_identifier == "nan_eta"
    assert info.optimizer_position_keys == ("eta",)
    assert info.last_non_nan_position["theta"] == pytest.approx(1.0)
    assert info.last_non_nan_position["eta"] == pytest.approx(0.0)
    assert info.reproduction_position["theta"] == pytest.approx(1.0)
    assert info.reproduction_position["eta"] == pytest.approx(0.0)
    assert info.reproduction_carry.fixed_position["theta"] == pytest.approx(1.0)
    assert "eta" not in info.reproduction_carry.fixed_position
    assert bool(jnp.isnan(info.nan_position["eta"]))

    carry_after = info.reproduce_step(engine)
    assert bool(jnp.isnan(carry_after.position["eta"]))

    epoch_key, _ = jax.random.split(jax.random.key(1))
    after_first_optimizer_key, _ = jax.random.split(epoch_key)
    _, expected_step_key = jax.random.split(after_first_optimizer_key)
    assert jnp.array_equal(
        jax.random.key_data(info.reproduction_carry.key),
        jax.random.key_data(expected_step_key),
    )


def test_debug_nans_position_before_capture():
    split = _split()
    engine = OptimEngine(
        loss=DebugNaNLoss(split, initial_nan=True),
        batches=Batches(["y"], axis_size=1, batch_size=1, shuffle=False),
        optimizers=[DebugNoOpOptimizer(["theta"])],
        stopper=Stopper(epochs=3, patience=3),
        seed=1,
        initial_state={},
        show_progress=False,
        debug_nans=True,
    )

    result = engine.fit()

    info = result.nan_debug
    assert info is not None
    assert result.final_epoch == 0
    assert info.kind == "position_before"
    assert info.optimizer_index is None
    assert bool(jnp.isnan(info.nan_position["theta"]))


def test_debug_nans_disabled_keeps_existing_nan_loss_behavior():
    split = PositionSplit(
        train=Position({"y": jnp.array([0.0, 1.0])}),
        validate=Position({}),
        test=Position({}),
        train_axis_size=2,
        validate_axis_size=0,
        test_axis_size=0,
    )
    engine = OptimEngine(
        loss=DebugNaNLoss(split, trigger_batch_value=0.0),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[DebugNoOpOptimizer(["theta"])],
        stopper=Stopper(epochs=3, patience=3),
        seed=1,
        initial_state={},
        show_progress=False,
        debug_nans=False,
    )

    result = engine.fit()

    assert result.nan_debug is None
    assert result.final_epoch == 1
    assert bool(jnp.isnan(result.history.loss_train[0]))


def test_no_validation_epoch_average_monitor_uses_arithmetic_average():
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
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
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
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
        batches=Batches(["y"], axis_size=2, batch_size=None, shuffle=False),
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
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
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
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
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
            batches=Batches(["missing"], axis_size=1, batch_size=None, shuffle=False),
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
                [Batches(["missing"], axis_size=1, batch_size=None, shuffle=False)]
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
    assert not hasattr(opt, "LieselVI")
    assert not hasattr(opt, "NegElboLoss")
    assert not hasattr(opt, "Elbo")
    assert not hasattr(opt, "VDist")
    assert not hasattr(opt, "CompositeVDist")
    assert not hasattr(opt, "QuickOptim")
    assert not hasattr(engine_module, "QuickOptim")
    assert not hasattr(engine_module, "LieselVI")


def test_progress_count_conversion_uses_a_ceiling():
    assert _progress_print_rate(100, 10) == 10
    assert _progress_print_rate(101, 100) == 2
    assert _progress_print_rate(201, 100) == 3


def test_progress_defaults_and_linked_count_properties():
    engine = OptimEngine(
        loss=_loss(),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=101, patience=10),
        seed=1,
        initial_state={},
        show_progress=False,
    )

    assert engine.progress_update_every == 10
    assert engine.step_progress_update_every == 10
    assert engine.show_step_progress is False
    assert engine.progress_n_updates == 11
    assert engine.step_progress_n_updates == 1

    engine.progress_n_updates = 100
    assert engine.progress_update_every == 2
    assert engine.progress_n_updates == 51

    engine.stopper = Stopper(epochs=201, patience=10)
    assert engine.progress_n_updates == 101

    engine.batches = Batches(["y"], axis_size=23, batch_size=1, shuffle=False)
    engine.step_progress_n_updates = 10
    assert engine.step_progress_update_every == 3
    assert engine.step_progress_n_updates == 8


def test_progress_count_aliases_override_intervals():
    engine = OptimEngine(
        loss=_loss(),
        batches=Batches(["y"], axis_size=10, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=10, patience=10),
        seed=1,
        initial_state={},
        show_progress=False,
        progress_update_every=2,
        progress_n_updates=3,
        step_progress_update_every=2,
        step_progress_n_updates=4,
    )

    assert engine.progress_update_every == 4
    assert engine.progress_n_updates == 3
    assert engine.step_progress_update_every == 3
    assert engine.step_progress_n_updates == 4


def test_progress_count_keeps_historical_positional_slot():
    engine = OptimEngine(
        _loss(),
        Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        [_optimizer()],
        Stopper(epochs=10, patience=10),
        1,
        {},
        True,
        True,
        False,
        True,
        3,
    )

    assert engine.progress_update_every == 4
    assert engine.progress_n_updates == 3


def test_nested_progress_matches_monolithic_and_never_uses_callback(monkeypatch):
    expected = _progress_engine(show_progress=False).fit()
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)

    def fail_callback(*args, **kwargs):
        del args, kwargs
        raise AssertionError("jax.debug.callback must not be used for progress")

    monkeypatch.setattr(jax.debug, "callback", fail_callback)
    actual_engine = _progress_engine(show_step_progress=True)
    actual = actual_engine.fit()

    assert actual.final_epoch == expected.final_epoch == 5
    assert jnp.allclose(actual.history.loss_train, expected.history.loss_train)
    assert jnp.allclose(actual.history.loss_validate, expected.history.loss_validate)

    assert len(FakeTqdm.instances) == 2
    outer, inner = FakeTqdm.instances
    assert outer.position == 0
    assert outer.updates == [1, 1, 1, 1, 1]
    assert actual_engine.progress_update_every == 2
    assert inner.position == 1
    assert inner.leave is False
    assert inner.updates == [2, 2, 1] * 5
    assert outer.closed and inner.closed
    assert set(outer.thread_ids + inner.thread_ids) == {threading.get_ident()}


def test_large_step_interval_uses_epoch_only_progress(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    engine = _progress_engine(
        show_step_progress=True,
        step_progress_update_every=5,
    )

    result = engine.fit()

    assert result.final_epoch == 5
    assert len(FakeTqdm.instances) == 1
    assert FakeTqdm.instances[0].updates == [2, 2, 1]


def test_large_intervals_use_monolithic_final_only_progress(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    engine = _progress_engine(
        progress_update_every=5,
        show_step_progress=True,
        step_progress_update_every=5,
    )

    def fail_chunked(*args, **kwargs):
        del args, kwargs
        raise AssertionError("chunked progress path must not be used")

    monkeypatch.setattr(engine, "_fit_epoch_chunks", fail_chunked)
    monkeypatch.setattr(engine, "_fit_nested_progress", fail_chunked)
    result = engine.fit()

    assert result.final_epoch == 5
    assert len(FakeTqdm.instances) == 1
    assert FakeTqdm.instances[0].updates == [5]
    assert FakeTqdm.instances[0].closed


def test_nested_progress_renders_partial_nan_batch_and_closes(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    engine = _progress_engine(
        show_step_progress=True,
        debug_nans=True,
        loss=lambda split: DebugNaNLoss(split, trigger_batch_value=2.0),
    )

    result = engine.fit()

    assert result.final_epoch == 0
    assert result.nan_debug is not None
    assert len(FakeTqdm.instances) == 2
    outer, inner = FakeTqdm.instances
    assert outer.updates == []
    assert inner.updates == [2, 1]
    assert outer.closed and inner.closed


def test_nested_progress_uses_last_completed_losses_after_nan(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    engine = _progress_engine(
        show_step_progress=True,
        debug_nans=True,
        loss=lambda split: DebugNaNLoss(
            split, trigger_batch_value=2.0, trigger_epoch=3
        ),
    )

    result = engine.fit()

    assert result.final_epoch == 3
    outer, inner = FakeTqdm.instances
    assert outer.updates == [1, 1, 1]
    assert outer.descriptions[-1].startswith("Training loss: 2.000")
    assert inner.updates == [2, 2, 1] * 3 + [2, 1]


def test_nested_progress_supports_batch_manager(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    expected_engine = _progress_engine(show_progress=False)
    expected_engine.batches = BatchManager([expected_engine.batches])
    actual_engine = _progress_engine(show_step_progress=True)
    actual_engine.batches = BatchManager([actual_engine.batches])

    expected = expected_engine.fit()
    actual = actual_engine.fit()

    assert actual.final_epoch == expected.final_epoch
    assert jnp.allclose(actual.history.loss_train, expected.history.loss_train)
    assert jnp.allclose(actual.history.loss_validate, expected.history.loss_validate)
    assert len(FakeTqdm.instances) == 2


def test_epoch_progress_renders_early_stop_remainder(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    engine = OptimEngine(
        loss=_loss(),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=4, patience=2),
        seed=1,
        initial_state={},
        progress_update_every=2,
    )

    result = engine.fit()

    assert result.final_epoch == 3
    assert FakeTqdm.instances[0].updates == [2, 1]


def test_progress_bars_close_when_an_update_raises(monkeypatch):
    class RaisingFakeTqdm(FakeTqdm):
        def update(self, value):
            super().update(value)
            if self.position == 1:
                raise RuntimeError("display failed")

        def close(self):
            super().close()
            raise RuntimeError("close failed")

    RaisingFakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", RaisingFakeTqdm)
    engine = _progress_engine(show_step_progress=True)

    with pytest.raises(RuntimeError, match="display failed"):
        engine.fit()

    assert len(RaisingFakeTqdm.instances) == 2
    assert all(bar.closed for bar in RaisingFakeTqdm.instances)


def test_only_process_zero_constructs_progress_bars(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(jax, "process_index", lambda: 1)

    result = _progress_engine(show_step_progress=True).fit()

    assert result.final_epoch == 5
    assert FakeTqdm.instances == []
