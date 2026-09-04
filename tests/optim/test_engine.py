from __future__ import annotations

import inspect
import io
import math
import threading
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import optax
import pytest

import liesel.optim as opt
import liesel.optim.engine as engine_module
from liesel.optim import (
    LBFGS,
    Batches,
    BatchManager,
    CompositeVDist,
    Elbo,
    EmaTrainLossMonitor,
    LieselOptim,
    LieselVI,
    LossMonitor,
    NegElboLoss,
    OptimEngine,
    Optimizer,
    PositionSplit,
    PositionSplitManager,
    Stopper,
    VDist,
)
from liesel.optim.engine import _progress_print_rate
from liesel.optim.liesel_optim import LieselOptim as LieselOptimFromQuick
from liesel.optim.loss import Loss, LossMixin
from liesel.optim.state import OptimCarry
from liesel.optim.types import Position


def test_ema_train_loss_monitor_required_and_explicit_windows():
    with pytest.raises(TypeError):
        EmaTrainLossMonitor()  # ty: ignore[missing-argument]

    assert EmaTrainLossMonitor(1.0).effective_window == 1.0
    assert EmaTrainLossMonitor(0.25).effective_window == 0.25


def test_ema_train_loss_monitor_is_frozen():
    monitor = EmaTrainLossMonitor(effective_window=1.0)

    with pytest.raises(FrozenInstanceError):
        monitor.effective_window = 2.0  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "effective_window", [True, 0.0, -1.0, float("inf"), float("-inf"), float("nan")]
)
def test_ema_train_loss_monitor_rejects_invalid_windows(effective_window):
    with pytest.raises(ValueError, match="effective_window"):
        EmaTrainLossMonitor(effective_window)


@pytest.mark.parametrize("half_life", [0.25, 1.0, 7.0])
def test_ema_train_loss_monitor_from_half_life(half_life):
    monitor = EmaTrainLossMonitor.from_half_life(half_life)

    assert monitor.effective_window == pytest.approx(2.0 * half_life / math.log(2.0))


@pytest.mark.parametrize(
    "half_life", [True, 0.0, -1.0, float("inf"), float("-inf"), float("nan")]
)
def test_ema_train_loss_monitor_rejects_invalid_half_lives(half_life):
    with pytest.raises(ValueError, match="half_life"):
        EmaTrainLossMonitor.from_half_life(half_life)


def test_loss_monitor_configuration_is_public():
    assert opt.EmaTrainLossMonitor is EmaTrainLossMonitor
    assert opt.LossMonitor is LossMonitor


def test_engine_requires_explicit_loss_monitor():
    with pytest.raises(TypeError, match="loss_monitor"):
        OptimEngine(  # ty: ignore[missing-argument]
            loss=_loss(),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=1, patience=1),
            seed=1,
            initial_state={},
        )


def test_engine_rejects_lbfgs_with_mini_batches():
    with pytest.raises(ValueError, match="LBFGS.*full-data.*deterministic"):
        OptimEngine(
            loss=SequenceLoss(_monitor_split()),
            batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
            optimizers=[LBFGS(["theta"])],
            stopper=Stopper(epochs=1, patience=1),
            seed=1,
            initial_state={},
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        )


def test_engine_accepts_lbfgs_with_full_data_batch():
    result = OptimEngine(
        loss=QuadraticEngineLoss(_split()),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[LBFGS(["theta"])],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor="train_full_data",
    ).fit()

    assert result.history.loss_monitor.tolist() == pytest.approx([0.0], abs=1e-6)


@dataclass
class SequenceOptimizer:
    position_keys: list[str]
    values: jax.Array
    identifier: str = "sequence"
    activate_after_epochs: int = 0

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

    def step(
        self, position: Position, loss, carry: OptimCarry
    ) -> tuple[OptimCarry, jax.Array]:
        value = loss.loss_train_batched(position, carry)
        carry.position = Position(carry.position | {"theta": self.values[carry.epoch]})
        return carry, value


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

    def loss_monitor(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"]

    def grad(self, params: Position, carry: OptimCarry):
        return {key: jnp.zeros_like(value) for key, value in params.items()}

    def value_and_grad(self, params: Position, carry: OptimCarry):
        return self.loss_train_batched(params, carry), self.grad(params, carry)


@dataclass
class QuadraticEngineLoss(SequenceLoss):
    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"] ** 2

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        return self.loss_train_batched(params, carry)


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

    def loss_monitor(self, params: Position, carry: OptimCarry) -> jax.Array:
        del params, carry
        return jnp.array(-999.0)

    def grad(self, params: Position, carry: OptimCarry):
        del carry
        return {key: jnp.zeros_like(value) for key, value in params.items()}

    def value_and_grad(self, params: Position, carry: OptimCarry):
        return self.loss_train_batched(params, carry), self.grad(params, carry)


@dataclass
class BatchedOnlyLoss(BatchSensitiveLoss):
    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        del params, carry
        raise AssertionError("full training loss should not be evaluated")


@dataclass
class DistinctFullDataLoss(BatchSensitiveLoss):
    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["theta"] + 10.0


@dataclass
class DivergentExactLoss(SequenceLoss):
    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        del params
        return -jnp.asarray(carry.epoch, dtype=float)


@dataclass
class EpochSequenceLoss(SequenceLoss):
    epoch_losses: jax.Array

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        del params
        return self.epoch_losses[carry.epoch]


@dataclass
class UnitGradientLoss(LossMixin):
    split: PositionSplit

    def position(self, position_keys) -> Position:
        return Position({key: jnp.array(0.0) for key in position_keys})

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return sum((jnp.sum(value) for value in params.values()), start=jnp.array(0.0))

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        return self.loss_train_batched(params, carry)

    def loss_monitor(self, params: Position, carry: OptimCarry) -> jax.Array:
        return self.loss_train_batched(params, carry)

    def grad(self, params: Position, carry: OptimCarry):
        del carry
        return {key: jnp.ones_like(value) for key, value in params.items()}


@dataclass
class RandomGradientLoss(UnitGradientLoss):
    def grad(self, params: Position, carry: OptimCarry):
        return {
            key: jax.random.normal(carry.key, shape=value.shape)
            for key, value in params.items()
        }


@dataclass
class DebugNoOpOptimizer:
    position_keys: list[str]
    identifier: str = "noop"
    activate_after_epochs: int = 0

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

    def step(
        self, position: Position, loss, carry: OptimCarry
    ) -> tuple[OptimCarry, jax.Array]:
        return carry, loss.loss_train_batched(position, carry)


@dataclass
class AddOneOptimizer(DebugNoOpOptimizer):
    def step(
        self, position: Position, loss, carry: OptimCarry
    ) -> tuple[OptimCarry, jax.Array]:
        value = loss.loss_train_batched(position, carry)
        key = self.position_keys[0]
        carry.position = Position(carry.position | {key: position[key] + 1.0})
        return carry, value


@dataclass
class ConstantLossOptimizer(DebugNoOpOptimizer):
    returned_loss: float = 0.0

    def step(
        self, position: Position, loss, carry: OptimCarry
    ) -> tuple[OptimCarry, jax.Array]:
        del loss
        key = self.position_keys[0]
        carry.position = Position(carry.position | {key: position[key] + 1.0})
        return carry, jnp.asarray(self.returned_loss)


@dataclass
class NanOptimizer(DebugNoOpOptimizer):
    def step(
        self, position: Position, loss, carry: OptimCarry
    ) -> tuple[OptimCarry, jax.Array]:
        value = loss.loss_train_batched(position, carry)
        key = self.position_keys[0]
        carry.position = Position(carry.position | {key: position[key] * jnp.nan})
        return carry, value


@dataclass
class NanLossAndPositionOptimizer(NanOptimizer):
    def step(
        self, position: Position, loss, carry: OptimCarry
    ) -> tuple[OptimCarry, jax.Array]:
        carry, _ = super().step(position, loss, carry)
        return carry, jnp.asarray(jnp.nan)


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
        param_sum = sum(
            (jnp.sum(value) for value in params.values()), start=jnp.array(0.0)
        )
        batch_sum = sum(
            (jnp.sum(value) for value in carry.batch.values()), start=jnp.array(0.0)
        )
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
        return sum((jnp.sum(value) for value in params.values()), start=jnp.array(0.0))

    def loss_monitor(self, params: Position, carry: OptimCarry) -> jax.Array:
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


def _monitor_split_with_validation() -> PositionSplit:
    return PositionSplit(
        train=Position({"y": jnp.array([1.0, 3.0])}),
        validate=Position({"y": jnp.array([5.0])}),
        test=Position({}),
        train_axis_size=2,
        validate_axis_size=1,
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
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
    instances: ClassVar[list[FakeTqdm]] = []

    def __init__(self, total, desc, leave, position=None, ncols=None, bar_format=None):
        self.total = total
        self.desc = desc
        self.position = position
        self.leave = leave
        self.ncols = ncols
        self.bar_format = bar_format
        self.n = 0
        self.updates = []
        self.descriptions = [desc]
        self.thread_ids = [threading.get_ident()]
        self.refreshes = 0
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

    def set_description_str(self, desc, refresh=True):
        self.set_description(desc, refresh)

    def reset(self, total=None):
        self.thread_ids.append(threading.get_ident())
        self.n = 0
        if total is not None:
            self.total = total

    def refresh(self):
        self.thread_ids.append(threading.get_ident())
        self.refreshes += 1

    def close(self):
        self.thread_ids.append(threading.get_ident())
        self.closed = True


@pytest.mark.parametrize("save_position_history", [True, False])
def test_ema_result_recommends_terminal_and_retains_minimum_monitor_position(
    save_position_history,
):
    loss = _loss()
    optimizer = _optimizer()
    engine = OptimEngine(
        loss=loss,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[optimizer],
        stopper=Stopper(epochs=4, patience=2),
        seed=1,
        initial_state={},
        save_position_history=save_position_history,
        show_progress=False,
    )

    result = engine.fit()

    assert result.history.loss_monitor.tolist() == [-1.0, 0.0, 5.0]
    assert result.n_epochs == 3
    assert result.patience == 2
    assert result.monitor_source == "train_ema"
    assert result.min_monitor_epoch == 0
    position = result.position
    position_min_monitor = result.position_min_monitor
    assert position is not None
    assert position_min_monitor is not None
    assert position["theta"] == pytest.approx(6.0)
    assert result.position_final["theta"] == pytest.approx(6.0)
    assert position_min_monitor["theta"] == pytest.approx(0.0)


def test_removed_result_and_engine_api_is_absent():
    removed_engine_argument = "restore_" + "best_" + "position"
    assert removed_engine_argument not in inspect.signature(OptimEngine).parameters

    result = OptimEngine(
        loss=_loss(),
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
    ).fit()
    for removed_field in (
        "best_" + "position",
        "best_" + "epoch",
        "final_" + "epoch",
    ):
        assert not hasattr(result, removed_field)


def test_engine_uses_loss_split():
    loss = _loss()
    engine = OptimEngine(
        loss=loss,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


def test_optimizer_activation_delay_must_allow_an_active_epoch():
    optimizer = Optimizer(["theta"], optax.sgd(0.1), activate_after_epochs=4)

    with pytest.raises(ValueError, match="activate_after_epochs"):
        OptimEngine(
            loss=_loss(),
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[optimizer],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )


@pytest.mark.parametrize("debug_nans", [False, True])
def test_optimizer_activates_after_completed_epoch_delay(debug_nans):
    loss = UnitGradientLoss(_split())

    def delayed_learning_rate(count):
        return jnp.where(count == 0, 1.0, 10.0)

    engine = OptimEngine(
        loss=loss,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[
            Optimizer(["theta"], optax.sgd(1.0), identifier="theta"),
            Optimizer(
                ["eta"],
                optax.sgd(delayed_learning_rate),
                identifier="eta",
                activate_after_epochs=2,
            ),
        ],
        stopper=Stopper(epochs=4, patience=4),
        seed=1,
        initial_state={},
        show_progress=False,
        debug_nans=debug_nans,
    )

    result = engine.fit()
    assert result.history.position is not None

    assert {
        key: values.tolist() for key, values in result.history.position.items()
    } == {
        "theta": [-1.0, -2.0, -3.0, -4.0],
        "eta": [0.0, 0.0, -1.0, -11.0],
    }


def test_inactive_optimizer_does_not_consume_random_key():
    def first_theta_position(optimizers):
        engine = OptimEngine(
            loss=RandomGradientLoss(_split()),
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=optimizers,
            stopper=Stopper(epochs=2, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
        )
        result = engine.fit()
        assert result.history.position is not None
        return result.history.position["theta"][0]

    theta = Optimizer(["theta"], optax.sgd(1.0), identifier="theta")
    delayed_eta = Optimizer(
        ["eta"],
        optax.sgd(1.0),
        identifier="eta",
        activate_after_epochs=1,
    )

    expected = first_theta_position([theta])
    actual = first_theta_position([delayed_eta, theta])

    assert jnp.array_equal(actual, expected)


def test_fit_can_stop_before_any_optimizer_activates():
    engine = OptimEngine(
        loss=UnitGradientLoss(_split()),
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[Optimizer(["theta"], optax.sgd(1.0), activate_after_epochs=3)],
        stopper=Stopper(epochs=5, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
    )

    result = engine.fit()
    assert result.history.position is not None

    assert (result.n_epochs, result.history.position["theta"].tolist()) == (
        2,
        [0.0, 0.0],
    )


def test_duplicate_optimizer_position_keys_raise():
    with pytest.raises(ValueError, match="Position keys"):
        OptimEngine(
            loss=_loss(),
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            **kwargs,
        )


def test_invalid_loss_monitor_raises():
    with pytest.raises(ValueError, match="loss_monitor"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            loss_monitor="sometimes",  # ty: ignore[invalid-argument-type]
        )


def test_validation_monitor_requires_validation_data():
    with pytest.raises(ValueError, match="validation"):
        OptimEngine(
            loss=_loss(),
            batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
            optimizers=[_optimizer()],
            stopper=Stopper(epochs=4, patience=2),
            seed=1,
            initial_state={},
            show_progress=False,
            loss_monitor="validation",
        )


def test_debug_nans_no_active_loss_capture_reproduces_loss():
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
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=Batches(["y"], axis_size=3, batch_size=1, shuffle=False),
        optimizers=[DebugNoOpOptimizer(["theta"], activate_after_epochs=1)],
        stopper=Stopper(epochs=3, patience=3),
        seed=1,
        initial_state={},
        show_progress=False,
        debug_nans=True,
    )

    result = engine.fit()

    info = result.nan_debug
    assert info is not None
    assert result.n_epochs == 0
    assert result.position is None
    assert result.position_min_monitor is None
    assert result.min_monitor_epoch is None
    assert result.position_final["theta"] == pytest.approx(0.0)
    assert info.kind == "loss"
    assert info.batch == 1
    assert info.optimizer_index is None
    assert info.nan_position is None
    assert info.loss is not None
    assert info.obs_batch["y"].tolist() == pytest.approx([1.0])
    assert info.last_non_nan_position["theta"] == pytest.approx(0.0)
    assert bool(jnp.isnan(info.loss))
    assert bool(jnp.isnan(info.reproduce_loss(engine)))
    assert info.reproduction_carry.batch["y"].tolist() == pytest.approx([1.0])
    assert info.reproduction_carry.fixed_position == {}


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
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
    assert result.n_epochs == 0
    assert result.position is None
    assert result.position_min_monitor is None
    assert result.min_monitor_epoch is None
    assert result.position_final["theta"] == pytest.approx(1.0)
    assert bool(jnp.isnan(result.position_final["eta"]))
    assert info.kind == "position_after"
    assert info.batch == 0
    assert info.optimizer_index == 1
    assert info.optimizer_identifier == "nan_eta"
    assert info.optimizer_position_keys == ("eta",)
    assert info.nan_position is not None
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


def test_debug_nans_later_optimizer_loss_takes_precedence_over_nan_update():
    split = _split()
    engine = OptimEngine(
        loss=DebugNaNLoss(split),
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=Batches(["y"], axis_size=1, batch_size=1, shuffle=False),
        optimizers=[
            AddOneOptimizer(["theta"], identifier="add_theta"),
            NanLossAndPositionOptimizer(["eta"], identifier="nan_eta"),
        ],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        debug_nans=True,
    )

    info = engine.fit().nan_debug

    assert info is not None
    assert info.kind == "loss"
    assert info.optimizer_index == 1
    assert info.optimizer_identifier == "nan_eta"
    assert info.reproduction_position["theta"] == pytest.approx(1.0)
    assert info.reproduction_position["eta"] == pytest.approx(0.0)
    assert bool(jnp.isnan(info.reproduce_loss(engine)))


def test_debug_nans_position_before_capture():
    split = _split()
    engine = OptimEngine(
        loss=DebugNaNLoss(split, initial_nan=True),
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
    assert result.n_epochs == 0
    assert info.kind == "position_before"
    assert info.optimizer_index is None
    assert info.nan_position is not None
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
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
    assert result.n_epochs == 1
    assert bool(jnp.isnan(result.history.loss_train[0]))


@pytest.mark.parametrize("debug_nans", [False, True])
def test_training_history_and_ema_use_pre_update_losses_across_epochs(debug_nans):
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=2, patience=2),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        debug_nans=debug_nans,
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([2.0, 4.5])
    assert result.history.loss_monitor.tolist() == pytest.approx([2.5, 5.875])


def test_train_full_data_monitor_uses_exact_training_loss():
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor="train_full_data",
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([2.0])
    assert result.history.loss_monitor.tolist() == pytest.approx([4.0])


def test_ema_fractional_window_uses_one_step_lower_bound():
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor=EmaTrainLossMonitor(effective_window=0.25),
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([2.0])
    assert result.history.loss_monitor.tolist() == pytest.approx([3.0])


def test_training_history_and_ema_use_first_active_optimizer_loss():
    split = _split()
    engine = OptimEngine(
        loss=UnitGradientLoss(split),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[
            ConstantLossOptimizer(["theta"], identifier="theta", returned_loss=10.0),
            ConstantLossOptimizer(["eta"], identifier="eta", returned_loss=99.0),
        ],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([10.0])
    assert result.history.loss_monitor.tolist() == pytest.approx([10.0])
    assert result.position_final == pytest.approx({"theta": 1.0, "eta": 1.0})


def test_training_history_uses_fallback_then_first_active_loss():
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[
            ConstantLossOptimizer(
                ["theta"],
                identifier="theta",
                activate_after_epochs=2,
                returned_loss=10.0,
            ),
            ConstantLossOptimizer(
                ["eta"],
                identifier="eta",
                activate_after_epochs=1,
                returned_loss=99.0,
            ),
        ],
        stopper=Stopper(epochs=3, patience=3),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([2.0, 99.0, 10.0])
    assert result.history.position is not None
    assert result.history.position["theta"].tolist() == pytest.approx([0.0, 0.0, 2.0])
    assert result.history.position["eta"].tolist() == pytest.approx([0.0, 2.0, 4.0])


def test_nan_from_later_optimizer_stops_ordinary_fit():
    engine = OptimEngine(
        loss=UnitGradientLoss(_split()),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[
            ConstantLossOptimizer(["theta"], identifier="theta", returned_loss=10.0),
            ConstantLossOptimizer(
                ["eta"], identifier="eta", returned_loss=float("nan")
            ),
        ],
        stopper=Stopper(epochs=3, patience=3),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
    )

    result = engine.fit()

    assert result.n_epochs == 1
    assert bool(jnp.isnan(result.history.loss_train[0]))
    assert bool(jnp.isnan(result.history.loss_monitor[0]))


def test_nan_updated_position_stops_ordinary_fit():
    engine = OptimEngine(
        loss=UnitGradientLoss(_split()),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[
            AddOneOptimizer(["theta"], identifier="theta"),
            NanOptimizer(["eta"], identifier="eta"),
        ],
        stopper=Stopper(epochs=3, patience=3),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
    )

    result = engine.fit()

    assert result.n_epochs == 1
    assert bool(jnp.isnan(result.history.loss_train[0]))
    assert bool(jnp.isnan(result.history.loss_monitor[0]))
    assert result.position_final["theta"] == pytest.approx(1.0)
    assert bool(jnp.isnan(result.position_final["eta"]))


def test_ema_monitor_adds_no_full_data_evaluation():
    split = _monitor_split()
    engine = OptimEngine(
        loss=BatchedOnlyLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
    )

    result = engine.fit()

    assert result.history.loss_monitor.tolist() == pytest.approx([2.5])


def test_validation_monitor_is_unsmoothed():
    split = _monitor_split_with_validation()
    engine = OptimEngine(
        loss=BatchSensitiveLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor="validation",
    )

    result = engine.fit()

    assert result.history.loss_train.tolist() == pytest.approx([2.0])
    assert result.history.loss_monitor.tolist() == pytest.approx([-999.0])


def test_train_full_data_monitor_uses_exact_callback_with_full_data_batch():
    split = _monitor_split()
    engine = OptimEngine(
        loss=DistinctFullDataLoss(split),
        batches=Batches(["y"], axis_size=2, batch_size=None, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=1, patience=1),
        seed=1,
        initial_state={},
        show_progress=False,
        loss_monitor="train_full_data",
    )

    result = engine.fit()

    assert result.history.loss_monitor.tolist() == pytest.approx([10.0])


@pytest.mark.parametrize("loss_monitor", ["validation", "train_full_data"])
def test_exact_monitor_source_drives_epoch_stopping(loss_monitor):
    split = _monitor_split_with_validation()
    engine = OptimEngine(
        loss=DivergentExactLoss(split),
        loss_monitor=loss_monitor,
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=4, patience=2),
        seed=1,
        initial_state={},
        save_position_history=False,
        show_progress=False,
    )

    result = engine.fit()

    assert result.n_epochs == 3
    assert result.history.loss_train.tolist() == pytest.approx([0.0, -1.0, -2.0])
    assert result.history.loss_monitor.tolist() == pytest.approx([0.0, 5.0, 6.0])
    assert result.monitor_source == loss_monitor
    assert result.min_monitor_epoch == 0
    position = result.position
    position_min_monitor = result.position_min_monitor
    assert position is not None
    assert position_min_monitor is not None
    assert position["theta"] == pytest.approx(0.0)
    assert position_min_monitor["theta"] == pytest.approx(0.0)
    assert result.position_final["theta"] == pytest.approx(6.0)
    assert result.history.position is None


def test_ema_monitor_source_drives_epoch_stopping():
    split = _monitor_split()
    engine = OptimEngine(
        loss=EpochSequenceLoss(split, jnp.array([100.0, 0.0, 1.0, 2.0])),
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=Batches(["y"], axis_size=2, batch_size=1, shuffle=False),
        optimizers=[DebugNoOpOptimizer(["theta"])],
        stopper=Stopper(epochs=4, patience=2),
        seed=1,
        initial_state={},
        show_progress=False,
    )

    result = engine.fit()

    assert result.n_epochs == 4
    assert result.history.loss_train.tolist() == pytest.approx([100.0, 0.0, 1.0, 2.0])
    assert result.history.loss_monitor[:3].tolist() == pytest.approx(
        [100.0, 10.0, 1.9890109]
    )


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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
            loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
    assert opt.LieselVI is LieselVI
    assert opt.NegElboLoss is NegElboLoss
    assert opt.Elbo is Elbo
    assert opt.VDist is VDist
    assert opt.CompositeVDist is CompositeVDist
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
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
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
        False,
        True,
        3,
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
    )

    assert engine.progress_update_every == 4
    assert engine.progress_n_updates == 3


def test_nested_progress_matches_monolithic_and_never_uses_callback(monkeypatch):
    expected = _progress_engine(show_progress=False).fit()
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(engine_module.sys.stderr, "isatty", lambda: True)

    def fail_callback(*args, **kwargs):
        del args, kwargs
        raise AssertionError("jax.debug.callback must not be used for progress")

    monkeypatch.setattr(jax.debug, "callback", fail_callback)
    actual_engine = _progress_engine(show_step_progress=True)
    actual = actual_engine.fit()

    assert actual.n_epochs == expected.n_epochs == 5
    assert jnp.allclose(actual.history.loss_train, expected.history.loss_train)
    assert jnp.allclose(actual.history.loss_monitor, expected.history.loss_monitor)

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


def test_non_tty_progress_uses_one_fixed_width_bar(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(engine_module.sys.stderr, "isatty", lambda: False)

    result = _progress_engine(show_step_progress=True).fit()

    assert result.n_epochs == 5
    assert len(FakeTqdm.instances) == 1
    progress_bar = FakeTqdm.instances[0]
    assert progress_bar.position is None
    assert progress_bar.ncols == 88
    assert progress_bar.bar_format == "{l_bar}{bar}| [{elapsed}, {rate_fmt}]"
    assert progress_bar.total == 25
    assert progress_bar.updates == [2, 2, 1] * 5
    assert progress_bar.refreshes == 5
    assert progress_bar.descriptions[-1].endswith("[E 5/5, B 5/5]")
    assert all(desc.startswith("Train=") for desc in progress_bar.descriptions)
    assert all("Monitor=" in desc for desc in progress_bar.descriptions)
    assert progress_bar.closed


def test_step_progress_refreshes_before_finishing_epoch(monkeypatch):
    events = []

    class RecordingTqdm(FakeTqdm):
        def refresh(self):
            events.append("refresh")
            super().refresh()

    engine = _progress_engine(epochs=1, show_step_progress=True)
    finish_epoch = engine._finish_epoch

    def record_finish(carry):
        events.append("finish")
        return finish_epoch(carry)

    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", RecordingTqdm)
    monkeypatch.setattr(engine_module.sys.stderr, "isatty", lambda: False)
    monkeypatch.setattr(engine_module.jax, "jit", lambda fn: fn)
    monkeypatch.setattr(engine, "_finish_epoch", record_finish)

    engine.fit()

    assert events == ["refresh", "finish"]


def test_shared_progress_description_uses_fixed_width_counts():
    description = OptimEngine._shared_progress_description(
        epoch=1,
        max_epochs=10,
        batch=587,
        n_batches=781,
        loss_train=1.25,
        loss_monitor=2.5,
    )

    assert description == ("Train=1.250, Monitor=2.500 [E  1/10, B 587/781]")


def test_non_tty_progress_renders_without_ansi_cursor_movement(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(engine_module.sys, "stderr", stream)

    result = _progress_engine(epochs=1, show_step_progress=True).fit()
    output = stream.getvalue()

    assert result.n_epochs == 1
    assert "Train=" in output
    assert "Monitor=" in output
    assert "E 1/1, B 5/5" in output
    assert "%|" in output
    assert "it/s" in output
    assert "| 5/5 [" not in output
    assert "\x1b[A" not in output
    assert all(
        not line.rstrip().endswith(":")
        for line in output.replace("\n", "\r").split("\r")
    )
    assert all(len(line) <= 88 for line in output.replace("\n", "\r").split("\r"))


def test_large_step_interval_uses_epoch_only_progress(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    engine = _progress_engine(
        show_step_progress=True,
        step_progress_update_every=5,
    )

    result = engine.fit()

    assert result.n_epochs == 5
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

    assert result.n_epochs == 5
    assert len(FakeTqdm.instances) == 1
    assert FakeTqdm.instances[0].updates == [5]
    assert FakeTqdm.instances[0].closed


def test_nested_progress_renders_partial_nan_batch_and_closes(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(engine_module.sys.stderr, "isatty", lambda: True)
    engine = _progress_engine(
        show_step_progress=True,
        debug_nans=True,
        loss=lambda split: DebugNaNLoss(split, trigger_batch_value=2.0),
    )

    result = engine.fit()

    assert result.n_epochs == 0
    assert result.nan_debug is not None
    assert len(FakeTqdm.instances) == 2
    outer, inner = FakeTqdm.instances
    assert outer.updates == []
    assert inner.updates == [2, 1]
    assert outer.closed and inner.closed


def test_nested_progress_uses_last_completed_losses_after_nan(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(engine_module.sys.stderr, "isatty", lambda: True)
    engine = _progress_engine(
        show_step_progress=True,
        debug_nans=True,
        loss=lambda split: DebugNaNLoss(
            split, trigger_batch_value=2.0, trigger_epoch=3
        ),
    )

    result = engine.fit()

    assert result.n_epochs == 3
    outer, inner = FakeTqdm.instances
    assert outer.updates == [1, 1, 1]
    assert outer.descriptions[-1] == "Train=2.000, Monitor=2.758"
    assert inner.updates == [2, 2, 1] * 3 + [2, 1]


def test_nested_progress_supports_batch_manager(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(engine_module.sys.stderr, "isatty", lambda: True)
    expected_engine = _progress_engine(show_progress=False)
    expected_batches = expected_engine.batches
    assert isinstance(expected_batches, Batches)
    expected_engine.batches = BatchManager([expected_batches])
    actual_engine = _progress_engine(show_step_progress=True)
    actual_batches = actual_engine.batches
    assert isinstance(actual_batches, Batches)
    actual_engine.batches = BatchManager([actual_batches])

    expected = expected_engine.fit()
    actual = actual_engine.fit()

    assert actual.n_epochs == expected.n_epochs
    assert jnp.allclose(actual.history.loss_train, expected.history.loss_train)
    assert jnp.allclose(actual.history.loss_monitor, expected.history.loss_monitor)
    assert len(FakeTqdm.instances) == 2


def test_epoch_progress_renders_early_stop_remainder(monkeypatch):
    FakeTqdm.instances = []
    monkeypatch.setattr(engine_module, "tqdm", FakeTqdm)
    engine = OptimEngine(
        loss=_loss(),
        loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
        batches=Batches(["y"], axis_size=1, batch_size=None, shuffle=False),
        optimizers=[_optimizer()],
        stopper=Stopper(epochs=4, patience=2),
        seed=1,
        initial_state={},
        progress_update_every=2,
    )

    result = engine.fit()

    assert result.n_epochs == 3
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
    monkeypatch.setattr(engine_module.sys.stderr, "isatty", lambda: True)
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

    assert result.n_epochs == 5
    assert FakeTqdm.instances == []
