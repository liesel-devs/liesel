"""Loss state carried through optimization."""

import jax
import jax.numpy as jnp
import optax
import pytest

from liesel.optim import (
    LBFGS,
    Batches,
    EmaTrainLossMonitor,
    OptimEngine,
    Optimizer,
    PositionSplit,
    Stopper,
)
from liesel.optim.loss import LossMixin
from liesel.optim.types import Position

N_BATCHES = 4


class CountingLoss(LossMixin):
    """Counts its training evaluations in the loss state."""

    def __init__(self, nan_at: int | None = None):
        self.split = PositionSplit(
            Position({"y": jnp.arange(8.0)}),
            Position({"y": jnp.arange(4.0)}),
            Position({}),
            8,
            4,
            0,
        )
        self.nan_at = nan_at

    def init_loss_state(self):
        return jnp.zeros((), dtype=jnp.int32)

    def position(self, position_keys):
        return Position({key: jnp.array(0.0) for key in position_keys})

    def _value(self, params, y, carry):
        value = sum((x - jnp.mean(y)) ** 2 for x in params.values())
        value = value + sum((x - 1.0) ** 2 for x in carry.fixed_position.values())
        if self.nan_at is not None:
            value = jnp.where(carry.loss_state == self.nan_at, jnp.nan, value)
        return value

    def loss_train_batched_and_state(self, params, carry):
        return self._value(params, carry.batch["y"], carry), carry.loss_state + 1

    def loss_train_batched(self, params, carry):
        return self._value(params, carry.batch["y"], carry)

    def loss_train(self, params, carry):
        return self._value(params, self.split.train["y"], carry)

    def loss_monitor(self, params, carry):
        return self._value(params, self.split.validate["y"], carry)


def make_engine(
    epochs=5,
    loss=None,
    optimizers=None,
    loss_monitor="validation",
    batch_size=2,
    **kwargs,
):
    return OptimEngine(
        loss=CountingLoss() if loss is None else loss,
        batches=Batches(["y"], axis_size=8, batch_size=batch_size),
        optimizers=optimizers or [Optimizer(["theta"], optax.adam(0.05))],
        stopper=Stopper(epochs=epochs, patience=epochs, min_epochs=epochs),
        seed=1,
        initial_state={},
        loss_monitor=loss_monitor,
        show_progress=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "loss_monitor", [EmaTrainLossMonitor(1.0), "validation", "train_full_data"]
)
def test_final_loss_state_counts_training_evaluations(loss_monitor):
    result = make_engine(loss_monitor=loss_monitor).fit()

    assert int(result.loss_state_final) == 5 * N_BATCHES


def test_each_optimizer_stores_its_loss_state():
    optimizers = [
        Optimizer(["theta"], optax.adam(0.05)),
        Optimizer(["phi"], optax.adam(0.05)),
    ]
    result = make_engine(optimizers=optimizers).fit()

    assert int(result.loss_state_final) == 2 * 5 * N_BATCHES


def test_batch_loss_without_active_optimizer_keeps_no_loss_state():
    optimizers = [Optimizer(["theta"], optax.adam(0.05), activate_after_epochs=2)]
    result = make_engine(optimizers=optimizers).fit()

    assert int(result.loss_state_final) == 3 * N_BATCHES


def test_min_monitor_loss_state_matches_min_monitor_epoch():
    # Fitting moves theta away from the validation mean, so the best epoch is early.
    optimizers = [Optimizer(["theta"], optax.adam(0.3))]
    result = make_engine(epochs=8, optimizers=optimizers).fit()

    assert result.min_monitor_epoch is not None
    assert result.min_monitor_epoch < 7
    expected = (result.min_monitor_epoch + 1) * N_BATCHES
    assert int(result.loss_state_min_monitor) == expected


def test_resumed_fit_continues_loss_state(tmp_path):
    path = tmp_path / "optim.pkl"
    first = make_engine().fit(checkpoint=path, pause_after=2)
    assert int(first.loss_state_final) == 2 * N_BATCHES

    resumed = make_engine().fit(checkpoint=path)

    assert int(resumed.loss_state_final) == 5 * N_BATCHES
    assert resumed.position_final == make_engine().fit().position_final


def test_checkpoint_with_incompatible_loss_state_is_rejected():
    class FloatCountingLoss(CountingLoss):
        def init_loss_state(self):
            return jnp.zeros(())

    checkpoint = make_engine().fit(pause_after=2).checkpoint

    with pytest.raises(ValueError, match="loss state"):
        make_engine(loss=FloatCountingLoss()).fit(checkpoint=checkpoint)


def test_nan_debug_keeps_loss_state_of_failing_step():
    result = make_engine(loss=CountingLoss(nan_at=6), debug_nans=True).fit()

    assert result.status == "nan"
    assert result.nan_debug is not None
    assert int(result.nan_debug.reproduction_carry.loss_state) == 6
    engine = make_engine(loss=CountingLoss(nan_at=6), debug_nans=True)
    assert jnp.isnan(result.nan_debug.reproduce_loss(engine))


def test_lbfgs_rejects_loss_with_state():
    engine = make_engine(optimizers=[LBFGS(["theta"])], batch_size=8)

    with pytest.raises(ValueError, match="loss with state"):
        engine.fit()


class QuadraticLoss(LossMixin):
    def __init__(self):
        self.split = PositionSplit(
            Position({"y": jnp.arange(8.0)}), Position({}), Position({}), 8, 0, 0
        )

    def position(self, position_keys):
        return Position({key: jnp.array(0.0) for key in position_keys})

    def loss_train_batched(self, params, carry):
        return (params["theta"] - jnp.mean(carry.batch["y"])) ** 2


def test_loss_without_state_has_empty_loss_state():
    loss = QuadraticLoss()
    engine = make_engine(loss=loss, loss_monitor=EmaTrainLossMonitor(1.0))
    result = engine.fit()

    assert loss.init_loss_state() == ()
    assert result.loss_state_final == ()
    assert result.loss_state_min_monitor == ()


def test_mixin_value_and_grad_and_state_matches_value_and_grad():
    loss = QuadraticLoss()
    engine = make_engine(loss=loss, loss_monitor=EmaTrainLossMonitor(1.0))
    carry = engine._init_carry(1)
    carry.batch = Position({"y": jnp.arange(2.0)})
    params = Position({"theta": jnp.array(2.0)})

    value, grad, state = loss.value_and_grad_and_state(params, carry)
    expected_value, expected_grad = loss.value_and_grad(params, carry)

    assert value == expected_value
    assert grad == expected_grad
    assert state == ()


def test_loss_state_leaves_are_traced_through_jit():
    loss = CountingLoss()
    carry = make_engine(loss=loss)._init_carry(1)
    carry.batch = Position({"y": jnp.arange(2.0)})
    params = Position({"theta": jnp.array(2.0)})

    _, _, state = jax.jit(loss.value_and_grad_and_state)(params, carry)

    assert int(state) == 1
