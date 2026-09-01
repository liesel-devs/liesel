import jax
import jax.numpy as jnp
import optax
import pytest

from liesel.optim import LBFGS, Batches, Optimizer, PositionSplit
from liesel.optim.loss import LossMixin
from liesel.optim.state import OptimCarry
from liesel.optim.types import Position


class QuadraticLoss(LossMixin):
    split = PositionSplit(
        train=Position({"y": jnp.array([0.0])}),
        validate=Position({}),
        test=Position({}),
        train_axis_size=1,
        validate_axis_size=0,
        test_axis_size=0,
    )

    def position(self, position_keys) -> Position:
        return Position({key: jnp.array(0.0) for key in position_keys})

    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return params["x"] ** 2

    def loss_train(self, params: Position, carry: OptimCarry) -> jax.Array:
        return self.loss_train_batched(params, carry)

    def loss_monitor(self, params: Position, carry: OptimCarry) -> jax.Array:
        return self.loss_train_batched(params, carry)


class DifferentObjectiveDrawLoss(QuadraticLoss):
    def loss_train_batched(self, params: Position, carry: OptimCarry) -> jax.Array:
        del carry
        return 3.0 * params["x"] ** 2

    def grad(self, params: Position, carry: OptimCarry) -> Position:
        del carry
        return Position({"x": 6.0 * params["x"]})

    def value_and_grad(
        self, params: Position, carry: OptimCarry
    ) -> tuple[jax.Array, Position]:
        del carry
        return 2.0 * params["x"] ** 2, Position({"x": 4.0 * params["x"]})


def test_optimizer_rejects_empty_position_keys():
    with pytest.raises(ValueError, match="position_keys"):
        Optimizer([], optax.sgd(0.1))


def test_optimizer_rejects_duplicate_position_keys():
    with pytest.raises(ValueError, match="Duplicate position_keys"):
        Optimizer(["x", "x"], optax.sgd(0.1))


def test_optimizer_normalizes_position_keys():
    optimizer = Optimizer(["x"], optax.sgd(0.1))

    assert optimizer.position_keys == ("x",)


def test_optimizer_activation_delay_defaults_to_zero():
    optimizer = Optimizer(["x"], optax.sgd(0.1))

    assert optimizer.activate_after_epochs == 0


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_optimizer_rejects_invalid_activation_delay(value):
    with pytest.raises(ValueError, match="activate_after_epochs"):
        Optimizer(["x"], optax.sgd(0.1), activate_after_epochs=value)


def test_position_requires_all_claimed_keys():
    optimizer = Optimizer(["x", "missing"], optax.sgd(0.1))

    with pytest.raises(KeyError, match="missing"):
        optimizer.position(Position({"x": jnp.array(1.0)}))


def test_position_and_not_position_return_expected_subsets():
    optimizer = Optimizer(["x"], optax.sgd(0.1))
    position = Position({"x": jnp.array(1.0), "y": jnp.array(2.0)})

    owned = optimizer.position(position)
    fixed = optimizer.not_position(position)

    assert set(owned) == {"x"}
    assert set(fixed) == {"y"}
    assert owned["x"] == pytest.approx(1.0)
    assert fixed["y"] == pytest.approx(2.0)


def test_step_returns_pre_update_loss_and_updates_only_owned_position_keys():
    optimizer = Optimizer(["x"], optax.sgd(0.1), identifier="x_opt")
    position = Position({"x": jnp.array(1.0), "y": jnp.array(5.0)})
    carry = OptimCarry.new(
        key=jax.random.key(0),
        epochs=1,
        position=position,
        tracked=None,
        batches=Batches([], axis_size=1, batch_size=None),
        optimizers=[optimizer],
        model_state={},
        save_position_history=False,
    )

    carry, loss = optimizer.step(
        optimizer.position(position), DifferentObjectiveDrawLoss(), carry
    )

    assert loss == pytest.approx(2.0)
    assert carry.position["x"] == pytest.approx(0.6)
    assert carry.position["y"] == pytest.approx(5.0)


def test_lbfgs_step_returns_pre_update_loss():
    optimizer = LBFGS(["x"], identifier="x_lbfgs")
    position = Position({"x": jnp.array(1.0)})
    carry = OptimCarry.new(
        key=jax.random.key(0),
        epochs=1,
        position=position,
        tracked=None,
        batches=Batches([], axis_size=1, batch_size=None),
        optimizers=[optimizer],
        model_state={},
        save_position_history=False,
    )

    _, loss = optimizer.step(optimizer.position(position), QuadraticLoss(), carry)

    assert loss == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("delay", "expected"),
    [
        (0, "LBFGS(('x',), identifier=x_lbfgs)"),
        (3, "LBFGS(('x',), identifier=x_lbfgs, activate_after_epochs=3)"),
    ],
)
def test_lbfgs_uses_compact_optimizer_repr(delay, expected):
    optimizer = LBFGS(["x"], identifier="x_lbfgs", activate_after_epochs=delay)

    assert repr(optimizer) == expected


def test_optimizer_repr_shows_nonzero_activation_delay():
    optimizer = Optimizer(
        ["x"], optax.sgd(0.1), identifier="x_opt", activate_after_epochs=3
    )

    assert repr(optimizer) == (
        "Optimizer(('x',), identifier=x_opt, activate_after_epochs=3)"
    )


def test_optimizer_pytree_round_trip_preserves_activation_delay():
    optimizer = Optimizer(["x"], optax.sgd(0.1), activate_after_epochs=3)

    rebuilt = jax.tree.map(lambda value: value, optimizer)

    assert rebuilt.activate_after_epochs == 3
