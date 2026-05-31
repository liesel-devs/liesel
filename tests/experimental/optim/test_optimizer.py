from types import SimpleNamespace

import jax.numpy as jnp
import optax
import pytest

from liesel.experimental.optim import LBFGS, Optimizer
from liesel.experimental.optim.types import Position


class QuadraticLoss:
    def grad(self, params, carry):
        del carry
        return {key: 2.0 * value for key, value in params.items()}


def test_optimizer_rejects_empty_position_keys():
    with pytest.raises(ValueError, match="position_keys"):
        Optimizer([], optax.sgd(0.1))


def test_optimizer_rejects_duplicate_position_keys():
    with pytest.raises(ValueError, match="Duplicate position_keys"):
        Optimizer(["x", "x"], optax.sgd(0.1))


def test_optimizer_normalizes_position_keys():
    optimizer = Optimizer(["x"], optax.sgd(0.1))

    assert optimizer.position_keys == ("x",)


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


def test_step_updates_only_owned_position_keys():
    optimizer = Optimizer(["x"], optax.sgd(0.1), identifier="x_opt")
    position = Position({"x": jnp.array(1.0), "y": jnp.array(5.0)})
    carry = SimpleNamespace(
        optimizer_states={"x_opt": optimizer.init(position)},
        position=position,
    )

    carry = optimizer.step(optimizer.position(position), QuadraticLoss(), carry)

    assert carry.position["x"] == pytest.approx(0.8)
    assert carry.position["y"] == pytest.approx(5.0)


def test_lbfgs_uses_compact_optimizer_repr():
    optimizer = LBFGS(["x"], identifier="x_lbfgs")

    assert repr(optimizer) == "LBFGS(('x',), identifier=x_lbfgs)"
