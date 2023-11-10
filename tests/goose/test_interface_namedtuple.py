import jax
import jax.numpy as jnp
import pytest

from typing import NamedTuple
from liesel.goose.interface import NamedTupleInterface
from liesel.goose.types import ModelInterface, Position


class State(NamedTuple):
    x: jax.Array
    y: int

def log_prob(state: State) -> float:
    x = state.x
    return -jnp.sum(x**2)


def create_state() -> State:
    return State(x = jnp.array([-1.0, 0.0, 1.0]), y = 1)


def test_log_prob() -> None:
    conn: ModelInterface = NamedTupleInterface(log_prob)
    state = create_state()
    lp = float(conn.log_prob(state))
    assert lp == pytest.approx(-2.0)


def test_extract() -> None:
    conn = NamedTupleInterface(log_prob)
    state = create_state()

    pos1 = conn.extract_position(["x"], state)
    assert pos1["x"] == pytest.approx(state.x)
    assert len(pos1) == 1

    pos2 = conn.extract_position(["x", "y"], state)
    assert pos2["x"] == pytest.approx(state.x)
    assert pos2["y"] == state.y
    assert len(pos2) == len(state)
    assert list(pos2.keys()) == ["x", "y"]

    pos3 = conn.extract_position(["y", "x"], state)
    assert list(pos3.keys()) == ["y", "x"]


def test_update() -> None:
    conn = NamedTupleInterface(log_prob)
    state0 = create_state()

    state1: State = conn.update_state(Position({"y": 2}), state0)
    assert state1.y == 2
    assert state0.y == 1

    state2: State = conn.update_state(Position({"y": 3, "x": 2 * state0.x}), state0)
    assert state2.y == 3
    assert state0.y == 1

    assert conn.log_prob(state0) == pytest.approx(-2.0)
    assert conn.log_prob(state2) == pytest.approx(-8.0)
