from types import MappingProxyType
from typing import Any, assert_type

import liesel
import liesel.goose as gs
import liesel.model as lsl
from liesel.types import Position, PositionInput


def test_position_accepts_dict_without_copying() -> None:
    values: dict[str, float] = {"mu": 1.0}
    position: Position = values

    assert position is values
    assert Position(values) is values


def test_input_aliases_and_concrete_outputs() -> None:
    position: PositionInput = MappingProxyType({"x": 2.0})
    model = lsl.Model(lsl.Var(0.0, name="x"))
    state: lsl.LieselModelStateInput = MappingProxyType(model.state)
    updated: lsl.LieselModelState = model.update_state(position, state)
    assert_type(updated, dict[str, lsl.NodeState])
    extracted = model.extract_position(["x"], updated)
    assert_type(extracted, dict[str, Any])
    assert type(updated) is dict
    assert type(extracted) is dict
    assert extracted == {"x": 2.0}
    assert (
        liesel.PositionInput is gs.PositionInput is lsl.PositionInput is PositionInput
    )
