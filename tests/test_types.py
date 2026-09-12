from liesel.types import Position


def test_position_accepts_dict_without_copying() -> None:
    values: dict[str, float] = {"mu": 1.0}
    position: Position = values

    assert position is values
    assert Position(values) is values
