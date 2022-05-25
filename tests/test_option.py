import pytest

from liesel.option import Option


def test_none() -> None:
    opt: Option[int] = Option(None)
    assert opt.is_none()
    assert not opt.is_some()

    opt2: Option[int] = Option.none()
    assert opt2.is_none()
    assert not opt2.is_some()


def test_some() -> None:
    opt: Option[int] = Option(1)
    assert not opt.is_none()
    assert opt.is_some()

    opt2: Option[int] = Option.some(1)
    assert not opt2.is_none()
    assert opt2.is_some()


def test_unwrap() -> None:
    opt: Option[int] = Option(1)
    assert opt.unwrap() == 1

    opt2: Option[int] = Option.some(1)
    assert opt2.unwrap() == 1

    opt3: Option[int] = Option(None)
    with pytest.raises(RuntimeError):
        opt3.unwrap()


def test_expect() -> None:
    opt: Option[int] = Option(1)
    assert opt.expect("my msg") == 1

    opt2: Option[int] = Option.some(1)
    assert opt2.expect("my msg") == 1

    opt3: Option[int] = Option(None)
    with pytest.raises(RuntimeError):
        opt3.expect("This should fail!")


def test_unwrap_or() -> None:
    opt: Option[int] = Option(1)
    assert opt.unwrap_or(2) == 1

    opt2: Option[int] = Option(None)
    assert opt2.unwrap_or(2) == 2


def test_map() -> None:
    opt: Option[int] = Option(1)
    assert opt.map(lambda x: 2 * x) == Option(2)

    opt2: Option[int] = Option(None)
    assert opt2.map(lambda x: 2 * x) == Option(None)


def test_map_or() -> None:
    opt: Option[int] = Option(1)
    assert opt.map_or(4, lambda x: 2 * x) == 2

    opt2: Option[int] = Option(None)
    assert opt2.map_or(4, lambda x: 2 * x) == 4
