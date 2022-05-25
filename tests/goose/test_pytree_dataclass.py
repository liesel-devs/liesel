from dataclasses import dataclass

import jax
import pytest

from liesel.goose.pytree import register_dataclass_as_pytree


@dataclass
class NoPytree:
    foo: float
    bar: int


@register_dataclass_as_pytree
@dataclass
class Pytree:
    foo: float
    bar: int


@jax.jit
def set(foobar):
    foobar.bar = 2
    foobar.foo = 0.0
    return foobar


def test_dataclass_not_working():
    no_pytree = NoPytree(foo=1.0, bar=0)
    with pytest.raises(TypeError):
        _ = set(no_pytree)


def test_registered_dataclass_is_working():
    pytree = Pytree(foo=1.0, bar=0)

    pytree = set(pytree)

    assert pytree.foo == 0.0
    assert pytree.bar == 2
