from dataclasses import dataclass

import jax
import pytest

from liesel.goose.pytree import register_dataclass_as_pytree
from liesel.goose.rw import RWKernelState


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


def test_kernel_state_internal_fields_are_leaves():
    state = RWKernelState(
        step_size=1.0,
        error_sum=2.0,
        log_avg_step_size=3.0,
        mu=4.0,
    )

    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, [5.0, 6.0, 7.0, 8.0])

    assert leaves == [1.0, 2.0, 3.0, 4.0]
    assert restored == RWKernelState(
        step_size=5.0,
        error_sum=6.0,
        log_avg_step_size=7.0,
        mu=8.0,
    )
