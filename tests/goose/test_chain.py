from typing import Any

import jax.numpy as jnp

from liesel.goose.chain import (
    Chain,
    EpochChain,
    EpochChainManager,
    ListChain,
    ListEpochChain,
)
from liesel.goose.epoch import EpochConfig, EpochType
from liesel.goose.types import PyTree


def create_epoch(dur: int) -> EpochConfig:
    return EpochConfig(
        type=EpochType.POSTERIOR, duration=dur, thinning=1, optional=None
    )


def create_chunk() -> PyTree:
    chunk = (
        jnp.expand_dims(jnp.array([1.0, 2.0, 3.0]), (0, 1)),
        {
            "foo": jnp.expand_dims(jnp.array(1.0), (0, 1)),
            "bar": jnp.expand_dims(jnp.array([[1.0, 2], [3.0, 4.0]]), (0, 1)),
        },
    )
    return chunk


def test_list_chain() -> None:
    chain: Chain = ListChain()
    assert chain.get().is_none()

    chain.append(create_chunk())
    assert chain.get().is_some()

    chain.append(create_chunk())
    assert chain.get().is_some()

    chain.append(create_chunk())
    chain.append(create_chunk())
    chain.append(create_chunk())
    comp = chain.get().unwrap()
    assert comp[0].shape == (1, 5, 3)
    assert comp[1]["foo"].shape == (1, 5)
    assert comp[1]["bar"].shape == (1, 5, 2, 2)


def test_epoch_list_chain():
    chain: EpochChain[Any] = ListEpochChain(create_epoch(11))
    assert chain.get().is_none()

    chain.append(create_chunk())
    assert chain.get().is_some()

    chain.append(create_chunk())
    assert chain.get().is_some()

    chain.append(create_chunk())
    chain.append(create_chunk())
    chain.append(create_chunk())
    comp = chain.get().unwrap()
    assert comp[0].shape == (1, 5, 3)
    assert comp[1]["foo"].shape == (1, 5)
    assert comp[1]["bar"].shape == (1, 5, 2, 2)


def test_epoch_chain_manager() -> None:
    manager: EpochChainManager[Any] = EpochChainManager()

    nepochs = 3
    for i in range(nepochs):
        dur = (1 + i) * 5
        manager.advance_epoch(create_epoch(dur))
        for _ in range(manager.get_current_epoch().duration):
            manager.append(create_chunk())

    # test combine_all
    chain = manager.combine_all().unwrap()
    assert chain[0].shape == (1, 5 + 10 + 15, 3)

    # test combine
    chain = manager.combine((1, 2)).unwrap()
    assert chain[0].shape == (1, 10 + 15, 3)

    # test combine_filtered
    chain = manager.combine_filtered(lambda config: config.duration == 10).unwrap()
    assert chain[0].shape == (1, 10, 3)


def test_epoch_chain_thinning() -> None:
    # thinning larger than chunk size
    chain: ListEpochChain[Any] = ListEpochChain(
        EpochConfig(EpochType.POSTERIOR, 100, 10, None), apply_thinning=True
    )

    data = jnp.arange(0, 10).reshape(2, 5)

    for _ in range(100 // 5):
        chain.append(data)
    res = chain.get().unwrap()

    assert res.shape == (2, 10)
    assert jnp.all(res.sum(axis=1) == 10 * jnp.array((4, 9)))

    # thinning smaller than chunk size
    chain = ListEpochChain(
        EpochConfig(EpochType.POSTERIOR, 99, 3, None), apply_thinning=True
    )

    data = jnp.arange(0, 18).reshape(2, 9)

    for _ in range(99 // 9):
        chain.append(data)
    res = chain.get().unwrap()

    assert res.shape == (2, 33)
    assert jnp.all(
        res.sum(axis=1) == 11 * jnp.array(((2, 5, 8), (11, 14, 17))).sum(axis=1)
    )

    # thinning enabled but set to 1
    chain = ListEpochChain(
        EpochConfig(EpochType.POSTERIOR, 10, 1, None), apply_thinning=True
    )

    data = jnp.arange(0, 10).reshape(2, 5)

    for _ in range(10 // 5):
        chain.append(data)
    res = chain.get().unwrap()

    assert res.shape == (2, 10)
    assert jnp.all(res.sum(axis=1) == 2 * data.sum(axis=1))
