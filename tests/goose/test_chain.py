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


def test_multi_list_chain() -> None:
    chain: Chain = ListChain(True)
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


def test_single_list_chain() -> None:
    chain: Chain = ListChain(False)
    assert chain.get().is_none()

    chain.append(create_chunk())
    assert chain.get().is_some()

    chain.append(create_chunk())
    assert chain.get().is_some()

    chain.append(create_chunk())
    chain.append(create_chunk())
    chain.append(create_chunk())
    comp = chain.get().unwrap()
    assert comp[0].shape == (5, 1, 3)
    assert comp[1]["foo"].shape == (5, 1)
    assert comp[1]["bar"].shape == (5, 1, 2, 2)


def test_epoch_list_chain():
    chain: EpochChain[Any] = ListEpochChain(True, create_epoch(11))
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


# def test_epoch_chain_no_epoch():
#     chain = DefaultEpochChain()
#     with pytest.raises(RuntimeError):
#         chain.append(make_chunk())

#     assert chain.get_latest_epoch() is None
#     assert chain.get_latest_chain() is None
#     assert not chain.is_epoch_active()


def test_epoch_chain_manager() -> None:
    manager: EpochChainManager[Any] = EpochChainManager(False)

    nepochs = 3
    for i in range(nepochs):
        dur = (1 + i) * 5
        manager.advance_epoch(create_epoch(dur))
        for _ in range(manager.get_current_epoch().duration):
            manager.append(create_chunk())

    # test combine_all
    chain = manager.combine_all().unwrap()
    assert chain[0].shape == (5 + 10 + 15, 1, 3)

    # test combine
    chain = manager.combine((1, 2)).unwrap()
    assert chain[0].shape == (10 + 15, 1, 3)

    # test combine_filtered
    chain = manager.combine_filtered(lambda config: config.duration == 10).unwrap()
    assert chain[0].shape == (10, 1, 3)
