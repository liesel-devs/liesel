import jax
import pytest

from liesel.goose.epoch import EpochConfig, EpochManager, EpochType


def test_epoch_state_time() -> None:
    state = EpochConfig(EpochType.POSTERIOR, 10, 1, None).to_state(0, 0)

    assert state.time_left() == 10

    state.advance_time(5)
    assert state.time_left() == 5
    assert state.time == 5
    assert state.time_in_epoch == 5

    state.advance_time(5)
    assert state.time_left() == 0
    assert state.time == 10
    assert state.time_in_epoch == 10

    state = EpochConfig(EpochType.POSTERIOR, 10, 1, None).to_state(1, 10)
    assert state.time_left() == 10
    assert state.time == 10
    assert state.time_in_epoch == 0
    state.advance_time(10)
    assert state.time == 20
    assert state.time_in_epoch == 10
    assert state.time_left() == 0


def test_is_epoch_type() -> None:
    type = EpochType.INITIAL_VALUES
    assert not EpochType.is_adaptation(type)
    assert not EpochType.is_warmup(type)

    type = EpochType.SLOW_ADAPTATION
    assert EpochType.is_adaptation(type)
    assert EpochType.is_warmup(type)

    type = EpochType.FAST_ADAPTATION
    assert EpochType.is_adaptation(type)
    assert EpochType.is_warmup(type)

    type = EpochType.BURNIN
    assert not EpochType.is_adaptation(type)
    assert EpochType.is_warmup(type)

    type = EpochType.POSTERIOR
    assert not EpochType.is_adaptation(type)
    assert not EpochType.is_warmup(type)


def test_is_epoch_type_jitted() -> None:
    jis_adapt = jax.jit(EpochType.is_adaptation)
    jis_warmup = jax.jit(EpochType.is_warmup)
    type = EpochType.INITIAL_VALUES
    assert not jis_adapt(type)
    assert not jis_warmup(type)

    type = EpochType.SLOW_ADAPTATION
    assert jis_adapt(type)
    assert jis_warmup(type)

    type = EpochType.FAST_ADAPTATION
    assert jis_adapt(type)
    assert jis_warmup(type)

    type = EpochType.BURNIN
    assert not jis_adapt(type)
    assert jis_warmup(type)

    type = EpochType.POSTERIOR
    assert not jis_adapt(type)
    assert not jis_warmup(type)


def test_epoch_manager() -> None:
    configs = (
        EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
        EpochConfig(EpochType.FAST_ADAPTATION, 3, 1, None),
        EpochConfig(EpochType.POSTERIOR, 5, 1, None),
    )

    manager = EpochManager(configs)

    # in the beginning we are in no epoch
    assert manager.has_more()

    # advance to first epoch
    state = manager.next()
    assert state.nth_epoch == 0
    assert state.time_before_epoch == 0
    assert manager.has_more()

    # advance to second epoch
    state = manager.next()
    assert state.nth_epoch == 1
    assert state.time_before_epoch == 1
    assert manager.has_more()

    # advance to third epoch
    state = manager.next()
    assert state.nth_epoch == 2
    assert state.time_before_epoch == 4

    # manager is now exhausted
    assert not manager.has_more()
    with pytest.raises(RuntimeError):
        manager.next()

    # add epoch configs
    manager.append(EpochConfig(EpochType.POSTERIOR, 7, 1, None))
    manager.append(EpochConfig(EpochType.POSTERIOR, 11, 1, None))

    # more epoch are now available
    assert manager.has_more()

    # advance to fourth epoch
    state = manager.next()
    assert state.nth_epoch == 3
    assert state.time_before_epoch == 9

    # advance to fith epoch
    state = manager.next()
    assert state.nth_epoch == 4
    assert state.time_before_epoch == 16

    # assert that the config matches
    assert state.config.type == EpochType.POSTERIOR
    assert state.config.duration == 11

    # manager is now exhausted
    assert not manager.has_more()
    with pytest.raises(RuntimeError):
        manager.next()


def test_epoch_manager_with_faulty_configs() -> None:
    # first epoch must be initial values
    with pytest.raises(RuntimeError):
        _ = EpochManager([EpochConfig(EpochType.POSTERIOR, 1, 1, None)])

    # first epoch must have length 1
    with pytest.raises(RuntimeError):
        _ = EpochManager([EpochConfig(EpochType.INITIAL_VALUES, 2, 1, None)])

    manager = EpochManager([EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None)])

    # initial values may not follow at a later position
    with pytest.raises(RuntimeError):
        manager.append(EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None))

    # thinning must be smaller or equal to duration
    with pytest.raises(RuntimeError):
        manager.append(EpochConfig(EpochType.FAST_ADAPTATION, 5, 10, None))

    # warm up cannot follow posterior
    manager.append(EpochConfig(EpochType.FAST_ADAPTATION, 3, 1, None))
    manager.append(EpochConfig(EpochType.POSTERIOR, 4, 1, None))
    with pytest.raises(RuntimeError):
        manager.append(EpochConfig(EpochType.FAST_ADAPTATION, 1, 1, None))

    # duration must be positive
    with pytest.raises(RuntimeError):
        manager.append(EpochConfig(EpochType.POSTERIOR, -1, 1, None))

    # duration must be multiple of thinning
    with pytest.raises(RuntimeError):
        manager.append(EpochConfig(EpochType.POSTERIOR, 13, 7, None))


def test_epoch_manager_with_thinning() -> None:
    manager = EpochManager([EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None)])

    # thinning must be positive
    with pytest.raises(RuntimeError):
        manager.append(EpochConfig(EpochType.FAST_ADAPTATION, 20, -1, None))

    # thinning can be larger than one in WARMUP
    manager.append(EpochConfig(EpochType.FAST_ADAPTATION, 100, 10, None))

    # thinning can be larger than one in POSTERIOR epoch
    manager.append(EpochConfig(EpochType.POSTERIOR, 100, 20, None))
