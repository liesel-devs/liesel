from liesel.goose.epoch import EpochType
from liesel.goose.warmup import stan_epochs


def test_stan_epochs():
    epochs = stan_epochs()
    durations = [epoch.duration for epoch in epochs]
    types = [epoch.type for epoch in epochs]
    i = EpochType.INITIAL_VALUES
    f = EpochType.FAST_ADAPTATION
    s = EpochType.SLOW_ADAPTATION
    p = EpochType.POSTERIOR

    assert durations == [1, 75, 25, 50, 100, 200, 500, 50, 1000]
    assert types == [i, f, s, s, s, s, s, f, p]
