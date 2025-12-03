from liesel.goose.epoch import EpochType
from liesel.goose.warmup import stan_epochs


def test_stan_epochs():
    epochs = stan_epochs()
    durations = [epoch.duration for epoch in epochs]
    types = [epoch.type for epoch in epochs]
    f = EpochType.FAST_ADAPTATION
    s = EpochType.SLOW_ADAPTATION
    p = EpochType.POSTERIOR

    assert durations == [75, 25, 50, 100, 200, 500, 50, 1000]
    assert types == [f, s, s, s, s, s, f, p]


def test_stan_epochs_thinning():
    epochs = stan_epochs(thinning_posterior=10, thinning_warmup=5)
    thinning = [epoch.thinning for epoch in epochs]
    types = [epoch.type for epoch in epochs]
    f = EpochType.FAST_ADAPTATION
    s = EpochType.SLOW_ADAPTATION
    p = EpochType.POSTERIOR

    assert thinning == [5, 5, 5, 5, 5, 5, 5, 10]
    assert types == [f, s, s, s, s, s, f, p]
