import pytest


def to_int(value):
    try:
        value = int(value)
    except ValueError:
        raise pytest.UsageError("--mcmc-seed must specify an integer")

    return value


def pytest_addoption(parser):
    parser.addoption(
        "--run-mcmc", action="store_true", default=False, help="run mcmc tests"
    )

    parser.addoption(
        "--mcmc-seed", action="store", default=42, help="set mcmc seed", type=to_int
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "mcmc: mark test as mcmc test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-mcmc"):
        # --run-mcmc given in cli: do not skip mcmc tests
        return

    skip_mcmc = pytest.mark.skip(reason="need --run-mcmc option to run")

    for item in items:
        if "mcmc" in item.keywords:
            item.add_marker(skip_mcmc)


@pytest.fixture
def mcmc_seed(request):
    return request.config.getoption("--mcmc-seed")
