import logging
from contextlib import contextmanager
from typing import Generator

import pytest
from _pytest.logging import LogCaptureHandler


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


@contextmanager
def local_caplog_fn(
    level: int = logging.INFO, name: str = "liesel"
) -> Generator[LogCaptureHandler, None, None]:
    """
    Context manager that captures records from non-propagating loggers.

    After the end of the ``with`` statement, the log level is restored to its original
    value. Code adapted from `this GitHub comment <GH>`_.

    .. _GH: https://github.com/pytest-dev/pytest/issues/3697#issuecomment-790925527

    Parameters
    ----------
    level
        The log level.
    name
        The name of the logger to update.
    """

    logger = logging.getLogger(name)

    old_level = logger.level
    logger.setLevel(level)

    handler = LogCaptureHandler()
    logger.addHandler(handler)

    try:
        yield handler
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(handler)


@pytest.fixture
def local_caplog():
    """
    Fixture that yields a context manager for capturing records from non-propagating
    loggers.

    Examples
    --------
    Usage example::

        import liesel.liesel.distreg as dr

        def test_build_empty(local_caplog):
            with local_caplog() as caplog:
                drb = dr.DistRegBuilder()
                model = drb.build()
                assert len(caplog.records) == 1
                assert caplog.records[0].levelname == "WARNING"
    """

    yield local_caplog_fn
