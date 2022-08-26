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


class LocalLogCaptureFixture:
    """
    Provides access and control of log capturing.

    Solves the issue that the caplog fixture listens only to the root logger - which
    causes it to miss messages from non-propagating loggers.

    Code adapted from
    https://github.com/pytest-dev/pytest/issues/3697#issuecomment-790925527
    """

    def __init__(self):
        self.handler = LogCaptureHandler()

    @contextmanager
    def __call__(
        self, level: int = logging.INFO, name: str = "liesel"
    ) -> Generator[None, None, None]:
        """
        Context manager that sets the level for capturing of logs. After the end of the
        'with' statement the level is restored to its original value.

        Parameters
        ----------
        level
            The log level.
        name
            The name of the logger to update.

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
        logger = logging.getLogger(name)

        orig_level = logger.level
        logger.setLevel(level)

        logger.addHandler(self.handler)
        try:
            yield self.handler
        finally:
            logger.setLevel(orig_level)
            logger.removeHandler(self.handler)


@pytest.fixture
def local_caplog() -> Generator[LocalLogCaptureFixture, None, None]:
    yield LocalLogCaptureFixture()
