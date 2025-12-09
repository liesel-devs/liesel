import logging
from collections.abc import Generator
from contextlib import contextmanager

import jax.numpy as jnp
import pytest
from _pytest.logging import LogCaptureHandler

from liesel.goose.builder import EngineBuilder
from liesel.goose.engine import SamplingResults
from liesel.goose.epoch import EpochConfig, EpochType
from liesel.goose.interface import DictInterface

from .mock_kernel import MockKernel


def pytest_addoption(parser):
    parser.addoption(
        "--run-mcmc", action="store_true", default=False, help="run mcmc tests"
    )

    parser.addoption(
        "--mcmc-seed", action="store", default=42, help="set mcmc seed", type=int
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
) -> Generator[LogCaptureHandler]:
    """
    Context manager that captures records from non-propagating loggers.

    After the end of the ``with`` statement, the log level is restored to its original
    value. Code adapted from `this GitHub comment <GH_>`_.

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

        import liesel.model.distreg as dr


        def test_build_empty(local_caplog):
            with local_caplog() as caplog:
                drb = dr.DistRegBuilder()
                model = drb.build()
                assert len(caplog.records) == 1
                assert caplog.records[0].levelname == "WARNING"
    """

    yield local_caplog_fn


@pytest.fixture(scope="module")
def result() -> SamplingResults:
    builder = EngineBuilder(0, 3)
    state = {
        "foo": jnp.arange(3, dtype=jnp.float32),
        "bar": jnp.zeros((3, 5, 7)),
        "baz": jnp.array(1.0),
    }

    builder.add_kernel(MockKernel(list(state.keys())))
    builder.set_model(DictInterface(log_prob_fn=lambda state: 0.0))
    builder.set_initial_values(state)
    builder.set_epochs(
        [
            EpochConfig(EpochType.BURNIN, 50, 1, None),
            EpochConfig(EpochType.POSTERIOR, 250, 1, None),
        ]
    )
    engine = builder.build()
    engine.sample_all_epochs()
    return engine.get_results()


@pytest.fixture(scope="module")
def result_thinned() -> SamplingResults:
    builder = EngineBuilder(0, 3)
    state = {
        "foo": jnp.arange(3, dtype=jnp.float32),
        "bar": jnp.zeros((3, 5, 7)),
        "baz": jnp.array(1.0),
    }

    builder.add_kernel(MockKernel(list(state.keys())))
    builder.set_model(DictInterface(log_prob_fn=lambda state: 0.0))
    builder.set_initial_values(state)
    builder.set_epochs(
        [
            EpochConfig(EpochType.BURNIN, 50, 2, None),
            EpochConfig(EpochType.POSTERIOR, 250, 2, None),
        ]
    )
    engine = builder.build()
    engine.sample_all_epochs()
    return engine.get_results()


@pytest.fixture(scope="module")
def result_for_quants() -> SamplingResults:
    builder = EngineBuilder(0, 4)
    state = {
        "foo": jnp.arange(5, dtype=jnp.float32),
        "bar": jnp.zeros((3, 5, 7)),
        "baz": jnp.array(1.0),
    }

    builder.add_kernel(MockKernel(list(state.keys())))
    builder.set_model(DictInterface(log_prob_fn=lambda state: 0.0))
    builder.set_initial_values(state)
    builder.set_epochs(
        [
            EpochConfig(EpochType.BURNIN, 50, 1, None),
            EpochConfig(EpochType.POSTERIOR, 250, 1, None),
        ]
    )
    engine = builder.build()
    engine.sample_all_epochs()
    return engine.get_results()


@pytest.fixture(scope="module")
def single_chain_result() -> SamplingResults:
    builder = EngineBuilder(0, 1)
    state = {
        "foo": jnp.arange(3, dtype=jnp.float32),
        "bar": jnp.zeros((3, 5, 7)),
        "baz": jnp.array(1.0),
    }

    builder.add_kernel(MockKernel(list(state.keys())))
    builder.set_model(DictInterface(log_prob_fn=lambda state: 0.0))
    builder.set_initial_values(state)
    builder.set_epochs(
        [
            EpochConfig(EpochType.BURNIN, 50, 1, None),
            EpochConfig(EpochType.POSTERIOR, 250, 1, None),
        ]
    )
    engine = builder.build()
    engine.sample_all_epochs()
    return engine.get_results()
