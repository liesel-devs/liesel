from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

import liesel
from liesel.experimental.arviz import to_arviz_inference_data
from liesel.goose.builder import EngineBuilder
from liesel.goose.engine import SamplingResults
from liesel.goose.epoch import EpochConfig, EpochState, EpochType
from liesel.goose.kernel import (
    DefaultTransitionInfo,
    ModelMixin,
    TransitionOutcome,
    TuningOutcome,
    WarmupOutcome,
)
from liesel.goose.models import DictModel
from liesel.goose.pytree import register_dataclass_as_pytree
from liesel.goose.types import Kernel, KeyArray, ModelInterface, ModelState, Position


@register_dataclass_as_pytree
@dataclass
class MockKernelState:
    pass

    @staticmethod
    def default() -> "MockKernelState":
        return MockKernelState()


@register_dataclass_as_pytree
@dataclass
class MockKernelTuningInfo:
    error_code: int
    time: int


@register_dataclass_as_pytree
@dataclass
class MockKernelTransInfo:
    error_code: int = 0
    acceptance_prob: float = 1
    position_moved: int = 1

    def minimize(self) -> DefaultTransitionInfo:
        return DefaultTransitionInfo(
            self.error_code, self.acceptance_prob, self.position_moved
        )


class MockKernel(ModelMixin):
    error_book: ClassVar[dict[int, str]] = {0: "no errors", 1: "error 1", 2: "error 2"}
    needs_history: ClassVar[bool] = False
    identifier: str = ""

    def __init__(
        self,
        position_keys: Sequence[str],
    ):
        self._model: None | ModelInterface = None
        self.position_keys = tuple(position_keys)

    def init_state(
        self, prng_key: KeyArray, model_state: ModelState
    ) -> MockKernelState:
        return MockKernelState.default()

    def start_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> MockKernelState:
        return kernel_state

    def end_epoch(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> MockKernelState:
        return kernel_state

    def transition(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        epoch: EpochState,
    ) -> TransitionOutcome[MockKernelState, MockKernelTransInfo]:
        position = self.model.extract_position(self.position_keys, model_state)
        for key in position:
            position[key] = position[key] + 1.0

        new_model_state = self.model.update_state(position, model_state)

        info = MockKernelTransInfo()
        return TransitionOutcome(info, kernel_state, new_model_state)

    def tune(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        epoch: EpochState,
        history: Position | None,
    ) -> TuningOutcome[MockKernelState, MockKernelTuningInfo]:
        info = MockKernelTuningInfo(error_code=0, time=epoch.time)
        return TuningOutcome(info, kernel_state)

    def end_warmup(
        self,
        prng_key: KeyArray,
        kernel_state: MockKernelState,
        model_state: ModelState,
        tuning_history: MockKernelTuningInfo | None,
    ) -> WarmupOutcome[MockKernelState]:
        return WarmupOutcome(0, kernel_state)


def typecheck() -> None:
    _: Kernel[MockKernelState, MockKernelTransInfo, MockKernelTuningInfo] = MockKernel(
        ["foo", "bar"]
    )


def logprob(state):
    return 0.0


@pytest.fixture(scope="module")
def result() -> SamplingResults:
    builder = EngineBuilder(0, 3)
    state = {
        "foo": jnp.arange(3, dtype=jnp.float32),
        "bar": jnp.zeros((3, 5, 7)),
        "baz": jnp.array(1.0),
    }

    builder.add_kernel(MockKernel(list(state.keys())))
    builder.set_model(DictModel(logprob))
    builder.set_initial_values(state)
    builder.set_epochs(
        [
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.BURNIN, 50, 1, None),
            EpochConfig(EpochType.POSTERIOR, 250, 1, None),
        ]
    )
    engine = builder.build()
    engine.sample_all_epochs()
    return engine.get_results()


def test_structure(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    assert infdat.groups() == ["posterior"]

    assert infdat.posterior["foo"].shape == (3, 250, 3)
    assert infdat.posterior["bar"].shape == (3, 250, 3, 5, 7)
    assert infdat.posterior["baz"].shape == (
        3,
        250,
    )


def test_attributes(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    infdat.posterior.attrs["inference_library"] == "liesel"
    infdat.posterior.attrs["inference_library_version"] == liesel.__version__


def test_posterior_mean(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    assert np.allclose(
        infdat.posterior["foo"].mean(["chain", "draw"]), [175.5, 176.5, 177.5]
    )

    assert np.allclose(
        infdat.posterior["bar"].mean(["chain", "draw"]), 175.5 * jnp.ones((3, 5, 7))
    )

    assert infdat.posterior["baz"].mean(["chain", "draw"]) == 176.5
