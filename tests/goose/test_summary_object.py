from dataclasses import dataclass
from typing import ClassVar, Sequence

import jax.numpy as jnp
import pytest
from jax.random import KeyArray

from liesel.goose.builder import EngineBuilder
from liesel.goose.engine import SamplingResult
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
from liesel.goose.summary_m import Summary
from liesel.goose.types import Kernel, ModelInterface, ModelState, Position


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
    error_book: ClassVar[dict[int, str]] = {0: "no errors"}
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
def result() -> SamplingResult:
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


# TODO: add tests to test correctness of quantities
# TODO: speed up tests


def test_shapes(result: SamplingResult):
    summary = Summary.from_result(result)

    # combined chains
    assert summary.quantities["mean"]["foo"].shape == (3,)
    assert summary.quantities["mean"]["bar"].shape == (3, 5, 7)
    assert summary.quantities["mean"]["baz"].shape == ()

    assert summary.quantities["quantile"]["foo"].shape == (
        3,
        3,
    )
    assert summary.quantities["quantile"]["bar"].shape == (3, 3, 5, 7)
    assert summary.quantities["quantile"]["baz"].shape == (3,)

    assert summary.quantities["hdi"]["foo"].shape == (
        2,
        3,
    )
    assert summary.quantities["hdi"]["bar"].shape == (2, 3, 5, 7)
    assert summary.quantities["hdi"]["baz"].shape == (2,)

    # combined chains
    summary = Summary.from_result(result, quantiles=(0.2, 0.4), per_chain=True)

    assert summary.quantities["mean"]["foo"].shape == (
        3,
        3,
    )
    assert summary.quantities["mean"]["bar"].shape == (3, 3, 5, 7)
    assert summary.quantities["mean"]["baz"].shape == (3,)

    assert summary.quantities["quantile"]["foo"].shape == (
        3,
        2,
        3,
    )
    assert summary.quantities["quantile"]["bar"].shape == (3, 2, 3, 5, 7)
    assert summary.quantities["quantile"]["baz"].shape == (3, 2)

    assert summary.quantities["hdi"]["foo"].shape == (
        3,
        2,
        3,
    )
    assert summary.quantities["hdi"]["bar"].shape == (3, 2, 3, 5, 7)
    assert summary.quantities["hdi"]["baz"].shape == (
        3,
        2,
    )


def test_additional_chain(result: SamplingResult):
    chain = result.get_posterior_samples()
    chain["expbaz"] = jnp.log(chain["baz"] + 1)
    chain["expbar"] = jnp.log(chain["bar"] + 1)

    summary = Summary.from_result(result, chain)

    assert summary.quantities["mean"]["expbar"].shape == (3, 5, 7)
    assert summary.quantities["mean"]["expbaz"].shape == ()


def test_selected(result: SamplingResult):
    summary = Summary.from_result(result, selected=["foo"])

    assert "foo" in summary.quantities["mean"]
    assert "bar" not in summary.quantities["mean"]
    assert "baz" not in summary.quantities["mean"]


def test_deselected(result: SamplingResult):
    summary = Summary.from_result(result, deselected=["baz"])

    assert "foo" in summary.quantities["mean"]
    assert "bar" in summary.quantities["mean"]
    assert "baz" not in summary.quantities["mean"]


def test_mean(result: SamplingResult):
    summary = Summary.from_result(result)
    assert jnp.allclose(
        summary.quantities["mean"]["foo"], jnp.array([175.5, 176.5, 177.5])
    )

    assert jnp.allclose(summary.quantities["mean"]["bar"], 175.5 * jnp.ones((3, 5, 7)))

    assert jnp.allclose(summary.quantities["mean"]["baz"], jnp.array(176.5))


def test_config(result: SamplingResult):
    summary = Summary.from_result(result, quantiles=(0.4, 0.6), hdi_prob=0.5)
    assert summary.config["chains_merged"]
    assert summary.config["quantiles"] == (0.4, 0.6)
    assert summary.config["hdi_prob"] == 0.5


def test_sample_info(result: SamplingResult):
    summary = Summary.from_result(result)
    print(summary.sample_info)
    assert summary.sample_info["num_chains"] == 3
    assert summary.sample_info["sample_size_per_chain"] == 250


def test_df_sample_info(result: SamplingResult):
    summary = Summary.from_result(result, selected=["baz"]).to_dataframe()
    assert summary["sample_size"][0] == 3 * 250

    summary = Summary.from_result(
        result, per_chain=True, selected=["baz"]
    ).to_dataframe()
    assert summary["sample_size"][0] == 250
