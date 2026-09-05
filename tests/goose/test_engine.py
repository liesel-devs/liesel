"""
some tests for the engine
"""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.goose.builder import EngineBuilder
from liesel.goose.chain import EpochChainManager
from liesel.goose.engine import (
    Engine,
    SamplingResults,
    _add_time_dimension,
    # stack_for_multi,
)
from liesel.goose.epoch import EpochConfig, EpochState, EpochType
from liesel.goose.interface import DictInterface
from liesel.goose.kernel import DefaultTransitionInfo
from liesel.goose.kernel_sequence import KernelSequence
from liesel.goose.pytree import (
    concatenate_leaves,
    register_dataclass_as_pytree,
    slice_leaves,
)
from liesel.goose.types import Array, KeyArray, ModelInterface, ModelState
from liesel.model import Model, Var
from liesel.option import Option

from .deterministic_kernels import (
    DetCountingKernel,
    DetCountingKernelState,
    DetCountingKernelTuningInfo,
)


@register_dataclass_as_pytree
@dataclass
class FooQuant:
    error_code: int
    result: tuple[Array, Array]


class FooQauntGen:
    error_book: ClassVar[dict[int, str]] = {0: "no errors"}

    def __init__(self, identifier):
        self.identifier = identifier

    def set_model(self, model: ModelInterface):
        pass

    def has_model(self) -> bool:
        return False

    def generate(
        self, prng_key: KeyArray, model_state: ModelState, epoch: EpochState
    ) -> FooQuant:
        u = jax.random.normal(prng_key)
        return FooQuant(0, (u, model_state["x"]))


def _engine_with_block_size(
    jitted_sample_duration: int = 2,
    epoch_configs: list[EpochConfig] | None = None,
    with_quantities: bool = False,
) -> Engine:
    num_chains = 2
    model_states = _stack_for_multi([{"x": jnp.array(0)}] * num_chains)
    model = DictInterface(lambda state: -0.5 * state["x"] ** 2)
    kernel = DetCountingKernel(["x"], DetCountingKernelState.default())
    kernel.set_model(model)
    kernel.identifier = "kernel_00"

    if epoch_configs is None:
        epoch_configs = [
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.BURNIN, 4, 1, None),
        ]

    return Engine(
        seeds=jax.random.split(jax.random.PRNGKey(1), num_chains),
        model_states=model_states,
        kernel_sequence=KernelSequence([kernel]),
        epoch_configs=epoch_configs,
        jitted_sample_duration=jitted_sample_duration,
        model=model,
        position_keys=["x"],
        quantity_generators=[FooQauntGen("foo")] if with_quantities else [],
        show_progress=False,
    )


def _stop_in_second_epoch(engine: Engine, monkeypatch) -> None:
    engine.sample_next_epoch()
    engine.max_wall_time = 1.0
    times = iter([10.0, 11.0])
    with monkeypatch.context() as patch:
        patch.setattr("liesel.goose.engine.monotonic", lambda: next(times))
        with pytest.raises(TimeoutError):
            engine.sample_next_epoch()
    engine.max_wall_time = None


def test_sample_next_epoch_resumes_active_epoch(monkeypatch):
    engine = _engine_with_block_size()
    _stop_in_second_epoch(engine, monkeypatch)

    assert not engine.is_sampling_done()

    engine.sample_next_epoch()

    samples = engine.get_results().get_samples()["x"]
    assert np.array_equal(samples[0], np.array([0, 10000, 10001, 10002, 10003]))
    assert engine.is_sampling_done()
    with pytest.raises(RuntimeError, match="No active epoch"):
        _ = engine.current_epoch


def test_sample_all_epochs_completes_active_final_epoch(monkeypatch):
    engine = _engine_with_block_size()
    _stop_in_second_epoch(engine, monkeypatch)

    engine.sample_all_epochs()

    samples = engine.get_results().get_samples()["x"]
    assert np.array_equal(samples[0], np.array([0, 10000, 10001, 10002, 10003]))
    assert engine.is_sampling_done()


def test_sample_all_epochs_resumes_before_pending_epochs(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.FAST_ADAPTATION, 4, 1, None),
            EpochConfig(EpochType.POSTERIOR, 2, 1, None),
        ]
    )
    _stop_in_second_epoch(engine, monkeypatch)

    engine.sample_all_epochs()

    samples = engine.get_results().get_samples()["x"]
    assert np.array_equal(
        samples[0], np.array([0, 10000, 10001, 10002, 10003, 20000, 20001])
    )
    assert np.array_equal(
        engine.get_results().get_tuning_times().unwrap(), np.array([[5], [5]])
    )
    assert engine.is_sampling_done()


def test_max_wall_time_can_be_changed_on_engine():
    engine = _engine_with_block_size()

    engine.max_wall_time = 1.5

    assert engine.max_wall_time == 1.5


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), 0.0, -1.0])
def test_engine_rejects_invalid_max_wall_time(value):
    engine = _engine_with_block_size()

    with pytest.raises(ValueError, match="max_wall_time"):
        engine.max_wall_time = value


def test_append_epoch_rejects_incompatible_duration():
    engine = _engine_with_block_size(2)

    with pytest.raises(ValueError, match=r"jit_block_size 2.*duration 3"):
        engine.append_epoch(EpochConfig(EpochType.POSTERIOR, 3, 1, None))


@pytest.mark.parametrize("value", [True, 1.5, "1", 0, -1])
def test_engine_rejects_invalid_jitted_sample_duration(value):
    with pytest.raises(ValueError, match="jit_block_size"):
        _engine_with_block_size(value)


def test_engine_rejects_block_size_incompatible_with_initial_schedule():
    with pytest.raises(ValueError, match=r"jit_block_size 3.*duration 4"):
        _engine_with_block_size(3)


def test_compile_preserves_state_and_supplies_later_sampling(monkeypatch):
    baseline = _engine_with_block_size(with_quantities=True)
    engine = _engine_with_block_size(with_quantities=True)

    assert engine.compile() is None
    assert engine.compile() is None
    assert not engine.is_sampling_done()
    with pytest.raises(RuntimeError, match="No active epoch"):
        _ = engine.current_epoch

    before = engine.get_results()
    assert before.positions.get_epochs() == []
    assert before.transition_infos.get_epochs() == []
    assert before.generated_quantities.unwrap().get_epochs() == []
    assert before.tuning_infos.unwrap().get().is_none()
    assert before.elapsed_wall_time is None
    assert before.stop_reason is None

    def unavailable_original_jit(*args, **kwargs):
        raise AssertionError("sampling used the original JIT wrapper")

    monkeypatch.setattr(engine, "_sample_many_jitted", unavailable_original_jit)
    engine.sample_all_epochs()
    baseline.sample_all_epochs()

    actual = engine.get_results()
    expected = baseline.get_results()
    jax.tree.map(
        np.testing.assert_array_equal,
        actual.positions.combine_all().unwrap(),
        expected.positions.combine_all().unwrap(),
    )
    jax.tree.map(
        np.testing.assert_array_equal,
        actual.transition_infos.combine_all().unwrap(),
        expected.transition_infos.combine_all().unwrap(),
    )
    jax.tree.map(
        np.testing.assert_array_equal,
        actual.generated_quantities.unwrap().combine_all().unwrap(),
        expected.generated_quantities.unwrap().combine_all().unwrap(),
    )


def test_compiled_engine_samples_epochs_with_different_argument_signatures():
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.BURNIN, 2, 1, jnp.zeros(1)),
            EpochConfig(EpochType.POSTERIOR, 2, 1, jnp.zeros(2)),
        ]
    )

    engine.compile()
    engine.sample_all_epochs()

    assert np.array_equal(
        engine.get_results().get_samples()["x"][0],
        np.array([0, 10000, 10001, 20000, 20001]),
    )


def test_compile_replaces_executable_for_next_argument_signature(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.BURNIN, 2, 1, jnp.zeros(1)),
            EpochConfig(EpochType.POSTERIOR, 2, 1, jnp.zeros(2)),
        ]
    )
    engine.compile()
    engine.sample_next_epoch()
    engine.sample_next_epoch()

    assert engine.compile() is None

    def unavailable_original_jit(*args, **kwargs):
        raise AssertionError("sampling used the original JIT wrapper")

    monkeypatch.setattr(engine, "_sample_many_jitted", unavailable_original_jit)
    engine.sample_next_epoch()

    assert np.array_equal(
        engine.get_results().get_samples()["x"][0],
        np.array([0, 10000, 10001, 20000, 20001]),
    )


@pytest.mark.parametrize("compiled_earlier", [False, True])
def test_compile_is_a_noop_without_a_sampling_epoch(compiled_earlier):
    if compiled_earlier:
        engine = _engine_with_block_size()
        engine.compile()
        engine.sample_all_epochs()
    else:
        engine = _engine_with_block_size(
            epoch_configs=[EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None)]
        )

    before = engine.get_results()
    epochs_before = before.positions.get_epochs()
    metadata_before = (before.elapsed_wall_time, before.stop_reason)

    assert engine.compile() is None

    after = engine.get_results()
    assert after.positions.get_epochs() == epochs_before
    assert (after.elapsed_wall_time, after.stop_reason) == metadata_before
    if compiled_earlier:
        assert np.array_equal(
            after.get_samples()["x"][0], np.array([0, 10000, 10001, 10002, 10003])
        )
    else:
        engine.sample_all_epochs()
        assert np.array_equal(engine.get_results().get_samples()["x"][0], np.array([0]))


def test_add_time_dimension():
    def get_dims(t):
        return [t[0].shape, t[1][0].shape, t[1][1]["f"].shape]

    tree0 = jax.jit(lambda x: x)(
        (jnp.array([1.0, 2.0]), [jnp.zeros((3, 3)), {"f": jnp.array([1.0, 2.0])}])
    )
    tree3 = _add_time_dimension(tree0)
    dims3 = [
        (2, 1),
        (3, 1, 3),
        (2, 1),
    ]
    assert dims3 == get_dims(tree3)


def test_error_log():
    errs: np.ndarray = np.array([0, 0, 1, 0, 0, 0, 1, 1]).reshape((2, -1))
    ti = DefaultTransitionInfo(errs, np.zeros((2, 4)), np.zeros((2, 4), np.int8))
    tis = {"kern0": ti}

    em = EpochChainManager()
    em.advance_epoch(EpochConfig(EpochType.POSTERIOR, 4, 1, None))
    em.append(tis)
    em.combine_all()

    sr = SamplingResults(
        EpochChainManager(),
        em,
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
    )

    error_log = sr.get_error_log().unwrap()
    kel = error_log["kern0"]
    assert kel.kernel_ident == "kern0"
    assert np.array_equal(kel.transition, np.array([2, 3]))
    assert np.array_equal(kel.error_codes, np.array([[1, 0], [1, 1]]))


def test_sampling_results_timing_metadata_defaults_to_none():
    results = SamplingResults(
        EpochChainManager(),
        EpochChainManager(),
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
        Option.none(),
    )

    assert results.elapsed_wall_time is None
    assert results.stop_reason is None


def test_sample_next_epoch_records_completed_call_metadata(monkeypatch):
    times = iter([10.0, 12.5])
    engine = _engine_with_block_size()
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    engine.sample_next_epoch()

    results = engine.get_results()
    assert results.elapsed_wall_time == 2.5
    assert results.stop_reason == "completed"


def test_sample_next_epoch_stops_at_safety_limit_with_resumable_state(monkeypatch):
    engine = _engine_with_block_size()
    engine.sample_next_epoch()
    engine.max_wall_time = 1.0
    times = iter([10.0, 11.0])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    with pytest.raises(
        TimeoutError,
        match=r"1\.0.*1\.0.*jit_block_size 2.*resumed",
    ):
        engine.sample_next_epoch()

    samples = engine.get_results().get_samples()["x"]
    assert np.array_equal(samples[0], np.array([0, 10000, 10001]))
    assert engine.current_epoch.time_in_epoch == 2
    assert not engine.is_sampling_done()
    assert engine.get_results().elapsed_wall_time == 1.0
    assert engine.get_results().stop_reason == "max_wall_time_reached"


def test_sample_all_epochs_uses_one_safety_budget_across_epochs(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.BURNIN, 2, 1, None),
            EpochConfig(EpochType.POSTERIOR, 2, 1, None),
        ]
    )
    engine.max_wall_time = 2.5
    times = iter([10.0, 11.0, 11.5, 13.0, 14.0])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    with pytest.raises(TimeoutError):
        engine.sample_all_epochs()

    assert engine.is_sampling_done()
    assert engine.get_results().elapsed_wall_time == 4.0
    assert engine.get_results().stop_reason == "max_wall_time_reached"


def test_sample_next_epoch_gets_a_new_safety_budget_when_resumed(monkeypatch):
    engine = _engine_with_block_size()
    engine.sample_next_epoch()
    engine.max_wall_time = 1.0
    times = iter([10.0, 11.0, 20.0, 20.5, 20.75, 20.75])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    with pytest.raises(TimeoutError):
        engine.sample_next_epoch()

    engine.sample_next_epoch()

    assert engine.is_sampling_done()
    assert engine.get_results().elapsed_wall_time == 0.75
    assert engine.get_results().stop_reason == "completed"


def test_safety_stop_on_final_block_finalizes_adaptation_before_raising(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.FAST_ADAPTATION, 2, 1, None),
        ]
    )
    engine.sample_next_epoch()
    engine.max_wall_time = 1.0
    times = iter([10.0, 10.5, 13.0])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    with pytest.raises(TimeoutError):
        engine.sample_next_epoch()

    assert engine.is_sampling_done()
    with pytest.raises(RuntimeError, match="No active epoch"):
        _ = engine.current_epoch
    assert np.array_equal(
        engine.get_results().get_tuning_times().unwrap(), np.array([[3], [3]])
    )
    assert engine.get_results().elapsed_wall_time == 3.0


def test_safety_clock_observes_synchronized_tuning_infos(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.FAST_ADAPTATION, 2, 1, None),
        ]
    )
    engine.sample_next_epoch()
    engine.max_wall_time = 1.0
    tuning_infos_ready = False
    real_block_until_ready = jax.block_until_ready

    def contains_tuning_info(value) -> bool:
        if isinstance(value, DetCountingKernelTuningInfo):
            return True
        if isinstance(value, dict):
            return any(contains_tuning_info(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_tuning_info(item) for item in value)
        return False

    def block_until_ready(value):
        nonlocal tuning_infos_ready
        result = real_block_until_ready(value)
        tuning_infos_ready |= contains_tuning_info(value)
        return result

    clock_started = False

    def monotonic():
        nonlocal clock_started
        if not clock_started:
            clock_started = True
            return 10.0
        if tuning_infos_ready:
            return 11.0
        return 10.5

    monkeypatch.setattr("liesel.goose.engine.jax.block_until_ready", block_until_ready)
    monkeypatch.setattr("liesel.goose.engine.monotonic", monotonic)

    with pytest.raises(TimeoutError):
        engine.sample_next_epoch()


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), 0, -1.0])
def test_sample_for_time_rejects_invalid_wall_time(value):
    engine = _engine_with_block_size()

    with pytest.raises(ValueError, match="wall_time"):
        engine.sample_for_time(value)


def test_sample_for_time_starts_clock_before_processing_arguments(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.BURNIN, 2, 1, None),
        ]
    )
    clock = [10.0]

    class WallTime(float):
        def __float__(self):
            clock[0] += 0.25
            return super().__float__()

    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: clock[0])

    engine.sample_for_time(WallTime(10.0))

    assert engine.get_results().elapsed_wall_time == 0.25


def test_sample_for_time_requires_explicit_jit_block_size():
    builder = EngineBuilder(seed=1, num_chains=2)
    builder.set_model(DictInterface(lambda state: -0.5 * state["x"] ** 2))
    builder.set_initial_values({"x": jnp.array(0)})
    builder.add_kernel(DetCountingKernel(["x"], DetCountingKernelState.default()))
    builder.set_epochs([EpochConfig(EpochType.BURNIN, 4, 1, None)])
    builder.show_progress = False
    engine = builder.build()

    with pytest.raises(
        ValueError,
        match=r"configure jit_block_size.*wall-clock overshoot.*JIT-block boundaries",
    ):
        engine.sample_for_time(1.0)


def test_sample_for_time_stops_normally_after_first_completed_block(monkeypatch):
    engine = _engine_with_block_size()
    times = iter([10.0, 11.0])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    engine.sample_for_time(1.0)

    samples = engine.get_results().get_samples()["x"]
    assert np.array_equal(samples[0], np.array([0, 10000, 10001]))
    assert engine.current_epoch.time_in_epoch == 2
    assert engine.get_results().elapsed_wall_time == 1.0
    assert engine.get_results().stop_reason == "wall_time_reached"


def test_sample_for_time_finishes_schedule_without_repeating_epochs(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.BURNIN, 2, 1, None),
        ]
    )
    times = iter([10.0, 10.5, 10.75, 11.0])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    engine.sample_for_time(10.0)

    samples = engine.get_results().get_samples()["x"]
    assert np.array_equal(samples[0], np.array([0, 10000, 10001]))
    assert engine.is_sampling_done()
    assert engine.get_results().elapsed_wall_time == 1.0
    assert engine.get_results().stop_reason == "completed"


def test_sample_for_time_uses_stricter_safety_limit_and_warns_once(monkeypatch, caplog):
    engine = _engine_with_block_size()
    engine.max_wall_time = 1.0
    times = iter([10.0, 11.0, 12.0, 12.5])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    with (
        caplog.at_level("WARNING", logger="liesel.goose.engine"),
        pytest.raises(TimeoutError),
    ):
        engine.sample_for_time(2.0)

    assert engine.get_results().elapsed_wall_time == 1.0
    assert engine.get_results().stop_reason == "max_wall_time_reached"
    assert [
        record.levelname for record in caplog.records if record.levelname == "WARNING"
    ] == ["WARNING"]


def test_sample_for_time_warning_preserves_threshold_precision(monkeypatch, caplog):
    engine = _engine_with_block_size()
    engine.max_wall_time = 0.01
    times = iter([10.0, 10.02])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    with (
        caplog.at_level("WARNING", logger="liesel.goose.engine"),
        pytest.raises(TimeoutError),
    ):
        engine.sample_for_time(0.02)

    assert "max_wall_time 0.01 is shorter than wall_time 0.02" in caplog.text


@pytest.mark.parametrize(
    ("max_wall_time", "elapsed"),
    [(1.0, 1.0), (1.5, 2.0)],
)
def test_sample_for_time_keeps_normal_horizon_in_control(
    monkeypatch, caplog, max_wall_time, elapsed
):
    engine = _engine_with_block_size()
    engine.max_wall_time = max_wall_time
    times = iter([10.0, 10.0 + elapsed])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    with caplog.at_level("WARNING", logger="liesel.goose.engine"):
        engine.sample_for_time(1.0)

    assert engine.get_results().elapsed_wall_time == elapsed
    assert engine.get_results().stop_reason == "wall_time_reached"
    assert not [record for record in caplog.records if record.levelname == "WARNING"]


@pytest.mark.parametrize(
    "resume_method", ["sample_next_epoch", "sample_all_epochs", "sample_for_time"]
)
def test_timed_partial_epoch_resumes_through_public_sampler(monkeypatch, resume_method):
    engine = _engine_with_block_size()
    times = iter([10.0, 11.0])
    with monkeypatch.context() as patch:
        patch.setattr("liesel.goose.engine.monotonic", lambda: next(times))
        engine.sample_for_time(1.0)

    if resume_method == "sample_for_time":
        engine.sample_for_time(100.0)
    else:
        getattr(engine, resume_method)()

    samples = engine.get_results().get_samples()["x"]
    assert np.array_equal(samples[0], np.array([0, 10000, 10001, 10002, 10003]))
    assert engine.is_sampling_done()


def test_timed_stop_on_final_block_finalizes_adaptation(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.FAST_ADAPTATION, 2, 1, None),
        ]
    )
    times = iter([10.0, 10.5, 11.0])
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: next(times))

    engine.sample_for_time(1.0)

    assert engine.is_sampling_done()
    assert np.array_equal(
        engine.get_results().get_tuning_times().unwrap(), np.array([[3], [3]])
    )
    assert engine.get_results().elapsed_wall_time == 1.0
    assert engine.get_results().stop_reason == "wall_time_reached"


def test_untimed_sample_all_epochs_synchronizes_once_at_top_level(monkeypatch):
    engine = _engine_with_block_size(
        epoch_configs=[
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.BURNIN, 2, 1, None),
            EpochConfig(EpochType.POSTERIOR, 2, 1, None),
        ]
    )
    elapsed = 0.0
    real_block_until_ready = jax.block_until_ready

    def block_until_ready(value):
        nonlocal elapsed
        result = real_block_until_ready(value)
        elapsed += 1.0
        return result

    monkeypatch.setattr("liesel.goose.engine.jax.block_until_ready", block_until_ready)
    monkeypatch.setattr("liesel.goose.engine.monotonic", lambda: elapsed)

    engine.sample_all_epochs()

    assert engine.get_results().elapsed_wall_time == 1.0
    assert engine.get_results().stop_reason == "completed"


def t_test_engine():
    num_chains = 4
    epoch_configs = [
        EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
        EpochConfig(EpochType.FAST_ADAPTATION, 50, 1, None),
        EpochConfig(EpochType.BURNIN, 50, 1, None),
        EpochConfig(EpochType.POSTERIOR, 100, 1, None),
    ]

    ms = {"x": jnp.array(1), "y": jnp.array(-1)}
    mss = _stack_for_multi([ms for _ in range(num_chains)])
    con = DictInterface(lambda ms: -0.5 * ms["x"] ** 2 - 0.5 * ms["y"])
    ker0 = DetCountingKernel(["x"], DetCountingKernelState.default())
    ker1 = DetCountingKernel(["y"], DetCountingKernelState.default())
    ker0.set_model(con)
    ker1.set_model(con)

    ks = KernelSequence([ker0, ker1])

    seeds = jax.random.split(jax.random.PRNGKey(0), num_chains)

    engine = Engine(
        seeds,
        mss,
        ks,
        epoch_configs,
        25,
        con,
        ["x"],
        minimize_transition_infos=False,
        store_kernel_states=True,
        quantity_generators=[FooQauntGen("foo"), FooQauntGen("bar")],
    )

    engine.sample_all_epochs()

    results: SamplingResults = engine.get_results()

    print(results.positions.combine_all())
    print(results.transition_infos.combine_all())
    print(results.kernel_states.unwrap().combine_all())

    print(results.get_posterior_samples())
    print(results.get_tuning_times())

    print(results.generated_quantities.unwrap().combine_all().unwrap())


def test_model_states():
    builder = EngineBuilder(seed=1, num_chains=2)
    builder.set_model(DictInterface(lambda ms: -0.5 * ms["x"] ** 2))
    builder.set_initial_values({"x": jnp.array(0), "unmonitored": jnp.array(7)})
    builder.add_kernel(DetCountingKernel(["x"], DetCountingKernelState.default()))
    builder.set_epochs([EpochConfig(EpochType.BURNIN, 4, 3, None)])
    engine = builder.build()

    initial = engine.model_states
    np.testing.assert_array_equal(initial["x"], [0, 0])
    initial["unmonitored"] = jnp.array([99, 99])
    np.testing.assert_array_equal(engine.model_states["unmonitored"], [7, 7])

    engine.sample_all_epochs()

    np.testing.assert_array_equal(initial["x"], [0, 0])
    np.testing.assert_array_equal(engine.model_states["x"], [10003, 10003])
    np.testing.assert_array_equal(engine.model_states["unmonitored"], [7, 7])
    samples = engine.get_results().get_samples()
    assert "unmonitored" not in samples
    np.testing.assert_array_equal(samples["x"][:, -1], [10002, 10002])

    survivor = slice_leaves(engine.model_states, jnp.array([1]))
    np.testing.assert_array_equal(survivor["x"], [10003])
    np.testing.assert_array_equal(survivor["unmonitored"], [7])


def test_liesel_model_in_engine_builder() -> None:
    builder = EngineBuilder(seed=1, num_chains=4)
    y = Var.new_obs(1.0, name="y")
    model = Model([y])

    with pytest.raises(TypeError):
        builder.set_model(model)


def test_set_kernel_states():
    builder = EngineBuilder(seed=1, num_chains=2)
    builder.set_model(DictInterface(lambda ms: -0.5 * ms["x"] ** 2))
    builder.set_initial_values({"x": jnp.array(0)})
    builder.add_kernel(DetCountingKernel(["x"], DetCountingKernelState.default()))
    builder.set_epochs([EpochConfig(EpochType.POSTERIOR, 3, 1, None)] * 2)
    engine = builder.build()

    states = engine.kernel_states
    states[0].increment_per_transition = jnp.array([2, 3])
    np.testing.assert_array_equal(engine.kernel_states[0].increment_per_transition, 1)
    engine.set_kernel_states(states)
    states[0].increment_per_transition = jnp.array([99, 99])

    engine.sample_next_epoch()  # Initial values.
    engine.sample_next_epoch()
    np.testing.assert_array_equal(
        engine.get_results().get_posterior_samples()["x"],
        [[10000, 10002, 10004], [10000, 10003, 10006]],
    )

    states = engine.kernel_states
    states[0].increment_per_transition = jnp.array([4, 5])
    engine.set_kernel_states(states)
    for invalid in (
        [],
        slice_leaves(states, jnp.array([0])),
        jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), states),
    ):
        with pytest.raises(ValueError):
            engine.set_kernel_states(invalid)
    np.testing.assert_array_equal(
        engine.kernel_states[0].increment_per_transition, [4, 5]
    )

    engine._start_epoch()
    with pytest.raises(RuntimeError, match="active epoch"):
        engine.set_kernel_states(states)
    engine._kernel_start_epoch()
    engine._sample_for_duration(3)
    engine._end_epoch()
    np.testing.assert_array_equal(
        engine.get_results().get_posterior_samples()["x"][:, 3:],
        [[20000, 20004, 20008], [20000, 20005, 20010]],
    )


def t_test_engine_builder() -> None:
    builder = EngineBuilder(seed=1, num_chains=4)

    builder.set_epochs(
        [
            EpochConfig(EpochType.INITIAL_VALUES, 1, 1, None),
            EpochConfig(EpochType.FAST_ADAPTATION, 50, 1, None),
            EpochConfig(EpochType.BURNIN, 55, 10, None),
            EpochConfig(EpochType.POSTERIOR, 100, 10, None),
        ]
    )
    ms = {"x": jnp.array(1), "y": jnp.array(-1)}
    builder.set_initial_values(ms, multiple_chains=False)
    builder.set_jitter_fns(
        {
            "x": lambda key, cv: cv + tfd.Uniform(-1.0, 1.0).sample(cv.shape, key),
            "y": lambda key, cv: cv + tfd.Uniform(-1.0, 1.0).sample(cv.shape, key),
        }
    )
    con = DictInterface(lambda ms: -0.5 * ms["x"] ** 2 - 0.5 * ms["y"])
    builder.set_model(con)
    builder.add_kernel(DetCountingKernel(["x"], DetCountingKernelState.default()))
    builder.add_kernel(DetCountingKernel(["y"], DetCountingKernelState.default()))
    builder.add_quantity_generator(FooQauntGen("foo"))
    builder.add_quantity_generator(FooQauntGen("bar"))
    builder.positions_excluded = ["y"]
    engine = builder.build()

    engine.sample_all_epochs()
    results: SamplingResults = engine.get_results()

    # print(results.get_posterior_samples())
    # print(results.get_tuning_times())

    # print(results.generated_quantities.unwrap().combine_all().unwrap())
    # print(results.transition_infos.combine_all().unwrap())

    # test thinning worked
    assert results.get_posterior_samples()["x"].shape == (4, 10)
    assert results.get_samples()["x"].shape == (4, 66)
    assert results.generated_quantities.unwrap().combine_all().unwrap()["foo"].result[
        0
    ].shape == (4, 66)

    # test thinning is not applied to TIs
    assert results.transition_infos.combine_all().unwrap()[
        "kernel_01"
    ].error_code.shape == (4, 205)


if __name__ == "__main__":
    t_test_engine_builder()


## helper functions
def _stack_for_multi(chunks: list):
    chunks = slice_leaves(chunks, jnp.s_[jnp.newaxis, ...])
    return concatenate_leaves(chunks, axis=0)
