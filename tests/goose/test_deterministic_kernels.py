from jax.random import PRNGKey

from liesel.goose.epoch import EpochConfig, EpochType
from liesel.goose.models import DictModel
from liesel.goose.types import Kernel

from .deterministic_kernels import (
    DetCountingKernel,
    DetCountingKernelState,
    DetCountingKernelTuningInfo,
    DetCountingTransInfo,
)


def test_counting_kernel() -> None:
    mstate = {"x": 0}
    kstate = DetCountingKernelState.default()
    model = DictModel(lambda state: 0.0)
    kernel: Kernel[
        DetCountingKernelState, DetCountingTransInfo, DetCountingKernelTuningInfo
    ] = DetCountingKernel(["x"], kstate)
    kernel.set_model(model)

    configs = [
        EpochConfig(EpochType.FAST_ADAPTATION, 3, 1, None),
        EpochConfig(EpochType.FAST_ADAPTATION, 5, 1, None),
        EpochConfig(EpochType.POSTERIOR, 7, 1, None),
    ]

    prng_key = PRNGKey(0)
    kstate = kernel.init_state(prng_key, mstate)

    # epoch 0
    epoch = configs[0].to_state(0, 0)
    kstate = kernel.start_epoch(prng_key, kstate, mstate, epoch)
    for _ in range(epoch.config.duration):
        epoch.time_in_epoch += 1
        res = kernel.transition(prng_key, kstate, mstate, epoch)
        mstate = res.model_state
        kstate = res.kernel_state
    kstate = kernel.end_epoch(prng_key, kstate, mstate, epoch)
    tuning_res = kernel.tune(prng_key, kstate, mstate, epoch, None)
    kstate = tuning_res.kernel_state
    _ = tuning_res.info

    # epoch 1
    epoch = configs[1].to_state(1, 3)
    kstate = kernel.start_epoch(prng_key, kstate, mstate, epoch)
    for _ in range(epoch.config.duration):
        epoch.time_in_epoch += 1
        res = kernel.transition(prng_key, kstate, mstate, epoch)
        mstate = res.model_state
        kstate = res.kernel_state
    kstate = kernel.end_epoch(prng_key, kstate, mstate, epoch)
    tuning_res = kernel.tune(prng_key, kstate, mstate, epoch, None)
    kstate = tuning_res.kernel_state
    tune_info1 = tuning_res.info

    # finalize warmup
    assert not kstate.warmup_finialized
    fin_warm = kernel.end_warmup(prng_key, kstate, mstate, tune_info1)
    kstate = fin_warm.kernel_state
    assert kstate.warmup_finialized
    assert kstate.total_adaptive_transitions == 8

    # epoch 2
    epoch = configs[2].to_state(2, 8)
    kstate = kernel.start_epoch(prng_key, kstate, mstate, epoch)
    for _ in range(epoch.config.duration):
        epoch.time_in_epoch += 1
        res = kernel.transition(prng_key, kstate, mstate, epoch)
        mstate = res.model_state
        kstate = res.kernel_state

    assert kstate.in_epoch
    assert kstate.epoch_counter == 3
    kstate = kernel.end_epoch(prng_key, kstate, mstate, epoch)
    assert not kstate.in_epoch

    assert kstate.tune_counter == 2
    assert kstate.total_transitions == 15
    assert kstate.total_adaptive_transitions == 8
    assert int(mstate["x"]) == 30_007
