from jax.random import PRNGKey

from liesel.goose.epoch import EpochConfig, EpochType
from liesel.goose.interface import DictInterface
from liesel.goose.kernel_sequence import KernelSequence, KernelStates
from liesel.goose.types import Kernel

from .deterministic_kernels import DetCountingKernel, DetCountingKernelState


def test_kernel_sequence() -> None:
    mstate = {"x": 0.0, "y": 0.0}
    kstate0 = DetCountingKernelState.default()
    kstate1 = DetCountingKernelState.default()
    model = DictInterface(lambda state: 0.0)
    ker0: Kernel = DetCountingKernel(["x"], kstate0)
    ker1: Kernel = DetCountingKernel(["y"], kstate1)
    ker0.set_model(model)
    ker1.set_model(model)
    ker0.identifier = "foo"
    ker1.identifier = "bar"
    kseq = KernelSequence([ker0, ker1])

    configs = [
        EpochConfig(EpochType.FAST_ADAPTATION, 3, 1, None),
        EpochConfig(EpochType.POSTERIOR, 5, 1, None),
    ]

    prng_key = PRNGKey(0)
    kstates: KernelStates = kseq.init_states(prng_key, mstate)

    # epoch 0
    epoch = configs[0].to_state(0, 0)
    kstates = kseq.start_epoch(prng_key, kstates, mstate, epoch)
    for _ in range(epoch.config.duration):
        epoch.time_in_epoch += 1
        res = kseq.transition(prng_key, kstates, mstate, epoch)
        mstate = res.model_state
        kstates = res.kernel_states
    kstates = kseq.end_epoch(prng_key, kstates, mstate, epoch)
    tuning_res = kseq.tune(prng_key, kstates, mstate, epoch, None)
    kstates = tuning_res.kernel_states
    tune_infos = tuning_res.infos

    # finalize warmup
    assert not kstates[0].warmup_finialized
    assert not kstates[1].warmup_finialized
    fin_warm = kseq.end_warmup(prng_key, kstates, mstate, tune_infos)
    kstates = fin_warm.kernel_states
    assert kstates[0].warmup_finialized
    assert kstates[1].warmup_finialized

    # epoch 1
    epoch = configs[1].to_state(1, 3)
    kstates = kseq.start_epoch(prng_key, kstates, mstate, epoch)
    for _ in range(epoch.config.duration):
        epoch.time_in_epoch += 1
        res = kseq.transition(prng_key, kstates, mstate, epoch)
        mstate = res.model_state
        kstates = res.kernel_states

    assert kstates[0].in_epoch
    assert kstates[1].epoch_counter == 2
    kstates = kseq.end_epoch(prng_key, kstates, mstate, epoch)
    assert not kstates[0].in_epoch

    assert kstates[0].tune_counter == 1
    assert kstates[1].tune_counter == 1

    assert kstates[0].total_transitions == 8
    assert kstates[1].total_transitions == 8

    assert int(mstate["x"]) == 20_005
    assert int(mstate["y"]) == 20_005
