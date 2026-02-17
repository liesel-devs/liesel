import pytest
from model_lm import run_kernel_test

import liesel.goose as gs
from liesel.goose.hmc import HMCKernelState, HMCTransitionInfo, HMCTuningInfo
from liesel.goose.types import Kernel


def type_check() -> None:
    kernel = gs.HMCKernel(["beta", "log_sigma"])
    _: Kernel[HMCKernelState, HMCTransitionInfo, HMCTuningInfo] = kernel


@pytest.mark.mcmc
def test_hmc(mcmc_seed):
    kernel = gs.HMCKernel(["beta", "log_sigma"], identifier="test")
    results = run_kernel_test(mcmc_seed, [kernel])
    kernel.identifier in results.get_posterior_transition_infos()
