import pytest
from model_lm import run_kernel_test

import liesel.goose as gs
from liesel.goose.rw import RWKernelState, RWTransitionInfo, RWTuningInfo
from liesel.goose.types import Kernel


def type_check() -> None:
    kernel = gs.RWKernel(["beta", "log_sigma"])
    _: Kernel[RWKernelState, RWTransitionInfo, RWTuningInfo] = kernel


@pytest.mark.mcmc
def test_rw(mcmc_seed):
    kernel = gs.RWKernel(["beta", "log_sigma"])
    run_kernel_test(mcmc_seed, [kernel])
