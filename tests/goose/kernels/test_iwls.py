import pytest
from model_lm import run_kernel_test

import liesel.goose as gs
from liesel.goose.iwls import IWLSKernelState, IWLSTransitionInfo, IWLSTuningInfo
from liesel.goose.types import Kernel


def type_check() -> None:
    kernel = gs.IWLSKernel(["beta", "log_sigma"])
    _: Kernel[IWLSKernelState, IWLSTransitionInfo, IWLSTuningInfo] = kernel


@pytest.mark.mcmc
def test_nuts(mcmc_seed):
    kernel = gs.IWLSKernel(["beta", "log_sigma"])
    run_kernel_test(mcmc_seed, [kernel])
