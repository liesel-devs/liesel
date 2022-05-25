import pytest
from model_lm import run_kernel_test

import liesel.goose as gs
from liesel.goose.nuts import NUTSKernelState, NUTSTransitionInfo, NUTSTuningInfo
from liesel.goose.types import Kernel


def type_check() -> None:
    kernel = gs.NUTSKernel(["beta", "log_sigma"])
    _: Kernel[NUTSKernelState, NUTSTransitionInfo, NUTSTuningInfo] = kernel


@pytest.mark.mcmc
def test_nuts(mcmc_seed):
    kernel = gs.NUTSKernel(["beta", "log_sigma"])
    run_kernel_test(mcmc_seed, [kernel])
