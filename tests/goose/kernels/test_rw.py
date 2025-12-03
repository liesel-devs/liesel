from model_lm import run_kernel_test

import liesel.goose as gs
from liesel.goose.rw import RWKernelState, RWTransitionInfo, RWTuningInfo
from liesel.goose.types import Kernel


def type_check() -> None:
    kernel = gs.RWKernel(["beta", "log_sigma"])
    _: Kernel[RWKernelState, RWTransitionInfo, RWTuningInfo] = kernel


# @pytest.mark.mcmc
def test_rw(mcmc_seed):
    kernel = gs.RWKernel(["beta", "log_sigma"], identifier="test")
    results = run_kernel_test(mcmc_seed, [kernel])
    kernel.identifier in results.get_posterior_transition_infos()
