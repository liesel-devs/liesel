import pytest
from model_lm import run_kernel_test

import liesel.goose as gs
from liesel.goose.iwls import IWLSKernelState, IWLSTransitionInfo, IWLSTuningInfo
from liesel.goose.types import Kernel


def type_check() -> None:
    kernel = gs.IWLSKernel(["beta", "log_sigma"])
    _: Kernel[IWLSKernelState, IWLSTransitionInfo, IWLSTuningInfo] = kernel


@pytest.mark.mcmc
def test_iwls(mcmc_seed):
    kernel = gs.IWLSKernel(["beta", "log_sigma"])
    run_kernel_test(mcmc_seed, [kernel])


@pytest.mark.mcmc
def test_iwls_scalar(mcmc_seed):
    kernel1 = gs.IWLSKernel(["beta"])
    kernel2 = gs.IWLSKernel(["log_sigma"])
    run_kernel_test(mcmc_seed, [kernel1, kernel2])


@pytest.mark.mcmc
def test_iwls_untuned(mcmc_seed):
    kernel = gs.IWLSKernel.untuned(["beta", "log_sigma"])
    run_kernel_test(mcmc_seed, [kernel], test_da_target_accept=False)
