import jax.random
import pytest
from model_lm import run_kernel_test

import liesel.goose as gs
from liesel.goose.mh_kernel import (
    MHProposal,
    MHTransitionInfo,
    MHTuningInfo,
    RWKernelState,
)
from liesel.goose.types import Kernel, KeyArray, ModelState, Position


def proposal_fn(key: KeyArray, model_state: ModelState, step_size: float) -> MHProposal:
    key0, key1 = jax.random.split(key)
    beta = model_state["beta"] + step_size * jax.random.normal(
        key0, model_state["beta"].shape
    )
    log_sigma = model_state["log_sigma"] + step_size * jax.random.normal(
        key1, model_state["log_sigma"].shape
    )
    return MHProposal(
        position=Position({"beta": beta, "log_sigma": log_sigma}), log_correction=0.0
    )


def type_check() -> None:
    kernel = gs.MHKernel(["beta", "log_sigma"], proposal_fn)
    _: Kernel[RWKernelState, MHTransitionInfo, MHTuningInfo] = kernel


@pytest.mark.mcmc
def test_mh_kernel(mcmc_seed) -> None:
    kernel = gs.MHKernel(["beta", "log_sigma"], proposal_fn, da_tune_step_size=True)
    run_kernel_test(mcmc_seed, [kernel])
