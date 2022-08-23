import jax.random
import jax.scipy.stats
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


def proposal_asym_fn(
    key: KeyArray, model_state: ModelState, step_size: float
) -> MHProposal:
    key0, key1 = jax.random.split(key)

    mean_prop = model_state["beta"] + step_size
    beta = mean_prop + step_size * jax.random.normal(key0, model_state["beta"].shape)
    log_d_prop = jax.scipy.stats.norm.logpdf(beta, loc=mean_prop, scale=step_size).sum()

    mean_old = beta + step_size
    log_d_old = jax.scipy.stats.norm.logpdf(
        model_state["beta"], loc=mean_old, scale=step_size
    ).sum()

    log_sigma = model_state["log_sigma"] + step_size * jax.random.normal(
        key1, model_state["log_sigma"].shape
    )

    log_correction = log_d_old - log_d_prop

    return MHProposal(
        position=Position({"beta": beta, "log_sigma": log_sigma}),
        log_correction=log_correction,
    )


def type_check() -> None:
    kernel = gs.MHKernel(["beta", "log_sigma"], proposal_fn)
    _: Kernel[RWKernelState, MHTransitionInfo, MHTuningInfo] = kernel


@pytest.mark.mcmc
def test_mh_kernel_symmetric(mcmc_seed) -> None:
    kernel = gs.MHKernel(["beta", "log_sigma"], proposal_fn, da_tune_step_size=True)
    run_kernel_test(mcmc_seed, [kernel])


@pytest.mark.mcmc
def test_mh_kernel_asymmetric(mcmc_seed) -> None:
    kernel = gs.MHKernel(
        ["beta", "log_sigma"],
        proposal_asym_fn,
        da_tune_step_size=True,
        da_target_accept=0.1,
    )
    run_kernel_test(mcmc_seed, [kernel])
