import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_lm import run_kernel_test
from scipy.stats import norm

import liesel.goose as gs
from liesel.goose.iwls import IWLSTransitionInfo, IWLSTuningInfo
from liesel.goose.types import Kernel, KernelState


def test_iwls_draws_exact_gaussian_conditional_with_exact_precision():
    with jax.enable_x64():
        precision = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        mean = jnp.array([1.0, -2.0])
        model = gs.DictInterface(
            lambda s: -0.5 * (s["beta"] - mean) @ precision @ (s["beta"] - mean)
        )
        kernel = gs.IWLSKernel(
            ["beta"], chol_info_fn=lambda s: jnp.linalg.cholesky(precision)
        )
        kernel.set_model(model)
        epoch = gs.EpochConfig(gs.EpochType.POSTERIOR, 1, 1, None).to_state(0, 0)
        key = jax.random.PRNGKey(370)
        starts = [jnp.array([-3.0, 2.0]), jnp.array([4.0, -5.0])]
        draws = []
        for start in starts:
            state = {"beta": start}
            outcome = jax.jit(kernel.transition)(
                key, kernel.init_state(key, state), state, epoch
            )
            assert outcome.info.acceptance_prob == pytest.approx(1.0, abs=1e-12)
            draws.append(outcome.model_state["beta"])
        np.testing.assert_allclose(draws[0], draws[1], atol=1e-12)

        state = {"beta": starts[0]}
        kernel_state = kernel.init_state(key, state)
        sample = jax.jit(
            jax.vmap(lambda k: kernel.transition(k, kernel_state, state, epoch))
        )(jax.random.split(key, 8192)).model_state["beta"]
        np.testing.assert_allclose(sample.mean(axis=0), mean, atol=0.04)
        np.testing.assert_allclose(
            np.cov(sample.T), np.linalg.inv(precision), atol=0.04
        )


def type_check() -> None:
    kernel = gs.IWLSKernel(["beta", "log_sigma"])
    _: Kernel[KernelState, IWLSTransitionInfo, IWLSTuningInfo] = kernel


def test_iwls_reverse_density_uses_proposed_position_and_regularized_precision():
    with jax.enable_x64():
        kernel = gs.IWLSKernel(["x"])
        kernel.set_model(
            gs.DictInterface(lambda s: -0.5 * s["x"] ** 2 - s["x"] ** 4 / 4)
        )
        state = {"x": jnp.array(0.5)}
        key = jax.random.PRNGKey(17)
        epoch = gs.EpochConfig(gs.EpochType.POSTERIOR, 1, 1, None).to_state(0, 0)
        outcome = jax.jit(kernel.transition)(
            key, kernel.init_state(key, state), state, epoch
        )

        # Closed-form quartic score/curvature, evaluated independently with SciPy.
        forward_precision = 1.75000175
        forward_mean = 0.5 - 0.625 / forward_precision
        draw_key, accept_key = jax.random.split(key)
        proposed = forward_mean + float(jax.random.normal(draw_key, (1,))[0]) / np.sqrt(
            forward_precision
        )
        reverse_precision = (1.0 + 3.0 * proposed**2) * 1.000001
        reverse_mean = proposed - (proposed + proposed**3) / reverse_precision
        correction = norm.logpdf(
            0.5, reverse_mean, 1 / np.sqrt(reverse_precision)
        ) - norm.logpdf(proposed, forward_mean, 1 / np.sqrt(forward_precision))
        log_ratio = -0.5 * proposed**2 - proposed**4 / 4 + 0.140625 + correction
        acceptance = min(1.0, np.exp(log_ratio))
        assert float(outcome.info.acceptance_prob) == pytest.approx(
            acceptance, abs=1e-12
        )
        expected = (
            proposed if float(jax.random.uniform(accept_key)) <= acceptance else 0.5
        )
        assert float(outcome.model_state["x"]) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "fallback,code", [("identity", 2), (None, 91), ("chol_of_modified_info", 3)]
)
def test_iwls_reports_precision_fallbacks(fallback, code):
    kernel = gs.IWLSKernel(["x"], fallback_chol_info=fallback)
    kernel.set_model(gs.DictInterface(lambda s: 0.5 * s["x"] ** 2 - s["x"] ** 4 / 4))
    state = {"x": jnp.array(0.0)}
    key = jax.random.PRNGKey(370)
    epoch = gs.EpochConfig(gs.EpochType.POSTERIOR, 1, 1, None).to_state(0, 0)
    outcome = jax.jit(kernel.transition)(
        key, kernel.init_state(key, state), state, epoch
    )
    assert int(outcome.info.error_code) == code
    assert np.isfinite(outcome.info.acceptance_prob)
    if fallback is None:
        assert not outcome.info.position_moved
        assert float(outcome.model_state["x"]) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_step_size": 1.0},
        {"da_tune_step_size": False},
        {"da_target_accept": 0.8},
    ],
)
def test_iwls_rejects_removed_tuning_arguments(kwargs):
    with pytest.raises(TypeError):
        gs.IWLSKernel(["x"], **kwargs)


def test_iwls_rejects_old_positional_step_size():
    with pytest.raises(TypeError):
        gs.IWLSKernel(["x"], None, 0.01)  # ty: ignore[too-many-positional-arguments]


def test_iwls_untuned_and_adaptation_epochs_use_the_same_proposal():
    key = jax.random.PRNGKey(370)
    state = {"x": jnp.array(0.5)}
    outcomes = []
    for kernel in (gs.IWLSKernel(["x"]), gs.IWLSKernel.untuned(["x"])):
        kernel.set_model(gs.DictInterface(lambda s: -0.5 * s["x"] ** 2))
        for kind in (gs.EpochType.POSTERIOR, gs.EpochType.FAST_ADAPTATION):
            epoch = gs.EpochConfig(kind, 1, 1, None).to_state(0, 0)
            ks = kernel.start_epoch(key, kernel.init_state(key, state), state, epoch)
            out = jax.jit(kernel.transition)(key, ks, state, epoch)
            ks = kernel.end_epoch(key, out.kernel_state, out.model_state, epoch)
            outcomes.append((out.info, out.model_state, ks))
    for out in outcomes[1:]:
        for a, e in zip(
            jax.tree.leaves(out), jax.tree.leaves(outcomes[0]), strict=True
        ):
            np.testing.assert_array_equal(a, e)


@pytest.mark.mcmc
def test_iwls(mcmc_seed):
    kernel = gs.IWLSKernel(["beta", "log_sigma"], identifier="my_id")
    results = run_kernel_test(mcmc_seed, [kernel])
    assert kernel.identifier in results.get_posterior_transition_infos()


@pytest.mark.mcmc
def test_iwls_scalar(mcmc_seed):
    kernel1 = gs.IWLSKernel(["beta"])
    kernel2 = gs.IWLSKernel(["log_sigma"])
    run_kernel_test(mcmc_seed, [kernel1, kernel2])


@pytest.mark.mcmc
def test_iwls_untuned(mcmc_seed):
    kernel = gs.IWLSKernel.untuned(["beta", "log_sigma"])
    run_kernel_test(mcmc_seed, [kernel], test_da_target_accept=False)
