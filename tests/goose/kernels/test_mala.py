from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import liesel.goose as gs

FILES = Path(__file__).parents[1] / "files"


@pytest.mark.mcmc
@pytest.mark.parametrize("kernel_cls", [gs.MALAKernel, gs.SMMALAKernel])
def test_default_langevin_adaptation_and_posterior_sampling(kernel_cls):
    kernel = kernel_cls(["x"])
    builder = gs.EngineBuilder(370, num_chains=2)
    builder.show_progress = False
    builder.store_kernel_states = True
    mean = jnp.array([1.0, -2.0])
    builder.set_model(gs.DictInterface(lambda s: -0.5 * jnp.sum((s["x"] - mean) ** 2)))
    builder.set_initial_values({"x": jnp.zeros(2)})
    builder.add_kernel(kernel)
    builder.set_duration(
        warmup_duration=2000, posterior_duration=2000, term_duration=1500
    )
    engine = builder.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = np.asarray(results.get_posterior_samples()["x"]).reshape(-1, 2)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.1)
    np.testing.assert_allclose(np.cov(samples.T), np.eye(2), atol=0.1)
    infos = results.get_posterior_transition_infos()[kernel.identifier]
    assert np.mean(infos.acceptance_prob) == pytest.approx(
        kernel.da_target_accept, abs=0.06
    )
    states = results.kernel_states.unwrap().combine_all().unwrap()[0]
    steps = np.asarray(states.step_size)
    assert np.ptp(steps[:, :2000]) > 0.0
    assert np.all(np.ptp(steps[:, -2000:], axis=1) == 0.0)


@pytest.mark.parametrize("kernel_cls", [gs.MALAKernel, gs.SMMALAKernel])
@pytest.mark.parametrize(
    "kwargs,argument",
    [
        ({"initial_step_size": x}, "initial_step_size")
        for x in (0.0, -1.0, float("inf"), float("nan"))
    ]
    + [({"da_target_accept": x}, "da_target_accept") for x in (0.0, 1.0, float("nan"))],
)
def test_langevin_rejects_invalid_step_size_and_acceptance_target(
    kernel_cls, kwargs, argument
):
    with pytest.raises(ValueError, match=argument):
        kernel_cls(["x"], **kwargs)


@pytest.mark.parametrize(
    "kernel_cls,target,scale_ratio",
    [(gs.MALAKernel, 0.574, 100.0), (gs.SMMALAKernel, 0.8, 1.0)],
)
def test_automatic_initial_step_respects_proposal_geometry(
    kernel_cls, target, scale_ratio
):
    with jax.enable_x64():
        steps = []
        for sd in (0.1, 10.0):
            kernel = kernel_cls(["x"], da_tune_step_size=False)
            assert kernel.initial_step_size is None
            assert kernel.da_target_accept == target
            kernel.set_model(
                gs.DictInterface(lambda s, sd=sd: -0.5 * (s["x"] / sd) ** 2)
            )
            state = {"x": jnp.array(0.5 * sd)}
            key = jax.random.PRNGKey(370)
            initial = jax.jit(kernel.init_state)(key, state)
            step = float(initial.step_size)
            assert np.isfinite(step) and step > 0.0
            assert float(state["x"]) == 0.5 * sd
            epoch = gs.EpochConfig(gs.EpochType.FAST_ADAPTATION, 1, 1, None).to_state(
                0, 0
            )
            outcome = jax.jit(kernel.transition)(key, initial, state, epoch)
            assert float(outcome.kernel_state.step_size) == step
            steps.append(step)
        # The search doubles/halves s. Identity MALA follows the parameter scale;
        # Hessian-preconditioned SMMALA already accounts for that scale in P.
        assert 0.5 * scale_ratio <= steps[1] / steps[0] <= 2.0 * scale_ratio


def test_mala_transition_matches_standard_normal_langevin_proposal():
    kernel = gs.MALAKernel(["x"], initial_step_size=1.0, da_tune_step_size=False)
    kernel.set_model(gs.DictInterface(lambda s: -0.5 * s["x"] ** 2))
    state = {"x": jnp.array(1.0)}
    key = jax.random.PRNGKey(1)
    epoch = gs.EpochConfig(gs.EpochType.POSTERIOR, 1, 1, None).to_state(0, 0)
    outcome = jax.jit(kernel.transition)(
        key, kernel.init_state(key, state), state, epoch
    )

    # For N(0, 1), s=1 and x=1: q(y|x)=N(0.5, 1), log(alpha)=(1-y^2)/8.
    proposal_key, accept_key = jax.random.split(key)
    proposed = 0.5 + float(jax.random.normal(proposal_key, (1,))[0])
    acceptance = min(1.0, np.exp((1.0 - proposed**2) / 8.0))
    accepted = float(jax.random.uniform(accept_key)) <= acceptance
    assert float(outcome.info.acceptance_prob) == pytest.approx(acceptance)
    assert bool(outcome.info.position_moved) == accepted
    assert float(outcome.model_state["x"]) == pytest.approx(
        proposed if accepted else 1.0
    )


def test_mala_accepts_a_model_with_only_first_derivatives():
    @jax.custom_jvp
    def score(x):
        return -x

    @score.defjvp
    def no_second_derivative(primals, tangents):
        raise AssertionError("This model does not supply a Hessian")

    @jax.custom_vjp
    def log_density(x):
        return -0.5 * x**2

    log_density.defvjp(
        lambda x: (-0.5 * x**2, x), lambda x, cotangent: (cotangent * score(x),)
    )
    kernel = gs.MALAKernel(["x"])
    kernel.set_model(gs.DictInterface(lambda s: log_density(s["x"])))
    state = {"x": jnp.array(0.5)}
    key = jax.random.PRNGKey(370)
    epoch = gs.EpochConfig(gs.EpochType.POSTERIOR, 1, 1, None).to_state(0, 0)
    ks = jax.jit(kernel.init_state)(key, state)
    outcome = jax.jit(kernel.transition)(key, ks, state, epoch)
    assert np.isfinite(outcome.info.acceptance_prob)
    assert int(outcome.info.error_code) == 0


def test_legacy_iwls_results_still_load_and_summarize():
    with jax.enable_x64():
        result = gs.SamplingResults.pkl_load(FILES / "iwls_0_5_adaptive.pkl")
        summary = gs.Summary(result)
        expected = np.asarray(result.get_posterior_samples()["x"]).mean()
        assert summary.quantities["mean"]["x"] == pytest.approx(expected)
        assert result.kernel_classes.unwrap()["kernel_00"] is gs.IWLSKernel
        assert not summary.to_dataframe().empty
        assert summary.error_df().empty


def legacy_run(kernel):
    """The public sampling setup used to capture the pre-0.6 IWLS fixtures."""
    builder = gs.EngineBuilder(370, num_chains=2)
    builder.show_progress = False
    builder.store_kernel_states = True
    builder.set_model(
        gs.DictInterface(lambda s: -0.5 * s["x"] ** 2 - 0.05 * s["x"] ** 4)
    )
    builder.set_initial_values({"x": jnp.array(0.5)})
    builder.add_kernel(kernel)
    builder.set_epochs(
        [
            gs.EpochConfig(gs.EpochType.FAST_ADAPTATION, 12, 1, None),
            gs.EpochConfig(gs.EpochType.POSTERIOR, 32, 1, None),
        ]
    )
    engine = builder.build()
    engine.sample_all_epochs()
    return engine.get_results()


@pytest.mark.parametrize("untuned", [False, True])
def test_smmala_preserves_legacy_sampling_and_adaptation(untuned):
    with jax.enable_x64():
        kernel = (
            gs.SMMALAKernel.untuned(["x"])
            if untuned
            else gs.SMMALAKernel(["x"], initial_step_size=0.01)
        )
        actual = legacy_run(kernel)
        name = "iwls_0_5_untuned.pkl" if untuned else "iwls_0_5_adaptive.pkl"
        expected = gs.SamplingResults.pkl_load(FILES / name)
        for actual_tree, expected_tree in (
            (actual.get_samples(), expected.get_samples()),
            (
                actual.transition_infos.combine_all().unwrap(),
                expected.transition_infos.combine_all().unwrap(),
            ),
            (
                actual.kernel_states.unwrap().combine_all().unwrap(),
                expected.kernel_states.unwrap().combine_all().unwrap(),
            ),
        ):
            for a, e in zip(
                jax.tree.leaves(actual_tree),
                jax.tree.leaves(expected_tree),
                strict=True,
            ):
                # Allow platform rounding; proposals and adaptation must agree.
                np.testing.assert_allclose(a, e, rtol=1e-10, atol=1e-12)
