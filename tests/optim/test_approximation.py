"""Named Gaussian blocks retain their marginal or conditional interpretation."""

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.experimental import io_callback

import liesel.model as lsl
import liesel.optim as opt
from liesel.optim.types import Position


def gaussian_approximation():
    # For independent standard normals e, the coordinates are
    # (e0-e1+e2-e3, e1-e2+e3, e2-e3, e3). Their covariances follow directly.
    return opt.LaplaceApproximation(
        mean=Position(
            {"z": jnp.array(0.0), "matrix": jnp.zeros((1, 1)), "a": jnp.zeros(2)}
        ),
        precision_cholesky=jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ),
        names=("z", "matrix", "a"),
        shapes=((), (1, 1), (2,)),
        valid=True,
        diagnostics={},
    )


def test_sample_requires_keyword_seed_and_defaults_to_one_draw():
    posterior = gaussian_approximation()
    key = jax.random.key(42)
    single = posterior.sample(seed=key)
    assert {name: value.shape for name, value in single.items()} == {
        "z": (),
        "matrix": (1, 1),
        "a": (2,),
    }
    legacy_sample: Any = posterior.sample
    with pytest.raises(TypeError, match="seed"):
        legacy_sample(key)


@pytest.mark.parametrize("shape", [1000, [1000]])
def test_sample_accepts_integer_and_sequence_shapes_with_identical_draws(shape):
    posterior = gaussian_approximation()
    key = jax.random.key(42)
    draws = posterior.sample(shape, seed=key)
    tuple_draws = posterior.sample((1000,), seed=key)
    assert {name: value.shape for name, value in draws.items()} == {
        "z": (1000,),
        "matrix": (1000, 1, 1),
        "a": (1000, 2),
    }
    for name, value in draws.items():
        np.testing.assert_array_equal(value, tuple_draws[name])


def test_marginal_covariance_blocks_include_dependence_on_other_parameters():
    posterior = gaussian_approximation()
    blocks = posterior.marginal_covariance_blocks()
    assert list(blocks) == ["z", "matrix", "a"]
    np.testing.assert_allclose(blocks["z"], [[4.0]])
    np.testing.assert_allclose(blocks["matrix"], [[3.0]])
    np.testing.assert_allclose(blocks["a"], [[2.0, -1.0], [-1.0, 1.0]])

    selected = posterior.marginal_covariance_blocks(["a", "z"])
    assert list(selected) == ["a", "z"]
    np.testing.assert_allclose(selected["a"], [[2.0, -1.0], [-1.0, 1.0]])


def test_marginal_precision_factors_invert_marginal_covariance_blocks():
    posterior = gaussian_approximation()
    factors = posterior.marginal_precision_cholesky_blocks()
    np.testing.assert_allclose(factors["z"], [[0.5]], atol=1e-6)
    np.testing.assert_allclose(factors["matrix"], [[1 / np.sqrt(3)]], atol=1e-6)
    np.testing.assert_allclose(factors["a"], [[1.0, 0.0], [1.0, 1.0]], atol=1e-6)
    assert list(posterior.marginal_precision_cholesky_blocks(["matrix"])) == ["matrix"]


def test_conditional_precision_blocks_include_cross_block_factor_entries():
    posterior = gaussian_approximation()
    blocks = posterior.conditional_precision_blocks()
    assert list(blocks) == ["z", "matrix", "a"]
    np.testing.assert_allclose(blocks["z"], [[1.0]])
    np.testing.assert_allclose(blocks["matrix"], [[2.0]])
    np.testing.assert_allclose(blocks["a"], [[2.0, 1.0], [1.0, 2.0]])
    assert list(posterior.conditional_precision_blocks(["a", "z"])) == ["a", "z"]


@pytest.mark.parametrize(
    "method",
    [
        "marginal_covariance_blocks",
        "marginal_precision_cholesky_blocks",
        "conditional_precision_blocks",
    ],
)
def test_block_selection_rejects_ambiguous_or_unknown_names(method):
    select = getattr(gaussian_approximation(), method)
    assert select([]) == {}
    for keys, message in [
        ("z", "sequence"),
        (["z", "z"], "Duplicate"),
        (["missing"], "Unknown"),
    ]:
        with pytest.raises(ValueError, match=message):
            select(keys)


@pytest.mark.parametrize("x64", [False, True])
def test_blocks_preserve_precision_and_reject_invalid_approximations(x64):
    with jax.enable_x64(x64):
        posterior = gaussian_approximation()
        for method in [
            "marginal_covariance_blocks",
            "marginal_precision_cholesky_blocks",
            "conditional_precision_blocks",
        ]:
            blocks = getattr(posterior, method)()
            assert all(
                block.dtype == posterior.precision_cholesky.dtype
                for block in blocks.values()
            )
            for invalid in [
                replace(posterior, valid=False),
                replace(posterior, precision_cholesky=None),
            ]:
                with pytest.raises(RuntimeError, match="invalid"):
                    getattr(invalid, method)()


@pytest.mark.parametrize("scale_loss", [False, True])
def test_normal_regression_posterior_is_exact_independent_of_loss_scaling(scale_loss):
    with jax.enable_x64():
        beta = lsl.Var.new_param(
            jnp.zeros(2), lsl.Dist(tfd.Normal, 0.0, 1.0), name="beta"
        )
        x = lsl.Var.new_value(jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]), name="x")
        loc = lsl.Var.new_calc(jnp.dot, x, beta)
        y = lsl.Var.new_obs(
            jnp.array([1.0, 2.0, 2.0]), lsl.Dist(tfd.Normal, loc, 1.0), name="y"
        )
        model = lsl.Model(y, to_float32=False)
        optim = opt.LieselOptim(
            model,
            optimizers="lbfgs",
            scale_loss=scale_loss,
            loss_monitor="train_full_data",
            show_progress=False,
            stopper=opt.Stopper(epochs=20, patience=5, rtol=1e-10),
        )
        result = optim.fit()
        posterior = optim.loss.approximate_joint_posterior(result)
        assert isinstance(posterior, opt.LaplaceApproximation)
        assert posterior.valid
        assert posterior.names == ("beta",)
        assert posterior.shapes == ((2,),)
        # Completing the square gives Q=[[4,3],[3,6]], mean=(4/5,3/5).
        np.testing.assert_allclose(posterior.mean["beta"], [0.8, 0.6], atol=1e-7)
        np.testing.assert_allclose(
            posterior.covariance(), [[0.4, -0.2], [-0.2, 4 / 15]], atol=1e-10
        )
        for field, value in [
            ("at", "best"),
            ("stationarity_tol", 0.0),
            ("stationarity_tol", True),
            ("stationarity_tol", float("inf")),
            ("raise_on_failure", 1),
        ]:
            options: dict[str, Any] = {"raise_on_failure": False, field: value}
            with pytest.raises(ValueError, match=field):
                optim.loss.approximate_joint_posterior(result, **options)


@pytest.mark.parametrize("loss_kind", ["laplace", "joint"])
@pytest.mark.parametrize("holdout", [False, True])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("precomputed", [False, True])
def test_callback_basis_is_prepared_for_fitting_and_joint_uncertainty(
    loss_kind, holdout, x64, precomputed
):
    with jax.enable_x64(x64):
        calls = []

        def square(values):
            calls.append(np.asarray(values).copy())
            return np.square(values)

        def basis_fn(values):
            return io_callback(
                square,
                jax.ShapeDtypeStruct(values.shape, values.dtype),
                values,
                ordered=True,
            )

        x = lsl.Var.new_value(jnp.arange(1.0, 5.0), name="x")
        basis = lsl.Var.new_calc(basis_fn, x, name="basis")
        alpha = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="alpha")
        beta = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="beta")
        loc = lsl.Var.new_calc(lambda a, b, v: a + b * v, alpha, beta, basis)
        y = lsl.Var.new_obs(
            1 + 2 * x.value**2, lsl.Dist(tfd.Normal, loc, 1.0), name="y"
        )
        model = lsl.Model(y, to_float32=False)
        split = opt.PositionSplit.from_model(
            model,
            position_keys=["basis" if precomputed else "x", "y"],
            validate_axis_share=0.5 if holdout else 0.0,
            shuffle=False,
        )
        loss = (
            opt.LaplaceLoss(model, split, latent=["beta"])
            if loss_kind == "laplace"
            else opt.NegLogProbLoss(model, split)
        )
        optim = opt.LieselOptim(
            model,
            loss=loss,
            optimizers="lbfgs",
            loss_monitor="train_full_data",
            stopper=opt.Stopper(epochs=20, patience=3, rtol=1e-9),
            show_progress=False,
        )
        jax.effects_barrier()
        calls.clear()
        result = optim.fit()
        jax.effects_barrier()
        fitting_calls = len(calls)
        calls.clear()
        posterior = loss.approximate_joint_posterior(result)
        jax.effects_barrier()
        assert posterior.valid
        assert fitting_calls == (0 if precomputed else 1)
        assert len(calls) == (0 if precomputed else 1)
        if not precomputed:
            np.testing.assert_array_equal(calls[0], [1, 2] if holdout else [1, 2, 3, 4])

        # Completing the square gives the exact joint posterior. Integrating beta
        # is also exact here, so the Laplace and joint-MAP results must coincide.
        if holdout:
            mean = np.array([21, 57]) / 29
            covariance = np.array([[18, -5], [-5, 3]]) / 29
        else:
            mean = np.array([580, 1770]) / 875
            covariance = np.array([[355, -30], [-30, 5]]) / 875
        np.testing.assert_allclose(
            [posterior.mean["alpha"], posterior.mean["beta"]], mean, atol=1e-5
        )
        np.testing.assert_allclose(posterior.covariance(), covariance, atol=1e-6)


def test_validation_best_must_be_stationary_for_training_posterior():
    mu = lsl.Var.new_param(jnp.array(0.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="mu")
    y = lsl.Var.new_obs(
        jnp.array([2.0, 2.0, 0.0, 0.0, 10.0, 10.0]),
        lsl.Dist(tfd.Normal, mu, 1.0),
        name="y",
    )
    model = lsl.Model(y)
    split = opt.PositionSplit.from_model(
        model, validate_axis_share=1 / 3, test_axis_share=1 / 3, shuffle=False
    )
    optim = opt.LieselOptim(
        model,
        split=split,
        optimizers=optax.sgd(0.2),
        scale_loss=False,
        loss_monitor="validation",
        show_progress=False,
        stopper=opt.Stopper(epochs=20, patience=20),
    )
    result = optim.fit()
    assert result.min_monitor_epoch == 0
    with pytest.raises(RuntimeError, match="stationarity"):
        optim.loss.approximate_joint_posterior(result)
    failed = optim.loss.approximate_joint_posterior(result, raise_on_failure=False)
    assert not failed.valid
    assert failed.precision_cholesky is None
    np.testing.assert_allclose(failed.diagnostics["gradient"], [-1.6], atol=1e-6)
    np.testing.assert_allclose(failed.diagnostics["joint_precision"], [[3.0]])
    np.testing.assert_allclose(
        failed.diagnostics["newton_decrement_squared"], 2.56 / 3, atol=1e-6
    )
    posterior = optim.loss.approximate_joint_posterior(result, at="final")
    np.testing.assert_allclose(posterior.mean["mu"], 4 / 3, atol=1e-6)
    np.testing.assert_allclose(posterior.covariance(), [[1 / 3]], atol=1e-6)


@pytest.mark.parametrize("curvature", ["negative", "singular"])
def test_posterior_rejects_nonpositive_curvature(curvature):
    theta = lsl.Var.new_param(jnp.zeros(2), name="theta")
    if curvature == "negative":
        loc = lsl.Var.new_calc(jnp.square, theta)
        observed = jnp.ones(2)
        precision = [[-2.0, 0.0], [0.0, -2.0]]
    else:
        loc = lsl.Var.new_calc(jnp.sum, theta)
        observed = jnp.zeros(2)
        precision = [[2.0, 2.0], [2.0, 2.0]]
    y = lsl.Var.new_obs(observed, lsl.Dist(tfd.Normal, loc, 1.0), name="y")
    optim = opt.LieselOptim(
        lsl.Model(y),
        optimizers=optax.sgd(0.0),
        show_progress=False,
        loss_monitor="train_full_data",
        stopper=opt.Stopper(epochs=1, patience=1),
    )
    result = optim.fit()
    with pytest.raises(RuntimeError, match="curvature"):
        optim.loss.approximate_joint_posterior(result)
    failed = optim.loss.approximate_joint_posterior(result, raise_on_failure=False)
    assert not failed.valid
    assert failed.precision_cholesky is None
    np.testing.assert_allclose(failed.diagnostics["joint_precision"], precision)
    with pytest.raises(RuntimeError, match="invalid"):
        failed.sample(seed=jax.random.key(0))
    with pytest.raises(RuntimeError, match="invalid"):
        failed.covariance()


def test_ema_stopping_selects_stored_snapshots_and_uses_full_training_curvature():
    mu = lsl.Var.new_param(jnp.array(0.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="mu")
    y = lsl.Var.new_obs(jnp.full(2, 2.0), lsl.Dist(tfd.Normal, mu, 1.0), name="y")
    optim = opt.LieselOptim(
        lsl.Model(y),
        optimizers=optax.sgd(2.0),
        batch_size=1,
        loss_monitor=opt.EmaTrainLossMonitor(effective_window=2.0),
        stopper=opt.Stopper(epochs=10, patience=1),
        show_progress=False,
    )
    result = optim.fit()
    assert result.status == "early_stopping"
    assert result.monitor_source == "train_ema"
    assert result.min_monitor_epoch == 0
    np.testing.assert_allclose(result.position_min_monitor["mu"], -4.0)
    np.testing.assert_allclose(result.position_final["mu"], -20.0)
    for at, mean, gradient in [("min_monitor", -4.0, -16.0), ("final", -20.0, -64.0)]:
        failed = optim.loss.approximate_joint_posterior(
            result, at=at, raise_on_failure=False
        )
        assert not failed.valid
        np.testing.assert_allclose(failed.mean["mu"], mean)
        np.testing.assert_allclose(failed.diagnostics["gradient"], [gradient])
        # The deliberately loose threshold isolates snapshot selection here.
        posterior = optim.loss.approximate_joint_posterior(
            result, at=at, stationarity_tol=1000.0
        )
        np.testing.assert_allclose(posterior.mean["mu"], mean)
        np.testing.assert_allclose(posterior.covariance(), [[1 / 3]], atol=1e-6)
    default = optim.loss.approximate_joint_posterior(result, raise_on_failure=False)
    np.testing.assert_array_equal(default.mean["mu"], result.position_min_monitor["mu"])


def test_nonfinite_density_is_invalid_even_with_finite_gradient_and_curvature():
    mu = lsl.Var.new_param(jnp.array(0.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="mu")
    y = lsl.Var.new_obs(jnp.zeros(1), lsl.Dist(tfd.Normal, mu, 1.0), name="y")
    z = lsl.Var.new_obs(jnp.array([1e20]), lsl.Dist(tfd.Normal, 0.0, 1.0), name="z")
    optim = opt.LieselOptim(
        lsl.Model([y, z]),
        optimizers=optax.sgd(0.0),
        show_progress=False,
        loss_monitor="train_full_data",
        stopper=opt.Stopper(epochs=1, patience=1),
    )
    result = optim.fit()
    with pytest.raises(RuntimeError, match="Non-finite"):
        optim.loss.approximate_joint_posterior(result, at="final")
    failed = optim.loss.approximate_joint_posterior(
        result, at="final", raise_on_failure=False
    )
    assert not failed.valid
    assert not jnp.isfinite(failed.diagnostics["value"])
    np.testing.assert_allclose(failed.diagnostics["gradient"], [0.0])
    np.testing.assert_allclose(failed.diagnostics["joint_precision"], [[2.0]])
    missing = optim.loss.approximate_joint_posterior(result, raise_on_failure=False)
    assert not missing.valid
    assert "unavailable" in missing.diagnostics["reason"]


@pytest.mark.parametrize("incompatible", ["unknown", "shape", "discrete"])
def test_posterior_rejects_incompatible_coordinates(incompatible):
    def model(value, name="mu", prior=None):
        mu = lsl.Var.new_param(value, prior, name=name)
        y = lsl.Var.new_obs(jnp.zeros(2), lsl.Dist(tfd.Normal, mu, 1.0), name="y")
        return lsl.Model(y)

    original = model(jnp.array(0.0))
    optim = opt.LieselOptim(
        original,
        optimizers=optax.sgd(0.0),
        show_progress=False,
        loss_monitor="train_full_data",
        stopper=opt.Stopper(epochs=1, patience=1),
    )
    result = optim.fit()
    if incompatible == "unknown":
        other = model(jnp.array(0.0), name="other")
    elif incompatible == "shape":
        other = model(jnp.zeros(2))
    else:
        other = model(jnp.array(0.0), prior=lsl.Dist(tfd.Poisson, rate=1.0))
    loss = opt.NegLogProbLoss(other, opt.PositionSplit.from_model(other))
    with pytest.raises(RuntimeError, match="coordinates"):
        loss.approximate_joint_posterior(result)
    failed = loss.approximate_joint_posterior(result, raise_on_failure=False)
    assert not failed.valid
    assert failed.precision_cholesky is None


def test_posterior_handles_managed_splits_transforms_and_fixed_parameters():
    with jax.enable_x64():
        scale = lsl.Var.new_param(
            1.0, lsl.Dist(tfd.LogNormal, 0.0, 1.0), bijector=tfb.Exp(), name="scale"
        )
        fixed = lsl.Var.new_param(3.0, name="fixed")
        loc = lsl.Var.new_calc(
            lambda scale, fixed: jnp.log(scale) + fixed - 3, scale, fixed
        )
        a = lsl.Var.new_obs(jnp.full(4, 2.0), lsl.Dist(tfd.Normal, loc, 1.0), name="a")
        b = lsl.Var.new_obs(jnp.full(6, -1.0), lsl.Dist(tfd.Normal, loc, 2.0), name="b")
        model = lsl.Model([a, b], to_float32=False)
        split = opt.PositionSplitManager.from_model(
            model, position_keys=[["a"], ["b"]], validate_axis_share=0.5, shuffle=False
        )
        optim = opt.LieselOptim(
            model,
            split=split,
            optimizers=[opt.LBFGS(["h(scale)"])],
            loss_monitor="train_full_data",
            show_progress=False,
            stopper=opt.Stopper(epochs=10, patience=5),
        )
        posterior = optim.loss.approximate_joint_posterior(optim.fit())
        # log(scale) ~ N(0,1); two N(eta,1) and three N(eta,4) training values.
        assert posterior.names == ("h(scale)",)
        np.testing.assert_allclose(posterior.mean["h(scale)"], 13 / 15, atol=1e-10)
        np.testing.assert_allclose(posterior.covariance(), [[4 / 15]], atol=1e-10)
        draws = posterior.sample((2, 3), seed=jax.random.key(4))
        predicted = model.predict(draws, predict=["scale", "fixed"])
        np.testing.assert_allclose(predicted["scale"], jnp.exp(draws["h(scale)"]))
        np.testing.assert_allclose(predicted["fixed"], 3.0)
        assert float(model.vars["scale"].value) == 1.0


def test_posterior_requires_at_least_one_optimized_coordinate():
    mu = lsl.Var.new_param(jnp.zeros(0), name="mu")
    loc = lsl.Var.new_calc(jnp.sum, mu)
    y = lsl.Var.new_obs(jnp.zeros(2), lsl.Dist(tfd.Normal, loc, 1.0), name="y")
    optim = opt.LieselOptim(
        lsl.Model(y),
        optimizers=optax.sgd(0.0),
        show_progress=False,
        loss_monitor="train_full_data",
        stopper=opt.Stopper(epochs=1, patience=1),
    )
    result = optim.fit()
    with pytest.raises(RuntimeError, match="No optimized coordinates"):
        optim.loss.approximate_joint_posterior(result)
    assert not optim.loss.approximate_joint_posterior(
        result, raise_on_failure=False
    ).valid
