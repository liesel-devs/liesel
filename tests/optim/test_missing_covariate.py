"""Missing-value lookup indices compose with splitting and Laplace integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt


@pytest.mark.parametrize("seed", [42, 73])
def test_missing_covariate_split_matches_integrated_fit(seed):
    rng = np.random.default_rng(12)
    n = 120
    complete = rng.normal(size=n)
    response = 1.0 + 2.0 * complete + rng.normal(scale=0.5, size=n)
    raw = np.where(rng.random(n) < 0.25, np.nan, complete)
    splitter = opt.Split(axis_size=n, validate_axis_size=24, seed=seed)
    ids = np.asarray(splitter.indices_train)
    missing = np.sort(ids[np.isnan(raw[ids])])
    base = jnp.asarray(np.where(np.isnan(raw), 0.0, raw))
    lookup = np.full(n, -1, dtype=np.int32)
    lookup[missing] = np.arange(len(missing))
    split = splitter.split_position(
        {"x_observed": base, "missing_index": jnp.asarray(lookup), "y": response}
    )
    latent = lsl.Var.new_param(
        jnp.linspace(-0.7, 0.8, len(missing)),
        lsl.Dist(tfd.Normal, 0.0, 1.0),
        name="x_missing",
    )
    alpha = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="alpha")
    beta = lsl.Var.new_param(1.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="beta")
    sigma = lsl.Var.new_param(
        1.0, lsl.Dist(tfd.HalfCauchy, 0.0, 5.0), bijector=tfb.Exp(), name="sigma"
    )
    x_observed = lsl.Var.new_obs(split.train["x_observed"], name="x_observed")
    missing_index = lsl.Var.new_obs(split.train["missing_index"], name="missing_index")
    x = lsl.Var.new_calc(
        lambda observed, index, z: jnp.where(
            index >= 0, z[jnp.maximum(index, 0)], observed
        ),
        x_observed,
        missing_index,
        latent,
        name="x",
    )
    mu = lsl.Var.new_calc(lambda a, b, x: a + b * x, alpha, beta, x)
    y = lsl.Var.new_obs(split.train["y"], lsl.Dist(tfd.Normal, mu, sigma), name="y")
    model = lsl.Model([y])
    calc = x.value_node
    assert isinstance(calc, lsl.Calc)
    traced = jax.make_jaxpr(calc.function)(
        split.train["x_observed"], split.train["missing_index"], latent.value
    )
    assert not traced.consts  # The calculation receives data explicitly.
    np.testing.assert_allclose(x.value, base.at[missing].set(latent.value)[ids])
    np.testing.assert_allclose(y.value, response[ids], rtol=1e-6)
    assert not np.intersect1d(missing, splitter.indices_validate).size
    np.testing.assert_array_equal(split.validate["missing_index"], -1)

    loss = opt.LaplaceLoss(model, split=split, latent=["x_missing"])
    result = opt.LieselOptim(
        model,
        split=split,
        loss=loss,
        optimizers="lbfgs",
        loss_monitor="train_full_data",
        show_progress=False,
    ).fit()
    assert result.status == "early_stopping", result.failure_reason
    fitted = result.position_min_monitor
    actual = np.array([fitted["alpha"], fitted["beta"], fitted["h(sigma)"]])
    observed = ~np.isnan(raw[ids])
    xt, yt = np.where(observed, raw[ids], 0.0), response[ids]

    def integrated_objective(params):
        a, b, log_s = params
        variance = np.exp(2 * log_s) + (~observed) * b**2
        residual = yt - a - b * xt
        likelihood = 0.5 * np.sum(np.log(variance) + residual**2 / variance)
        # Normal coefficient priors, Half-Cauchy scale prior, and Exp Jacobian.
        prior = (a**2 + b**2) / 50 + np.log1p(np.exp(2 * log_s) / 25)
        return likelihood + prior - log_s

    expected = scipy.optimize.minimize(integrated_objective, [0.0, 1.0, 0.0])
    assert expected.success, expected.message
    np.testing.assert_allclose(actual, expected.x, atol=1e-3, rtol=0)
    a, b, log_s = actual
    expected_modes = b * (response[missing] - a) / (np.exp(2 * log_s) + b**2)
    np.testing.assert_allclose(
        result.loss_state_min_monitor.latent_position["x_missing"],
        expected_modes,
        atol=1e-4,
        rtol=0,
    )
