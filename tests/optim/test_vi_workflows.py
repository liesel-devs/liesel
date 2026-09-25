import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
import liesel.optim as opt
from liesel.optim.types import Position


def _normal_model(values):
    mean = lsl.Var.new_param(0.0, name="m")
    scale = lsl.Var.new_param(1.0, name="s")
    unconstrained = scale.transform(tfb.Exp())
    response = lsl.Var.new_obs(values, lsl.Dist(tfd.Normal, mean, scale), name="x")
    return lsl.Model([response]), unconstrained.name


@pytest.mark.parametrize("case", ["diagonal", "fullrank", "high_n"])
def test_normal_fit_and_posterior_workflow(case):
    n, batch_size = (100_000, 128) if case == "high_n" else (1000, 32)
    values = 2.0 * jax.random.normal(jax.random.key(42), (n,)) + 3.0
    split = opt.Split.from_axis_shares(["x"], n, shuffle=True, seed=1)
    data = split.split_position(Position({"x": values}))
    batches = opt.Batches(["x"], data.train_axis_size, batch_size, shuffle=True)
    model, scale_key = _normal_model(batches.get_batched_position(data.train, 0)["x"])
    if case == "fullrank":
        vdist = opt.VDist(["m", scale_key], model).mvn_tril().build()
    else:
        vdist = opt.CompositeVDist(
            opt.VDist(["m"], model).mvn_diag(),
            opt.VDist([scale_key], model).mvn_diag(),
        ).build()
    loss = opt.NegElboLoss.from_vdist(
        vdist, data, nsamples=1 if case == "high_n" else 10, scale=case == "high_n"
    )
    initial_samples = vdist.sample((1, 1000), seed=jax.random.key(42))
    result = opt.OptimEngine(
        loss=loss,
        loss_monitor=opt.EmaTrainLossMonitor(20 if case == "high_n" else 1),
        batches=batches,
        optimizers=[opt.Optimizer(vdist.parameters, optax.adam(1e-3))],
        stopper=opt.Stopper(epochs=1000, patience=50, atol=0.0, rtol=1e-3),
        seed=42,
        initial_state=model.state,
        prune_history=True,
        save_position_history=True,
        show_progress=False,
    ).fit()
    for position in (result.position_final, result.position_min_monitor):
        assert all(jnp.isfinite(value).all() for value in position.values())
    samples = vdist.sample(
        (1, 1000), seed=jax.random.key(42), at_position=result.position_final
    )
    predicted = model.predict(samples, predict=["s"])
    assert samples["m"].shape == predicted["s"].shape == (1, 1000)
    np.testing.assert_allclose(predicted["s"], jnp.exp(samples[scale_key]), rtol=1e-6)
    assert jnp.all(jnp.isfinite(predicted["s"]) & (predicted["s"] > 0))
    assert abs(samples["m"].mean() - values.mean()) < abs(
        initial_samples["m"].mean() - values.mean()
    )
    summary = gs.SamplesSummary(samples | predicted).to_dataframe()
    np.testing.assert_allclose(
        summary.loc["s", "mean"], predicted["s"].mean(), rtol=1e-6
    )
    if case == "diagonal":
        for plot in (result.plot_loss(window=50), result.plot_params(window=50)):
            figure = plot.draw()
            assert figure.axes
            plt.close(figure)
        gs.plot_pairs(samples | predicted, show=False)
        plt.close("all")


def test_quick_fullrank_fit_with_ten_thousand_rows():
    values = 2.0 * jax.random.normal(jax.random.key(42), (10_000,)) + 3.0
    model, _ = _normal_model(values)
    builder = opt.LieselVI(
        model,
        loss="mvn_tril",
        optimizers=optax.adam(1e-3),
        loss_monitor=opt.EmaTrainLossMonitor(1),
        show_progress=False,
    )
    vdist = builder.loss.vdist
    assert isinstance(vdist, opt.VDist)
    initial = vdist.sample((1000,), seed=jax.random.key(42))
    result = builder.fit()
    samples = vdist.sample(
        (1000,), seed=jax.random.key(42), at_position=result.position_final
    )
    assert all(jnp.isfinite(value).all() for value in samples.values())
    assert abs(samples["m"].mean() - values.mean()) < abs(
        initial["m"].mean() - values.mean()
    )
    for plot in (result.plot_loss(), result.plot_params()):
        figure = plot.draw()
        assert figure.axes
        plt.close(figure)


def test_untransformed_dense_scale_samples_match_covariance():
    model, scale_key = _normal_model(jnp.zeros(24))
    vdist = (
        opt.VDist(["m", scale_key], model)
        .mvn_tril(scale_tril=0.1, scale_tril_bijector=None)
        .build()
    )
    samples = vdist.sample((10000,), seed=jax.random.key(99))
    matrix = np.column_stack((samples["m"], samples[scale_key]))
    # Six Monte Carlo standard errors for Normal(0, 0.1**2) moments.
    np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=6 * 0.1 / 100)
    np.testing.assert_allclose(
        np.cov(matrix.T), 0.01 * np.eye(2), atol=6 * 0.01 * np.sqrt(2 / 10_000)
    )


def test_custom_lognormal_block_and_separate_optimizers():
    values = 2.0 * jax.random.normal(jax.random.key(42), (1000,)) + 3.0
    model, scale_key = _normal_model(values)
    q_mean = opt.VDist(["m"], model).mvn_diag()
    distribution = lsl.Dist(
        tfd.LogNormal,
        loc=lsl.Var.new_param(0.0, name="q_log_scale_loc"),
        scale=lsl.Var.new_param(1.0, name="q_log_scale_scale"),
    )
    q_scale = distribution["scale"]
    assert isinstance(q_scale, lsl.Var)
    q_scale.transform(tfb.Softplus())
    q_log_scale = opt.VDist([scale_key], model).init(distribution)
    vdist = opt.CompositeVDist(q_mean, q_log_scale).build()
    samples = vdist.sample((10000,), seed=jax.random.key(42))
    assert jnp.all(samples[scale_key] > 0)
    # LogNormal(0, 1) has mean exp(1/2) and variance (e - 1) * e.
    np.testing.assert_allclose(
        samples[scale_key].mean(),
        np.exp(0.5),
        atol=6 * np.sqrt((np.e - 1) * np.e / 10_000),
    )
    optimizers = [
        opt.Optimizer(q_mean.parameters, optax.adam(1e-2)),
        opt.Optimizer(q_log_scale.parameters, optax.adam(1e-5)),
    ]
    position = vdist.q.extract_position(vdist.parameters)
    mean_position = optimizers[0].position(position)
    scale_position = optimizers[1].position(position)
    assert mean_position.keys().isdisjoint(scale_position)
    assert mean_position.keys() | scale_position.keys() == position.keys()


def test_prefitted_regression_initializes_composite_vi():
    _, key_beta, key_data = jax.random.split(jax.random.PRNGKey(13), 3)
    beta = tfd.Normal(0.0, jnp.sqrt(8.0)).sample((4,), seed=key_beta)
    key_x, key_y = jax.random.split(key_data)
    x_values = tfd.Uniform(0.0, 1.0).sample((1000, 3), seed=key_x)
    y_values = tfd.Normal(beta[0] + x_values @ beta[1:], 1.0).sample(seed=key_y)
    data = opt.Split.from_axis_shares(["y", "X"], 1000, shuffle=True, seed=1)
    split = data.split_position(Position({"X": x_values, "y": y_values}))
    x = lsl.Var.new_obs(x_values, name="X")
    intercept = lsl.Var.new_param(0.0, name="beta_loc_0")
    slopes = lsl.Var.new_param(
        jnp.zeros(3), lsl.Dist(tfd.Normal, 0.0, 10.0), name="beta_loc"
    )
    log_scale = lsl.Var.new_param(0.0, name="beta_scale_0")
    mean = lsl.Var.new_calc(lambda x, a, b: a + x @ b, x, intercept, slopes)
    scale = lsl.Var.new_calc(jnp.exp, log_scale, name="scale")
    y = lsl.Var.new_obs(y_values, lsl.Dist(tfd.Normal, mean, scale), name="y")
    model = lsl.Model([y])
    prefit = opt.LieselOptim(
        model,
        loss=opt.NegLogProbLoss(model, split),
        loss_monitor=opt.EmaTrainLossMonitor(1),
        stopper=opt.Stopper(epochs=2000, patience=5),
        optimizers=[opt.LBFGS(list(model.parameters))],
        show_progress=False,
    ).fit()
    mode = prefit.position_final
    design = np.column_stack((np.ones(1000), x_values))
    variance = np.exp(2 * float(mode["beta_scale_0"]))
    assert np.isfinite(variance) and variance > 0
    # Gaussian slope priors give penalized normal equations at fixed variance.
    precision = np.diag([0.0, 0.01, 0.01, 0.01])
    expected = np.linalg.solve(
        design.T @ design + variance * precision, design.T @ np.asarray(y_values)
    )
    actual = np.r_[mode["beta_loc_0"], mode["beta_loc"]]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    model.state = model.update_state(mode, model.state)
    vdist = opt.CompositeVDist(
        opt.VDist(["beta_loc_0", "beta_loc"], model).mvn_diag(scale_diag=0.01),
        opt.VDist(["beta_scale_0"], model).mvn_diag(scale_diag=0.01),
    ).build()
    initial = vdist.sample((1000,), seed=jax.random.key(42))
    for key in mode:
        np.testing.assert_allclose(
            initial[key].mean(axis=0), mode[key], atol=6 * 0.01 / np.sqrt(1000)
        )
    loss = opt.NegElboLoss.from_vdist(vdist, split, nsamples=10)
    result = opt.OptimEngine(
        loss=loss,
        loss_monitor=opt.EmaTrainLossMonitor(1),
        batches=opt.Batches(["y", "X"], 1000, 32, shuffle=False),
        optimizers=[opt.Optimizer(vdist.parameters, optax.adam(1e-3))],
        stopper=opt.Stopper(epochs=5000, patience=500, atol=0.0, rtol=1e-6),
        seed=42,
        initial_state=model.state,
        prune_history=True,
        save_position_history=True,
        show_progress=False,
    ).fit()
    samples = vdist.sample(
        (1, 1000), seed=jax.random.key(42), at_position=result.position_final
    )
    prediction = model.predict(samples, predict=["scale"])
    np.testing.assert_allclose(
        prediction["scale"], jnp.exp(samples["beta_scale_0"]), rtol=1e-6
    )
    assert jnp.all(jnp.isfinite(prediction["scale"]) & (prediction["scale"] > 0))
    fitted_mean = samples["beta_loc_0"].mean() + x_values @ samples["beta_loc"].mean(
        axis=(0, 1)
    )
    assert jnp.mean((y_values - fitted_mean) ** 2) < jnp.mean(y_values**2)
