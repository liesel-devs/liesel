import gc
import inspect
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
from liesel.optim.types import Position


def _model():
    alpha = lsl.Var.new_param(0.0, name="alpha")
    beta = lsl.Var.new_param(jnp.zeros(2), name="beta")
    positive = lsl.Var.new_calc(jnp.exp, alpha, name="positive")
    y = lsl.Var.new_obs(
        jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, alpha, 1.0), name="y"
    )
    return lsl.Model([y, beta, positive])


def _result(final, minimum):
    return opt.OptimResult(
        history=opt.OptimHistory.from_epochs(epochs=1, position=None),
        position_final=final,
        position_min_monitor=minimum,
        n_epochs=1,
        min_monitor_epoch=0 if minimum is not None else None,
        monitor_source="train_ema",
        patience=1,
        duration=0.0,
    )


@pytest.mark.parametrize("family", ["diagonal", "dense", "composite"])
def test_posterior_samples_match_existing_sampler(family):
    model = _model()
    if family == "composite":
        vdist = opt.CompositeVDist(
            opt.VDist(["alpha"], model).normal(),
            opt.VDist(["beta"], model).mvn_diag(),
        ).build()
    elif family == "dense":
        factor = jnp.array([[0.3, 0.0, 0.0], [0.1, 0.4, 0.0], [-0.1, 0.2, 0.5]])
        vdist = opt.VDist(["alpha", "beta"], model).mvn_tril(scale_tril=factor).build()
    else:
        vdist = opt.VDist(["alpha", "beta"], model).mvn_diag().build()
    loss = opt.NegElboLoss.from_vdist(vdist)
    position = loss.position(vdist.parameters)
    result = _result(position, position)
    posterior = loss.approximate_joint_posterior(result)
    assert isinstance(posterior, opt.VariationalApproximation)
    key = jax.random.key(51)
    for shape in ((), 7, (2, 7)):
        axes = (shape,) if isinstance(shape, int) else shape
        actual = posterior.sample(shape, seed=key)
        expected = vdist.sample(key, axes, at_position=position)
        assert actual["alpha"].shape == axes
        assert actual["beta"].shape == axes + (2,)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("problem", ["unknown", "shape", "dtype", "numpy_float64"])
def test_incompatible_fit_position_is_rejected(problem):
    loss = opt.NegElboLoss.mvn_diag(_model())
    position = loss.position(list(loss.q.parameters))
    key = next(iter(position))
    if problem == "unknown":
        position["unknown"] = position.pop(key)
    elif problem == "shape":
        position[key] = jnp.zeros((99,), dtype=position[key].dtype)
    elif problem == "numpy_float64":
        position[key] = np.asarray(position[key], dtype=np.float64)
    else:
        position[key] = position[key].astype(jnp.int32)
    with pytest.raises(ValueError, match="Unknown|shapes or dtypes"):
        loss.approximate_joint_posterior(_result(position, position))


def test_selected_position_is_a_snapshot_without_retaining_result():
    model = _model()
    vdist = opt.VDist(["alpha", "beta"], model).mvn_diag(scale_diag=0.2).build()
    loss = opt.NegElboLoss.from_vdist(vdist)
    loc_key = next(key for key in vdist.parameters if key.endswith("_loc"))
    minimum = Position({loc_key: np.full(3, 2.0, dtype=np.float32)})
    final = Position({loc_key: jnp.full(3, 5.0)})
    result = _result(final, minimum)
    result_ref, history_ref = weakref.ref(result), weakref.ref(result.history)
    posterior = loss.approximate_joint_posterior(result)
    terminal = loss.approximate_joint_posterior(result, at="final")
    key = jax.random.key(7)
    expected = vdist.sample(
        key, (8,), at_position=Position({loc_key: jnp.full(3, 2.0)})
    )
    minimum[loc_key][...] = -100.0
    minimum.clear()
    del result
    gc.collect()
    assert result_ref() is None and history_ref() is None
    actual = posterior.sample(8, seed=key)
    final_draws = terminal.sample(8, seed=key)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_allclose(final_draws[name] - actual[name], 3.0, atol=1e-6)


def test_direct_custom_loss_sampling_and_prediction():
    model = _model()
    lower = lsl.Var.new_param(-1.0, name="lower")
    z = lsl.Var.new_obs(0.5, lsl.Dist(tfd.Uniform, lower, 2.0), name="z")
    q = lsl.Model([z])

    def mapping(position):
        return Position(
            {"alpha": position["z"], "beta": jnp.stack([position["z"], -position["z"]])}
        )

    loss = opt.NegElboLoss(model, q, q_to_p=mapping)
    position = Position({"lower": jnp.array(-2.0)})
    posterior = loss.approximate_joint_posterior(_result(position, position))
    draws = posterior.sample((1, 1000), seed=jax.random.key(4))
    assert draws["alpha"].shape == (1, 1000)
    assert jnp.all((draws["alpha"] >= -2.0) & (draws["alpha"] <= 2.0))
    assert draws["alpha"].min() < -1.0  # The fitted lower bound, not q's initial one.
    np.testing.assert_array_equal(draws["beta"][..., 0], draws["alpha"])
    np.testing.assert_array_equal(draws["beta"][..., 1], -draws["alpha"])
    predicted = model.predict(draws, predict=["positive"])
    np.testing.assert_allclose(
        predicted["positive"], jnp.exp(draws["alpha"]), rtol=1e-6
    )


def test_posterior_from_actual_fit():
    model = _model()
    loss = opt.NegElboLoss.mvn_diag(model, nsamples=2)
    result = opt.LieselVI(
        model,
        loss=loss,
        optimizers=optax.adam(0.01),
        loss_monitor=opt.EmaTrainLossMonitor(1),
        stopper=opt.Stopper(epochs=3, patience=3),
        show_progress=False,
    ).fit()
    posterior = loss.approximate_joint_posterior(result)
    draws = posterior.sample((1, 10), seed=jax.random.key(18))
    predicted = model.predict(draws, predict=["positive"])
    assert jnp.isfinite(predicted["positive"]).all()
    np.testing.assert_allclose(
        predicted["positive"], jnp.exp(draws["alpha"]), rtol=1e-6
    )
    seed_parameter = inspect.signature(posterior.sample).parameters["seed"]
    assert seed_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert seed_parameter.default is inspect.Parameter.empty


def test_unavailable_nonfinite_and_invalid_selection():
    loss = opt.NegElboLoss.mvn_diag(_model())
    position = loss.position(list(loss.q.parameters))
    result = _result(position, None)
    with pytest.raises(RuntimeError, match="No finite monitoring loss"):
        loss.approximate_joint_posterior(result)
    loss.approximate_joint_posterior(result, at="final")
    with pytest.raises(ValueError, match="at must be"):
        loss.approximate_joint_posterior(result, at="best")
    name = next(iter(position))
    position[name] = jnp.full_like(position[name], jnp.nan)
    result = _result(position, position)
    for at in ("final", "min_monitor"):
        with pytest.raises(RuntimeError, match="NaN or infinity"):
            loss.approximate_joint_posterior(result, at=at)
