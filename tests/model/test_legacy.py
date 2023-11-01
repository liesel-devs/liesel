import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as jb
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.model.legacy import (
    PIT,
    Addition,
    Bijector,
    ColumnStack,
    DesignMatrix,
    Hyperparameter,
    InverseLink,
    Parameter,
    Predictor,
    RegressionCoef,
    Response,
    Smooth,
    SmoothingParam,
)
from liesel.model.nodes import Data, Dist, Var, param


def test_design_matrix() -> None:
    x = DesignMatrix(value=jnp.eye(2), name="x")
    assert jnp.allclose(x.value, jnp.eye(2))


def test_hyperparameter() -> None:
    x = Hyperparameter(value=1.0, name="x")
    assert jnp.allclose(x.value, 1.0)


def test_parameter() -> None:
    loc = param(0.0, name="loc")
    scale = param(1.0, name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    x = Parameter(value=1.0, distribution=dist, name="x")
    x.update()

    assert jnp.allclose(x.value, 1.0)
    assert jnp.allclose(x.log_prob, tfd.Normal(0.0, 1.0).log_prob(1.0))


def test_regression_coef() -> None:
    arr = jnp.array([1.0, 3.0, 2.3])

    loc = param(0.0, name="loc")
    scale = param(1.0, name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    beta = RegressionCoef(value=arr, distribution=dist, name="beta")
    beta.update()

    assert jnp.allclose(beta.value, arr)
    assert jnp.allclose(beta.log_prob, tfd.Normal(0.0, 1.0).log_prob(arr))


def test_response() -> None:
    arr = jnp.array([1.0, 0.0, 2.0])

    loc = param(0.0, name="loc")
    scale = param(1.0, name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    y = Response(value=arr, distribution=dist, name="y")
    y.update()

    assert jnp.allclose(y.value, arr)
    assert jnp.allclose(y.log_prob, tfd.Normal(0.0, 1.0).log_prob(arr))


def test_smoothing_param() -> None:
    loc = param(0.0, name="loc")
    scale = param(1.0, name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    tau2 = SmoothingParam(value=1.0, distribution=dist, name="tau2")
    tau2.update()

    assert jnp.allclose(tau2.value, 1.0)
    assert jnp.allclose(tau2.log_prob, tfd.Normal(0.0, 1.0).log_prob(1.0))


def test_addition() -> None:
    a = Data(1.0, _name="a")
    b = Data(2.0, _name="b")
    addition = Addition(a, b)
    addition.update()

    assert jnp.allclose(addition.value, 3.0)


def test_predictor() -> None:
    a = Data(1.0, _name="a")
    b = Data(2.0, _name="b")
    predictor = Predictor(a, b)
    predictor.update()

    assert jnp.allclose(predictor.value, 3.0)


def test_bijector() -> None:
    a = Data(1.0, _name="a")
    exp_bijector = Bijector(a, jb.Exp)
    exp_bijector.update()

    assert jnp.allclose(exp_bijector.value, jnp.e)

    log_bijector = Bijector(a, jb.Exp, inverse=True)
    log_bijector.update()

    assert jnp.allclose(log_bijector.value, 0.0)


def test_inverse_link() -> None:
    a = Data(0.5, _name="a")
    log_inverse_link = InverseLink(a, jb.Log)
    log_inverse_link.update()

    assert jnp.allclose(log_inverse_link.value, jnp.log(0.5))


def test_column_stack() -> None:
    a = Data(13.0, _name="a")
    b = Data(73.0, _name="b")
    c = Data(10.0, _name="c")

    cs = ColumnStack(a, b, c)
    cs.update()

    assert jnp.allclose(cs.value, jnp.array([13.0, 73.0, 10.0]))


def test_column_stack_with_distribution() -> None:
    a = Data(1.0, _name="a")
    b = Data(0.5, _name="b")
    c = Data(0.0, _name="c")

    loc = Data(jnp.zeros(3), _name="loc")
    cov = Data(jnp.eye(3), _name="cov")
    dist = Dist(tfd.MultivariateNormalFullCovariance, loc, cov)

    cs = ColumnStack(a, b, c, distribution=dist)
    cs.update()

    tfp_dist = tfd.MultivariateNormalFullCovariance(jnp.zeros(3), jnp.eye(3))
    assert jnp.allclose(cs.log_prob, tfp_dist.log_prob([1.0, 0.5, 0.0]))


def test_pit_var() -> None:
    x = Var(1.0, name="x")

    with pytest.raises(RuntimeError):
        pit = PIT(x)

    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    x.dist_node = dist

    pit = PIT(x)
    pit.update()

    assert jnp.allclose(pit.value, tfd.Normal(0.0, 1.0).cdf(1.0))


def test_pit_dist() -> None:
    loc = Data(0.0, _name="loc")
    scale = Data(1.0, _name="scale")
    dist = Dist(tfd.Normal, loc, scale)
    dist.at = Data(1.0, _name="x")

    pit = PIT(dist)
    pit.update()

    assert jnp.allclose(pit.value, tfd.Normal(0.0, 1.0).cdf(1.0))


def test_smooth() -> None:
    x = DesignMatrix(value=jnp.eye(2), name="x")
    beta = SmoothingParam(value=jnp.array([1.0, 3.0]), name="beta")
    smooth = Smooth(x, beta)
    smooth.update()

    assert jnp.allclose(smooth.value, jnp.array([1.0, 3.0]))
