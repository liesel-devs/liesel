import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl


@pytest.fixture
def model() -> lsl.Model:
    a = lsl.Var.new_param(0.5, lsl.Dist(tfd.Normal, 0.0, 1.0), name="a")
    s = lsl.Var.new_param(2.0, lsl.Dist(tfd.Gamma, 2.0, 1.0), name="s")
    s.transform()
    y = lsl.Var.new_obs(jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, a, s), name="y")
    return lsl.Model([y])


def test_terms_and_weights(model: lsl.Model) -> None:
    state = model.state
    lp = {name: jnp.sum(state[f"{name}_log_prob"].value) for name in ["a", "y"]}
    lp["s"] = state["s_transformed_log_prob"].value

    assert model.target()(state) == pytest.approx(model.log_prob)
    assert model.target(component="log_lik")(state) == pytest.approx(lp["y"])
    assert model.target(exclude=["y"])(state) == pytest.approx(lp["a"] + lp["s"])

    target = model.target(include=["s", "y"], weights={"y": 0.5})
    assert target.weights == {"s_transformed_log_prob": 1.0, "y_log_prob": 0.5}
    assert target(state) == pytest.approx(lp["s"] + 0.5 * lp["y"])


def test_invalid_targets(model: lsl.Model) -> None:
    with pytest.raises(ValueError, match="not part of the model"):
        model.target(exclude=["b"])
    with pytest.raises(ValueError, match="either 'include' or 'component'"):
        model.target(include=["a"], component="log_lik")
    with pytest.raises(ValueError, match="not part of the target"):
        model.target(exclude=["y"], weights={"y": 0.5})
    with pytest.raises(ValueError, match="does not contain any terms"):
        model.target(include=[])
