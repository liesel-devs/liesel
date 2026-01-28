from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl

y = jnp.linspace(-1.0, 1.0, 10)

mu = lsl.Var.new_param(0.0, name="mu")
sigma = lsl.Var.new_param(1.0, name="sigma")

y = lsl.Var.new_obs(y, lsl.Dist(tfd.Normal, loc=mu, scale=sigma), "y")
model = lsl.Model([y])
interface_liesel = gs.LieselInterface(model)

state_liesel = model.state
state_dict = {"mu": mu.value, "sigma": sigma.value, "y": y.value}


@dataclass
class State:
    mu: jax.Array
    sigma: jax.Array
    y: jax.Array


state_dataclass = State(mu.value, sigma.value, y.value)


def log_prob_dict(state) -> float:
    mu = state["mu"]
    sigma = state["sigma"]
    y = state["y"]
    return tfd.Normal(loc=mu, scale=sigma).log_prob(y).sum()


interface_dict = gs.DictInterface(log_prob_dict)


def log_prob_dataclass(state) -> float:
    mu = state.mu
    sigma = state.sigma
    y = state.y
    return tfd.Normal(loc=mu, scale=sigma).log_prob(y).sum()


interface_dataclass = gs.DataclassInterface(log_prob_dataclass)


interfaces = [interface_liesel, interface_dict, interface_dataclass]
states = [state_liesel, state_dict, state_dataclass]


class TestLogProb:
    def test_log_prob(self):
        vals = []
        for interface, state in zip(interfaces, states):
            lp = gs.LogProb(interface, state)

            pos = {"mu": 2.0}
            val = lp.log_prob(pos)

            vals.append(val)
            assert jnp.allclose(val, vals[0])
            assert not jnp.isnan(val)

    def test_grad(self):
        vals = []
        for interface, state in zip(interfaces, states):
            lp = gs.LogProb(interface, state)

            pos = {"mu": 2.0}
            val = lp.grad(pos)

            vals.append(val)
            assert jnp.allclose(val["mu"], vals[0]["mu"])
            assert not jnp.isnan(val["mu"])

    @pytest.mark.parametrize("diff_mode", ("forward", "reverse"))
    def test_hessian(self, diff_mode):
        vals = []
        for interface, state in zip(interfaces, states):
            lp = gs.LogProb(interface, state, diff_mode=diff_mode)

            pos = {"mu": 2.0}
            val = lp.hessian(pos)

            vals.append(val)
            assert jnp.allclose(val["mu"]["mu"], vals[0]["mu"]["mu"])
            assert not jnp.isnan(val["mu"]["mu"])


class TestFlatLogProb:
    def test_log_prob(self):
        vals = []
        for interface, state in zip(interfaces, states):
            lp = gs.FlatLogProb(interface, state, ["mu"])

            pos = jnp.array([2.0])
            val = lp.log_prob(pos)

            vals.append(val)
            assert jnp.allclose(val, vals[0])
            assert not jnp.isnan(val)

    def test_grad(self):
        vals = []
        for interface, state in zip(interfaces, states):
            lp = gs.FlatLogProb(interface, state, ["mu"])

            pos = jnp.array([2.0])
            val = lp.grad(pos)

            vals.append(val)
            assert jnp.allclose(val, vals[0])
            assert not jnp.isnan(val)

    @pytest.mark.parametrize("diff_mode", ("forward", "reverse"))
    def test_hessian(self, diff_mode):
        vals = []
        for interface, state in zip(interfaces, states):
            lp = gs.FlatLogProb(interface, state, ["mu"], diff_mode=diff_mode)

            pos = jnp.array([2.0])
            val = lp.hessian(pos)

            vals.append(val)
            assert jnp.allclose(val, vals[0])
            assert not jnp.isnan(val)
