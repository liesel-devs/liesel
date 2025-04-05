import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
n = 200
y = jax.random.normal(k1, (n,))
x1 = jax.random.uniform(k2, (n,))
x2 = jax.random.uniform(k3, (n,))

X = jnp.c_[jnp.ones_like(x1), x1, x2]

beta = lsl.Var.new_param(
    jnp.zeros(3), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="beta"
)

mu = lsl.Var.new_calc(jnp.dot, X, beta)

yvar = lsl.Var.new_obs(y, lsl.Dist(tfd.Normal, loc=mu, scale=1.0), name="y")

model = lsl.Model([yvar])


class TestLogProb:
    @pytest.mark.parametrize("component", ("log_prob", "log_lik", "log_prior"))
    def test_log_prob(self, component) -> None:
        logprob_fn = lsl.LogProb(model, component=component)
        logprob = jax.jit(logprob_fn)({"beta": jnp.array([1.0, 2.0, 3.0])})
        assert not jnp.isnan(logprob)

    @pytest.mark.parametrize("component", ("log_prob", "log_lik", "log_prior"))
    def test_grad(self, component) -> None:
        logprob_fn = lsl.LogProb(model, component=component)
        result = jax.jit(logprob_fn.grad)({"beta": jnp.array([1.0, 2.0, 3.0])})

        assert not jnp.any(jnp.isnan(result["beta"]))
        assert result["beta"].shape == (3,)

    @pytest.mark.parametrize("diff_mode", ("forward", "reverse"))
    @pytest.mark.parametrize("component", ("log_prob", "log_lik", "log_prior"))
    def test_hessian(self, diff_mode, component) -> None:
        logprob_fn = lsl.LogProb(model, component=component, diff_mode=diff_mode)
        result = jax.jit(logprob_fn.hessian)({"beta": jnp.array([1.0, 2.0, 3.0])})

        assert not jnp.any(jnp.isnan(result["beta"]["beta"]))
        assert result["beta"]["beta"].shape == (3, 3)


class TestFlatLogProb:
    @pytest.mark.parametrize("component", ("log_prob", "log_lik", "log_prior"))
    def test_log_prob(self, component) -> None:
        logprob_fn = lsl.FlatLogProb("beta", model=model, component=component)
        logprob = jax.jit(logprob_fn)(jnp.array([1.0, 2.0, 3.0]))
        assert not jnp.isnan(logprob)

    @pytest.mark.parametrize("component", ("log_prob", "log_lik", "log_prior"))
    def test_grad(self, component) -> None:
        logprob_fn = lsl.FlatLogProb("beta", model=model, component=component)
        result = jax.jit(logprob_fn.grad)(jnp.array([1.0, 2.0, 3.0]))

        assert not jnp.any(jnp.isnan(result))
        assert result.shape == (3,)

    @pytest.mark.parametrize("diff_mode", ("forward", "reverse"))
    @pytest.mark.parametrize("component", ("log_prob", "log_lik", "log_prior"))
    def test_hessian(self, diff_mode, component) -> None:
        logprob_fn = lsl.FlatLogProb(
            "beta", model=model, component=component, diff_mode=diff_mode
        )
        result = jax.jit(logprob_fn.hessian)(jnp.array([1.0, 2.0, 3.0]))

        assert not jnp.any(jnp.isnan(result))
        assert result.shape == (3, 3)
