import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
target_params = 0.5

# Generate some data.
xs = jax.random.normal(key, (50, 2))
ys = jnp.sum(xs * target_params, axis=-1) + jax.random.normal(subkey, (50,))

x = lsl.obs(xs, name="x")
coef = lsl.param(jnp.zeros(2), name="coef")
mu = lsl.Var(lsl.Calc(jnp.dot, x, coef), name="mu")
log_sigma = lsl.Var(2.0, name="log_sigma")
sigma = lsl.Var(lsl.Calc(jnp.exp, log_sigma), name="sigma")

ydist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
y = lsl.obs(ys, ydist, name="y")

gb = lsl.GraphBuilder().add(y)
model = gb.build_model()


def test_optim_jointly():
    result = gs.optim_flat(model, ["coef", "log_sigma"], batch_size=20)
    assert jnp.allclose(result.position["coef"], target_params, atol=0.2)
    assert jnp.allclose(result.position["log_sigma"], 0.0, atol=0.2)


def test_optim_no_batching():
    result = gs.optim_flat(model, ["coef", "log_sigma"], batch_size=None)
    assert jnp.allclose(result.position["coef"], target_params, atol=0.2)
    assert jnp.allclose(result.position["log_sigma"], 0.0, atol=0.2)


def test_optim_individually():
    interface = gs.LieselInterface(model)
    result_coef = gs.optim_flat(model, ["coef"])
    new_state = interface.update_state(result_coef.position, model.state)
    model.state = new_state
    result_sigma = gs.optim_flat(model, ["log_sigma"])
    assert jnp.allclose(result_coef.position["coef"], target_params, atol=0.2)
    assert jnp.allclose(result_sigma.position["log_sigma"], 0.0, atol=0.2)
