"""
some tests for the engine builder
"""

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
from liesel.goose.builder import EngineBuilder
from liesel.goose.models import DictModel


def test_jitter_fns():
    con = DictModel(lambda ms: -0.5 * ms["x"] ** 2 - 0.5 * ms["y"])
    ms = {"x": jnp.array(1), "y": jnp.array(-1)}

    num_chains = 4

    builder = EngineBuilder(seed=1, num_chains=num_chains)
    builder.set_model(con)
    builder.set_initial_values(ms, multiple_chains=False)
    builder.set_jitter_fns(
        {
            "x": lambda key, cv: cv
            + tfd.Uniform(-1.0, 1.0).sample(sample_shape=cv.shape, seed=key),
            "y": lambda key, cv: cv
            + tfd.Uniform(-1.0, 1.0).sample(sample_shape=cv.shape, seed=key),
        }
    )
    builder.add_kernel(gs.IWLSKernel(["x", "y"]))
    builder.set_duration(warmup_duration=200, posterior_duration=10, term_duration=10)
    engine = builder.build()

    for c in range(num_chains):
        assert not jnp.allclose(ms["x"], engine._model_states["x"][c])
        assert not jnp.allclose(ms["y"], engine._model_states["y"][c])
