import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
from liesel.experimental.batching import (
    BatchedLieselInterface,
    BatchedLogProb,
    BatchIndices,
    FlatBatchedLogProb,
)

y = jnp.linspace(-1.0, 1.0, 10)

mu = lsl.Var.new_param(0.0, name="mu")
sigma = lsl.Var.new_param(1.0, name="sigma")

y = lsl.Var.new_obs(y, lsl.Dist(tfd.Normal, loc=mu, scale=sigma), "y")
model = lsl.Model([y])

bidx = BatchIndices(list(model.observed), n=y.value.size, batch_size=3)
interface = BatchedLieselInterface(model)
state = model.state


class TestLogProb:
    def test_log_prob(self):
        lp = BatchedLogProb(interface, state, bidx)

        pos = {"mu": 2.0}
        val = lp.log_prob(pos)

        assert not jnp.isnan(val)

    def test_grad(self):
        lp = BatchedLogProb(interface, state, batch_indices=bidx)

        pos = {"mu": 2.0}
        val = lp.grad(pos)

        assert not jnp.isnan(val["mu"])

    @pytest.mark.parametrize("diff_mode", ("forward", "reverse"))
    def test_hessian(self, diff_mode):
        lp = BatchedLogProb(interface, state, diff_mode=diff_mode, batch_indices=bidx)

        pos = {"mu": 2.0}
        val = lp.hessian(pos)
        assert not jnp.isnan(val["mu"]["mu"])


class TestFlatLogProb:
    def test_log_prob(self):
        lp = FlatBatchedLogProb(interface, state, ["mu"], batch_indices=bidx)

        pos = jnp.array([2.0])
        val = lp.log_prob(pos)

        assert not jnp.isnan(val)

    def test_grad(self):
        lp = FlatBatchedLogProb(interface, state, ["mu"], batch_indices=bidx)

        pos = jnp.array([2.0])
        val = lp.grad(pos)

        assert not jnp.isnan(val)

    @pytest.mark.parametrize("diff_mode", ("forward", "reverse"))
    def test_hessian(self, diff_mode):
        lp = FlatBatchedLogProb(
            interface, state, ["mu"], diff_mode=diff_mode, batch_indices=bidx
        )

        pos = jnp.array([2.0])
        val = lp.hessian(pos)

        assert not jnp.isnan(val)
