"""
tests of liesel.model.Var concerning log_prob and transform
"""

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

import liesel.model.nodes as lnodes


def test_log_prob_no_dist() -> None:
    var = lnodes.Var(0.0, None)
    assert var.log_prob == 0.0

    var.update()
    assert var.log_prob == 0.0


def test_log_prob_std_normal() -> None:
    dist = lnodes.Dist(tfp.distributions.Normal, 0.0, 1.0)
    var = lnodes.Var(0.0, dist)
    var.update()
    assert var.log_prob == tfp.distributions.Normal(0.0, 1.0).log_prob(0.0)

    dist = lnodes.Dist(tfp.distributions.Normal, 0.0, 1.0)
    var = lnodes.Var(1.0, dist)
    var.update()
    assert var.log_prob == tfp.distributions.Normal(0.0, 1.0).log_prob(1.0)


def test_log_prob_std_normal_vector() -> None:
    dist = lnodes.Dist(tfp.distributions.Normal, 0.0, 1.0)
    var = lnodes.Var(jnp.array([0.0, 0, 0]), dist)
    var.update()
    assert jnp.allclose(
        var.log_prob, tfp.distributions.Normal(0.0, 1.0).log_prob([0.0, 0, 0])
    )
