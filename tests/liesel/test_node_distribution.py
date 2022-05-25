import numpy as np
from pytest import approx, raises

import liesel.tfp.jax.distributions as jd
import liesel.tfp.numpy.distributions as nd
from liesel.liesel import Node, NodeDistribution


def test_distribution():
    """Tests an untransformed Gaussian node distribution."""

    n_loc = Node(0.0)
    n_scale = Node(1.0)

    d = NodeDistribution("Normal", loc=n_loc, scale=n_scale)

    assert isinstance(d.distribution(), nd.Distribution)

    assert d.cdf(0.0) == approx(0.5)
    assert d.log_prob(0.0) == approx(-0.9189385)
    assert d.mean() == approx(0.0)

    assert np.mean(d.sample(10_000, seed=42)) == approx(0.0, abs=0.01)


def test_transfomed():
    """Tests an exp-transformed Gaussian (= log-normal) node distribution."""

    n_loc = Node(0.0)
    n_scale = Node(1.0)

    d = NodeDistribution("Normal", "Exp", loc=n_loc, scale=n_scale)

    assert isinstance(d.distribution(), nd.TransformedDistribution)

    assert d.cdf(1.0) == approx(0.5)
    assert d.log_prob(1.0) == approx(-0.9189385)

    with raises(NotImplementedError):
        d.mean()

    assert np.mean(d.sample(10_000, seed=42)) == approx(1.648721, abs=0.01)


def test_jaxify():
    """Tests if a node distribution can be jaxified and unjaxified."""

    n_loc = Node(0.0)
    n_scale = Node(1.0)

    d = NodeDistribution("Normal", loc=n_loc, scale=n_scale)

    d.jaxify()
    assert not isinstance(d.distribution(), nd.Distribution)
    assert isinstance(d.distribution(), jd.Distribution)

    d.unjaxify()
    assert not isinstance(d.distribution(), jd.Distribution)
    assert isinstance(d.distribution(), nd.Distribution)

    d = NodeDistribution("Normal", "Exp", loc=n_loc, scale=n_scale)

    d.jaxify()
    assert not isinstance(d.distribution(), nd.TransformedDistribution)
    assert isinstance(d.distribution(), jd.TransformedDistribution)

    d.unjaxify()
    assert not isinstance(d.distribution(), jd.TransformedDistribution)
    assert isinstance(d.distribution(), nd.TransformedDistribution)
