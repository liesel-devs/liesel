import jax.numpy as jnp
import numpy as np

import liesel
from liesel.experimental.arviz import to_arviz_inference_data
from liesel.goose.engine import SamplingResults


def test_structure(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    posterior = infdat["posterior"].dataset
    assert list(infdat.children) == ["posterior"]

    assert posterior["foo"].shape == (3, 250, 3)
    assert posterior["bar"].shape == (3, 250, 3, 5, 7)
    assert posterior["baz"].shape == (
        3,
        250,
    )


def test_structure_warmup(result: SamplingResults):
    infdat = to_arviz_inference_data(result, include_warmup=True)
    warmup_posterior = infdat["warmup_posterior"].dataset

    assert "posterior" in infdat.children
    assert "warmup_posterior" in infdat.children

    assert warmup_posterior["foo"].shape == (3, 50, 3)
    assert warmup_posterior["bar"].shape == (3, 50, 3, 5, 7)
    assert warmup_posterior["baz"].shape == (
        3,
        50,
    )


def test_attributes(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    posterior = infdat["posterior"].dataset
    assert posterior.attrs["inference_library"] == "liesel"
    assert posterior.attrs["inference_library_version"] == liesel.__version__


def test_posterior_mean(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    posterior = infdat["posterior"].dataset
    means = posterior.mean(["chain", "draw"])

    assert np.allclose(means["foo"], [175.5, 176.5, 177.5])

    assert np.allclose(means["bar"], 175.5 * jnp.ones((3, 5, 7)))

    assert means["baz"] == 176.5
