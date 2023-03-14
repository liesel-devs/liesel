import jax.numpy as jnp
import numpy as np

import liesel
from liesel.experimental.arviz import to_arviz_inference_data
from liesel.goose.engine import SamplingResults


def test_structure(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    assert infdat.groups() == ["posterior"]

    assert infdat["posterior"]["foo"].shape == (3, 250, 3)
    assert infdat["posterior"]["bar"].shape == (3, 250, 3, 5, 7)
    assert infdat["posterior"]["baz"].shape == (
        3,
        250,
    )


def test_structure_warmup(result: SamplingResults):
    infdat = to_arviz_inference_data(result, include_warmup=True)
    print(infdat)

    assert "posterior" in infdat.groups() and "warmup_posterior" in infdat.groups()

    assert infdat["warmup_posterior"]["foo"].shape == (3, 50, 3)
    assert infdat["warmup_posterior"]["bar"].shape == (3, 50, 3, 5, 7)
    assert infdat["warmup_posterior"]["baz"].shape == (
        3,
        50,
    )


def test_attributes(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    infdat["posterior"].attrs["inference_library"] == "liesel"
    infdat["posterior"].attrs["inference_library_version"] == liesel.__version__


def test_posterior_mean(result: SamplingResults):
    infdat = to_arviz_inference_data(result)
    means = infdat["posterior"].mean(["chain", "draw"])

    assert np.allclose(means["foo"], [175.5, 176.5, 177.5])

    assert np.allclose(means["bar"], 175.5 * jnp.ones((3, 5, 7)))

    assert means["baz"] == 176.5
