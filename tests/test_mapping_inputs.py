"""Mapping inputs must work without dict methods or changes to caller-owned data."""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import assert_type

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
from liesel.goose.optim import batched_nodes
from liesel.goose.summary_m import SamplesSummary
from liesel.goose.summary_viz import plot_trace
from liesel.model.nodes import NodeState


class ReadOnlyMapping[V](Mapping[str, V]):
    """Implements only Mapping's required operations; no copy or union methods."""

    def __init__(self, values: Mapping[str, V]):
        self.values_by_key = values

    def __getitem__(self, key: str) -> V:
        return self.values_by_key[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values_by_key)

    def __len__(self) -> int:
        return len(self.values_by_key)


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType, ReadOnlyMapping])
@pytest.mark.parametrize("inplace", [False, True])
def test_model_state_mappings(mapping_type, inplace):
    x = lsl.Var(1.0, name="x")
    y = lsl.Var.new_calc(lambda x: x + 1.0, x, name="y")
    group = lsl.Group("group", x=x)
    model = lsl.Model(y)
    original_state = model.state
    state: Mapping[str, NodeState] = mapping_type(original_state)
    position: Mapping[str, float] = mapping_type({"x": 3.0})

    updated = model.update_state(position, state, inplace=inplace)
    assert_type(updated, dict[str, NodeState])
    assert type(updated) is dict
    assert model.extract_position(["x", "y"], mapping_type(updated)) == {
        "x": 3.0,
        "y": 4.0,
    }
    assert group.value_from(mapping_type(updated), "x") == 3.0
    assert x.value == (3.0 if inplace else 1.0)
    assert state[x.value_node.name].value == 1.0
    assert dict(position) == {"x": 3.0}

    weak_position: Mapping[str, float] = mapping_type({"y": 9.0})
    overridden = model.update_state(weak_position, state, allow_weak_vars=True)
    assert model.extract_position(["y"], overridden)["y"] == 9.0
    model.state = state
    assert x.value == 1.0


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType, ReadOnlyMapping])
def test_prediction_and_sampling_mappings(mapping_type):
    x = lsl.Var.new_param(1.0, name="x")
    offset = lsl.Var(0.0, name="offset")
    mean = lsl.Var.new_calc(lambda x, offset: x + offset, x, offset, name="mean")
    y = lsl.Var.new_obs(0.0, lsl.Dist(tfd.Normal, loc=mean, scale=1.0), name="y")
    model = lsl.Model(y)
    values = jnp.array([[1.0, 2.0, 3.0]])
    samples: Mapping[str, jax.Array] = mapping_type({"x": values})
    newdata: Mapping[str, float] = mapping_type({"offset": 10.0})

    predicted = model.predict(samples, predict=["mean"], newdata=newdata)
    assert type(predicted) is dict
    np.testing.assert_allclose(predicted["mean"], values + 10.0)
    np.testing.assert_allclose(mean.predict(samples, newdata), values + 10.0)
    pointwise = lsl.log_prob_pointwise(model.observed, samples, newdata)
    assert type(pointwise) is dict
    assert next(iter(pointwise.values())).shape == values.shape

    class CustomDist(lsl.Dist):
        pass

    # The narrower value type also checks Mapping covariance with ty.
    dists: dict[str, CustomDist] = {"y": CustomDist(tfd.Deterministic, loc=7.0)}
    for sampler in (model.sample, y.sample):
        drawn = sampler(
            (2,),
            seed=jax.random.key(0),
            posterior_samples=samples,
            newdata=newdata,
            dists=dists,
        )
        mapped = sampler(
            (2,),
            seed=jax.random.key(0),
            posterior_samples=samples,
            newdata=newdata,
            dists=ReadOnlyMapping(dists),
        )
        assert type(mapped) is dict
        np.testing.assert_array_equal(mapped["y"], drawn["y"])
        np.testing.assert_array_equal(mapped["y"], jnp.full((2, 1, 3), 7.0))
    assert samples["x"] is values
    assert dict(newdata) == {"offset": 10.0}


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType, ReadOnlyMapping])
def test_dict_interface_mappings(mapping_type):
    interface: gs.ModelInterface = gs.DictInterface(lambda state: -(state["x"] ** 2))
    state = mapping_type({"x": 1.0, "y": 2.0})
    position: Mapping[str, float] = mapping_type({"x": 3.0})
    updated = interface.update_state(position, state)
    assert type(updated) is dict
    assert updated == {"x": 3.0, "y": 2.0}
    assert dict(state) == {"x": 1.0, "y": 2.0}
    assert dict(position) == {"x": 3.0}


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType, ReadOnlyMapping])
@pytest.mark.parametrize("diff_mode", ["forward", "reverse"])
def test_log_prob_derivatives_with_mappings(mapping_type, diff_mode):
    x = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="x")
    model = lsl.Model(x)
    position: Mapping[str, float] = mapping_type({"x": 2.0})
    for lp in (
        lsl.LogProb(model, diff_mode=diff_mode),
        gs.InterfaceLogProb(
            gs.LieselInterface(model), model.state, diff_mode=diff_mode
        ),
    ):
        assert lp(position) == pytest.approx(tfd.Normal(0.0, 1.0).log_prob(2.0))
        gradient = lp.grad(position)
        hessian = lp.hessian(position)
        assert type(gradient) is dict
        assert type(hessian) is dict
        assert gradient["x"] == pytest.approx(-2.0)
        assert hessian["x"]["x"] == pytest.approx(-1.0)
        assert jax.jit(lp.grad)(dict(position))["x"] == pytest.approx(-2.0)
    assert dict(position) == {"x": 2.0}


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType, ReadOnlyMapping])
def test_summary_and_batching_do_not_modify_samples(mapping_type):
    values = jnp.arange(40.0).reshape(2, 20)
    samples: Mapping[str, jax.Array] = mapping_type({"x": values, "excluded": values})
    summary = SamplesSummary(samples, deselected=["excluded"], which=["mean"])
    assert summary.quantities["mean"]["x"] == pytest.approx(19.5)
    assert set(samples) == {"x", "excluded"}
    assert samples["x"] is values
    assert samples["excluded"] is values

    batched = batched_nodes(samples, jnp.array([1]))
    assert type(batched) is dict
    np.testing.assert_array_equal(batched["x"], values[1:])


def test_bijector_and_inference_mappings():
    scale = lsl.Var.new_param(1.0, name="scale")
    bijectors: dict[str, tfb.Exp] = {"scale": tfb.Exp()}
    lsl.Dist(tfd.Normal, loc=0.0, scale=scale, bijectors=ReadOnlyMapping(bijectors))
    assert scale.weak
    assert list(bijectors) == ["scale"]

    spec = gs.MCMCSpec(gs.NUTSKernel)
    x = lsl.Var.new_param(0.0, inference=ReadOnlyMapping({"nuts": spec}))
    assert x.get_inference("nuts") is spec
    with pytest.raises(ValueError, match="Possible keys"):
        x.get_inference(None)


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType, ReadOnlyMapping])
def test_jitter_mapping_is_copied(mapping_type):
    builder = gs.EngineBuilder(seed=0, num_chains=1)
    functions = {"x": lambda key, value: value + 1.0}
    builder.set_jitter_fns(mapping_type(functions))
    stored = builder.jitter_fns.unwrap()
    assert type(stored) is dict
    assert stored == functions
    functions.clear()
    assert "x" in stored


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType, ReadOnlyMapping])
def test_inference_snapshot_preserves_shared_specs(mapping_type):
    kwargs = {"mm_diag": True}
    shared_kwargs = mapping_type(kwargs)
    first = gs.MCMCSpec(
        gs.NUTSKernel, kernel_group="joint", kernel_kwargs=shared_kwargs
    )
    second = gs.MCMCSpec(
        gs.NUTSKernel, kernel_group="joint", kernel_kwargs=shared_kwargs
    )
    first_config = {"nuts": first}
    x = lsl.Var.new_param(0.0, name="x", inference=mapping_type(first_config))
    y = lsl.Var.new_param(0.0, name="y", inference=mapping_type({"nuts": second}))
    first_config.clear()
    assert x.get_inference("nuts") is first
    assert first.kernel_kwargs is shared_kwargs
    assert second.kernel_kwargs is shared_kwargs
    groups = gs.LieselMCMC(lsl.Model(x, y), which="nuts").get_kernel_groups()
    assert len(groups) == 1
    assert set(groups["joint"].position_keys) == {"x", "y"}
    assert groups["joint"].kwargs is shared_kwargs

    # Direct attribute assignment intentionally keeps normal reference semantics.
    replacement = {"nuts": second}
    x.inference = replacement
    assert x.inference is replacement


def test_transform_snapshots_inference_mapping():
    x = lsl.Var.new_param(1.0, lsl.Dist(tfd.Exponential, rate=1.0), name="x")
    spec = gs.MCMCSpec(gs.NUTSKernel)
    inference = {"nuts": spec}
    transformed = x.transform(tfb.Exp(), inference=ReadOnlyMapping(inference))
    inference.clear()
    assert transformed.get_inference("nuts") is spec


def test_plot_accepts_sample_and_palette_mappings():
    samples: Mapping[str, jax.Array] = ReadOnlyMapping(
        {"x": jnp.arange(20.0).reshape(2, 10)}
    )
    palette: Mapping[int, str] = MappingProxyType({0: "red", 1: "blue"})
    try:
        plot = plot_trace(samples, color_palette=palette, show=False)
        assert plot is not None
        assert plot.data is not None
        assert set(plot.data["chain_index"]) == {0, 1}
    finally:
        plt.close("all")


@pytest.mark.parametrize("kernel_type", [gs.HMCKernel, gs.NUTSKernel])
def test_tuning_accepts_mapping_history(kernel_type):
    interface = gs.DictInterface(lambda state: -(state["x"] ** 2) / 2.0)
    kernel = kernel_type(["x"], initial_step_size=0.1)
    kernel.set_model(interface)
    key = jax.random.key(0)
    state = {"x": jnp.array(0.0)}
    kernel_state = kernel.init_state(key, state)
    epoch = gs.EpochConfig(gs.EpochType.SLOW_ADAPTATION, 5, 1, None).to_state(1, 0)
    values = jnp.arange(5.0)
    history: gs.PositionInput = ReadOnlyMapping({"x": values})
    tuned = kernel.tune(key, kernel_state, state, epoch, history)
    np.testing.assert_allclose(tuned.kernel_state.inverse_mass_matrix, [2.501])
    assert history["x"] is values
