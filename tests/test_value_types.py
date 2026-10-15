"""Runtime and static contracts for numerical values and arbitrary pytrees."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, assert_type

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.numpy.distributions as nfd

import liesel
import liesel.goose as gs
import liesel.model as lsl
from liesel.contrib.splines import basis_matrix, equidistant_knots
from liesel.distributions.mvn_degen import (
    Array as DistributionArray,
)
from liesel.distributions.mvn_degen import (
    MultivariateNormalDegenerate,
    _log_pdet,
    _rank,
)
from liesel.goose.interface_log_prob import Array as InterfaceArray
from liesel.goose.iwls_utils import mvn_log_prob
from liesel.goose.optim import array_to_dict
from liesel.goose.types import Array as GooseArray
from liesel.types import PositionInput, PyTree


def test_legacy_aliases_and_shared_pytree() -> None:
    model_value: lsl.Array = {"nested": [1.0]}
    goose_value: GooseArray = model_value
    interface_value: InterfaceArray = goose_value
    distribution_value: DistributionArray = interface_value
    assert_type(distribution_value, Any)
    assert distribution_value is model_value
    assert liesel.PyTree is lsl.PyTree is gs.PyTree is PyTree is Any


class Coordinates(NamedTuple):
    a: float
    b: jax.Array


def test_nested_derivatives_and_model_values() -> None:
    coordinates = Coordinates(2.0, jnp.array([3.0, 4.0]))
    position: PositionInput = MappingProxyType({"x": coordinates})
    interface = gs.DictInterface(
        lambda state: -0.5 * (state["x"].a ** 2 + jnp.sum(state["x"].b ** 2))
    )
    log_prob = gs.InterfaceLogProb(interface, dict(position))
    gradient = log_prob.grad(position)
    hessian = log_prob.hessian(position)
    assert_type(gradient, dict[str, Any])
    assert_type(hessian, dict[str, Any])
    assert isinstance(gradient["x"], Coordinates)
    np.testing.assert_allclose(gradient["x"].a, -2.0)
    np.testing.assert_allclose(gradient["x"].b, [-3.0, -4.0])
    np.testing.assert_allclose(hessian["x"].a["x"].a, -1.0)

    var = lsl.Var(coordinates, name="x", convert=lambda x: x)
    model = lsl.Model(var)
    updated = model.update_state(position)
    assert model.extract_position(["x"], updated)["x"] is coordinates


@pytest.mark.parametrize("interface", [False, True])
def test_flat_derivatives_have_array_outputs(interface: bool) -> None:
    var = lsl.Var.new_param(
        jnp.zeros(2), lsl.Dist(tfd.Normal, loc=0.0, scale=1.0), name="x"
    )
    model = lsl.Model(var)
    if interface:
        log_prob = gs.FlatInterfaceLogProb(
            gs.LieselInterface(model), model.state, ["x"]
        )
    else:
        log_prob = lsl.FlatLogProb(model, ["x"])
    position = np.array([1.0, 2.0], dtype=np.float32)
    gradient = log_prob.grad(position)
    hessian = log_prob.hessian(position)
    assert_type(gradient, jax.Array)
    assert_type(hessian, jax.Array)
    assert isinstance(gradient, jax.Array)
    np.testing.assert_allclose(gradient, -position)
    np.testing.assert_allclose(hessian, -np.eye(2))
    np.testing.assert_allclose(jax.jit(log_prob.grad)(position), gradient)
    if TYPE_CHECKING:
        flat = lsl.FlatLogProb(model, ["x"])
        flat.log_prob(object())  # ty: ignore[invalid-argument-type]
        flat_interface = gs.FlatInterfaceLogProb(
            gs.LieselInterface(model), model.state, ["x"]
        )
        flat_interface.log_prob(object())  # ty: ignore[invalid-argument-type]


def test_scalar_and_numpy_log_prob_outputs_are_preserved() -> None:
    empty_distribution = lsl.Var(1.0, name="x")
    model = lsl.Model(empty_distribution)
    assert type(model.log_prob) is int
    assert type(empty_distribution.log_prob) is float

    var = lsl.Var.new_param(
        np.ones(2),
        lsl.Dist(nfd.Normal, loc=0.0, scale=1.0),
        name="y",
        convert=lambda x: x,
    )
    numpy_model = lsl.Model(var)
    assert isinstance(var.log_prob, np.ndarray)
    assert isinstance(numpy_model.log_prob, np.number)

    scalar = np.float32(-1.0)
    interface = gs.DictInterface(lambda state: scalar)
    assert interface.log_prob({}) is scalar
    assert gs.InterfaceLogProb(interface, {}).log_prob({}) is scalar


def test_distribution_inputs_and_preserved_metadata() -> None:
    rank = np.int32(2)
    log_pdet = np.float32(0.0)
    dist = MultivariateNormalDegenerate(
        loc=[0.0, 0.0], prec=np.eye(2, dtype=np.float32), rank=rank, log_pdet=log_pdet
    )
    assert_type(dist.loc, jax.Array)
    assert_type(dist.prec, jax.Array)
    assert isinstance(dist.loc, jax.Array)
    if TYPE_CHECKING:
        MultivariateNormalDegenerate(object(), np.eye(2))  # ty: ignore[invalid-argument-type]
    assert dist.rank is rank
    assert dist.log_pdet is log_pdet
    assert dist.dtype == np.dtype("float32")
    assert MultivariateNormalDegenerate(0.0, np.eye(2)).dtype == np.dtype("float64")
    np.testing.assert_allclose(dist.log_prob(np.ones(2)), -np.log(2 * np.pi) - 1)
    eigenvalues = np.array([1.0, 2.0], dtype=np.float32)
    result = _log_pdet(eigenvalues, rank=2)
    assert_type(result, jax.Array)
    assert_type(_rank(eigenvalues), jax.Array)
    np.testing.assert_allclose(result, np.log(2.0))


@pytest.mark.parametrize("constructor", [np.asarray, jnp.asarray])
def test_array_to_dict_preserves_array_backend(constructor) -> None:
    vector = constructor([1.0, 2.0])
    assert array_to_dict(vector)["x"] is vector
    matrix = constructor([[1.0, 2.0], [3.0, 4.0]])
    columns = array_to_dict(matrix)
    assert type(columns["x0"]) is type(matrix)
    np.testing.assert_array_equal(columns["x1"], [2.0, 4.0])


def test_array_to_dict_specific_types_and_scalars() -> None:
    assert_type(array_to_dict(jnp.ones(2)), dict[str, jax.Array])
    assert_type(array_to_dict(1.0), dict[str, float])
    if TYPE_CHECKING:
        array_to_dict(object())  # ty: ignore[invalid-argument-type]
    assert array_to_dict(1) == {"x": 1}
    assert array_to_dict(np.float32(2.0))["x"] == 2.0
    with pytest.raises(ValueError, match="ndim <= 2"):
        array_to_dict(jnp.ones((2, 2, 2)))


def test_numerical_helpers_accept_scalar_inputs() -> None:
    density = mvn_log_prob(
        np.ones(1, dtype=np.float32), 0.0, np.eye(1, dtype=np.float32)
    )
    assert_type(density, jax.Array)
    if TYPE_CHECKING:
        mvn_log_prob(object(), 0.0, np.eye(1))  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(density, -0.5 * (np.log(2 * np.pi) + 1))
    spec = gs.MCMCSpec(gs.HMCKernel, jitter_dist=tfd.Normal(0.0, 1.0))
    jittered = spec.apply_jitter(jax.random.key(0), 1.0)
    assert isinstance(jittered, jax.Array)
    assert jittered.shape == ()
    no_jitter = gs.MCMCSpec(gs.HMCKernel)
    value = {"nested": [1.0]}
    assert no_jitter.apply_jitter(jax.random.key(0), value) is value


def test_summary_accepts_sequence_samples_without_mutation() -> None:
    values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    samples = MappingProxyType({"x": values})
    summary = gs.SamplesSummary(samples, which=("mean",))
    assert samples["x"] is values
    assert summary.sample_info == {"num_chains": 2, "sample_size_per_chain": 3}
    np.testing.assert_allclose(summary.quantities["mean"]["x"], 3.5)


def test_spline_inputs_keep_sequence_support() -> None:
    knots = equidistant_knots(np.linspace(-1.0, 1.0, 10), n_param=5)
    basis = basis_matrix([-0.5, 0.5], knots)
    assert_type(knots, jax.Array)
    assert_type(basis, jax.Array)
    if TYPE_CHECKING:
        basis_matrix(object(), knots)  # ty: ignore[invalid-argument-type]
    assert basis.shape == (2, 5)
