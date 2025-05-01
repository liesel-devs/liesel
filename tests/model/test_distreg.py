from collections.abc import Generator

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import random as jrd

import liesel.goose as gs
import liesel.model.distreg as dr
from liesel.model.nodes import Var

rng = np.random.default_rng(1337)
prng_key = jrd.PRNGKey(42)
n = 30
p = 2


@pytest.fixture
def X() -> Generator:
    X = np.asarray(rng.uniform(size=[n, p - 1]), dtype=np.float32)
    yield jnp.column_stack([jnp.ones(n), X])


@pytest.fixture
def y(X: jnp.ndarray):
    beta = jnp.ones(X.shape[1])
    gamma = jnp.array([0.1] * X.shape[1])

    sigma = jnp.exp(X @ gamma)
    y = rng.normal(X @ beta, sigma, size=n)
    yield np.asarray(y, dtype=np.float32)


@pytest.mark.filterwarnings(
    "ignore:.*Deprecated in v.0.4.0. "
    "Will be removed in a future release.*:FutureWarning"
)
class TestDistRegBuilder:
    def test_init(self) -> None:
        drb = dr.DistRegBuilder()
        assert drb is not None

    def test_add_response(self, y) -> None:
        drb = dr.DistRegBuilder().add_response(y, tfd.Normal)

        assert np.allclose(drb.response.value, y)

    def test_add_predictor(self, y) -> None:
        drb = (
            dr.DistRegBuilder()
            .add_response(y, tfd.Normal)
            .add_predictor("loc", tfb.Identity)
        )

        assert "loc" in drb._distributional_parameters
        assert "loc" in drb.response.dist_node.kwinputs  # type: ignore

    def test_add_p_smooth(self, y, X) -> None:
        drb = (
            dr.DistRegBuilder()
            .add_response(y, tfd.Normal)
            .add_predictor("loc", tfb.Identity)
            .add_p_smooth(X, m=0.0, s=0.0, predictor="loc", name="x")
        )

        assert "x" in drb.groups()
        smooth_value = drb.groups()["x"]["smooth"].update().value
        predictor_value = drb._predictors["loc"].update().value
        assert np.allclose(smooth_value, predictor_value)

    def test_add_np_smooth(self, y, X) -> None:
        K = jnp.eye(X.shape[1])
        drb = (
            dr.DistRegBuilder()
            .add_response(y, tfd.Normal)
            .add_predictor("loc", tfb.Identity)
            .add_np_smooth(X, K=K, a=0.5, b=0.001, predictor="loc", name="x")
        )

        assert "x" in drb.groups()
        smooth_value = drb.groups()["x"]["smooth"].update().value
        predictor_value = drb._predictors["loc"].update().value
        assert np.allclose(smooth_value, predictor_value)

    def test_build_model(self, y, X) -> None:
        drb = (
            dr.DistRegBuilder()
            .add_response(y, tfd.Normal)
            .add_predictor("loc", tfb.Identity)
            .add_predictor("scale", tfb.Exp)
            .add_p_smooth(X, m=0.0, s=0.0, predictor="loc", name="xloc")
            .add_p_smooth(X, m=0.0, s=0.0, predictor="scale", name="xscale")
        )

        model = drb.build_model()

        assert "xloc" in model.groups()
        assert "xscale" in model.groups()

    def test_build_empty(self, local_caplog) -> None:
        """An empty model can be built with a warning."""
        with local_caplog() as caplog:
            drb = dr.DistRegBuilder()
            model = drb.build_model()
            assert len(caplog.records) == 1
            assert caplog.records[0].levelname == "WARNING"
            assert model

    def test_wrong_order_not_allowed(self, y, X) -> None:
        with pytest.raises(RuntimeError, match="No response"):
            dr.DistRegBuilder().add_predictor("loc", tfb.Identity)

        with pytest.raises(RuntimeError, match="No predictor 'loc' found."):
            drb = dr.DistRegBuilder().add_response(y, tfd.Normal)
            drb.add_p_smooth(X, m=0.0, s=0.0, predictor="loc", name="xloc")

    def test_name(self, y, X) -> None:
        drb = (
            dr.DistRegBuilder()
            .add_response(y, tfd.Normal)
            .add_predictor("loc", tfb.Identity)
        )

        # returns 'name' argument, if the name is free
        name = drb._smooth_name(name="x", predictor="loc", prefix="p")
        assert name == "x"

        # raises error if manually selected name is taken
        drb.add_p_smooth(X, m=0.0, s=10.0, predictor="loc", name="x")
        with pytest.raises(RuntimeError, match="already exists"):
            drb._smooth_name(name="x", predictor="loc", prefix="p")

        # automatically assigns name
        name = drb._smooth_name(name=None, predictor="loc", prefix="p")
        assert name == "loc_p0"

        # up-counting works
        drb.add_p_smooth(X, m=0.0, s=10.0, predictor="loc")
        name = drb._smooth_name(name=None, predictor="loc", prefix="p")
        assert name == "loc_p1"


@pytest.fixture
def drb(y, X):
    drb = (
        dr.DistRegBuilder()
        .add_response(y, tfd.Normal)
        .add_predictor("loc", tfb.Identity)
        .add_predictor("scale", tfb.Exp)
        .add_p_smooth(X, m=0.0, s=0.0, predictor="loc", name="xloc")
        .add_p_smooth(X, m=0.0, s=0.0, predictor="scale", name="xscale")
    )

    yield drb


@pytest.fixture
def drb_np_smooth(y, X):
    K = jnp.eye(X.shape[1])
    drb = (
        dr.DistRegBuilder()
        .add_response(y, tfd.Normal)
        .add_predictor("loc", tfb.Identity)
        .add_predictor("scale", tfb.Exp)
        .add_np_smooth(X, K=K, a=0.5, b=0.001, predictor="loc", name="xloc")
        .add_np_smooth(X, K=K, a=0.5, b=0.001, predictor="scale", name="xscale")
    )

    yield drb


def construct_model_state_from_np_smooth(group: dict[str, Var]) -> dict:
    model_state = {}
    model_state[group["a"].value_node.name] = group["a"]
    model_state[group["rank"].value_node.name] = group["rank"]
    model_state[group["b"].value_node.name] = group["b"]
    model_state[group["beta"].value_node.name] = group["beta"]
    model_state[group["K"].value_node.name] = group["K"]
    return model_state


@pytest.mark.filterwarnings(
    "ignore:.*Deprecated in v.0.4.0. "
    "Will be removed in a future release.*:FutureWarning"
)
class TestTau2GibbsKernel:
    def test_return_gibbs_kernel(self, drb_np_smooth: dr.DistRegBuilder) -> None:
        """The function should successfully return a Gibbs kernel."""
        g = drb_np_smooth.build_model().groups()["xloc"]
        kernel = dr.tau2_gibbs_kernel(g)
        assert isinstance(kernel, gs.GibbsKernel)

    def test_plausible_transition(self, drb_np_smooth) -> None:
        """
        The transition outcome of a single Gibbs transition should be plausible, i.e.
        the drawn value should be within the support of the inverse gamma distribution.
        """
        g = drb_np_smooth.build_model().groups()["xloc"]
        kernel = dr.tau2_gibbs_kernel(g)

        kernel.set_model(gs.DictInterface(lambda model_state: 0.0))
        model_state = construct_model_state_from_np_smooth(g)

        epoch_config = gs.EpochConfig(
            gs.EpochType.POSTERIOR, duration=1, thinning=1, optional=None
        )

        epoch_state = epoch_config.to_state(nth_epoch=0, time_before_epoch=0)

        outcome = kernel.transition(
            prng_key, kernel_state={}, model_state=model_state, epoch=epoch_state
        )

        assert outcome.model_state[g["tau2"].name] > 0.0


@pytest.mark.filterwarnings(
    "ignore:.*Deprecated in v.0.4.0. "
    "Will be removed in a future release.*:FutureWarning"
)
class TestDistRegMCMC:
    def test_dist_reg_mcmc_p_smooth(self, drb: dr.DistRegBuilder) -> None:
        """
        When used with a DistRegBuilder that includes only parametric smooths,
        the function returns an engine builder with two IWLS kernels.
        """
        model = drb.build_model()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        assert len(ebuilder.kernels) == 2
        assert all([isinstance(k, gs.IWLSKernel) for k in ebuilder.kernels])

    def test_dist_reg_mcmc_build_p_smooth(self, drb) -> None:
        """
        When used with a DistRegBuilder that includes only parametric smooths,
        the returned engine builder successfully builds an engine.
        """
        model = drb.build_model()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        ebuilder.set_duration(1000, 1000)
        engine = ebuilder.build()
        assert engine

        engine.sample_next_epoch()
        assert engine.get_results()

    def test_dist_reg_mcmc_np_smooth(self, drb_np_smooth: dr.DistRegBuilder) -> None:
        """
        When used with a DistRegBuilder that includes two
        non-parametric smooths, the function returns an engine builder with two
        IWLS kernels and two Gibbs kernel.
        """
        drb = drb_np_smooth
        model = drb.build_model()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        assert len(ebuilder.kernels) == 4
        assert sum(isinstance(k, gs.IWLSKernel) for k in ebuilder.kernels) == 2
        assert sum(isinstance(k, gs.GibbsKernel) for k in ebuilder.kernels) == 2

    def test_dist_reg_mcmc_build_np_smooth(
        self, drb_np_smooth: dr.DistRegBuilder
    ) -> None:
        """
        When used with a DistRegBuilder that includes two
        non-parametric smooths, the engine builder successfully builds an engine.
        """
        drb = drb_np_smooth
        model = drb.build_model()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        ebuilder.set_duration(1000, 100)
        engine = ebuilder.build()
        assert engine
