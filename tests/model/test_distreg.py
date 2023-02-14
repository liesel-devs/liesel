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


@pytest.fixture
def drb(X: jnp.ndarray, y: jnp.ndarray):
    drb = dr.DistRegBuilder()
    m = jnp.zeros(X.shape[1])
    s = jnp.array([10.0] * (X.shape[1]))

    drb.add_p_smooth(X, m=m, s=s, predictor="loc")
    drb.add_p_smooth(X, m=m, s=s, predictor="scale")

    drb.add_predictor("loc", tfb.Identity)
    drb.add_predictor("scale", tfb.Exp)

    drb.add_response(y, tfd.Normal)

    yield drb


@pytest.fixture
def drb_np_smooth(X: jnp.ndarray, y: jnp.ndarray):
    m = jnp.zeros(X.shape[1])
    s = jnp.array([10.0] * (X.shape[1]))

    drb = dr.DistRegBuilder()

    K = jnp.eye(X.shape[1])
    name = "np_smooth_test"
    drb.add_np_smooth(X, K, a=1.0, b=0.001, predictor="loc", name=name)
    drb.add_p_smooth(X, m=m, s=s, predictor="scale")

    drb.add_predictor("loc", tfb.Identity)
    drb.add_predictor("scale", tfb.Exp)

    drb.add_response(y, tfd.Normal)

    yield drb


def construct_model_state_from_np_smooth(group: dict[str, Var]) -> dict:
    model_state = {}
    model_state[group["a"].value_node.name] = group["a"]
    model_state[group["rank"].value_node.name] = group["rank"]
    model_state[group["b"].value_node.name] = group["b"]
    model_state[group["beta"].value_node.name] = group["beta"]
    model_state[group["K"].value_node.name] = group["K"]
    return model_state


class TestDistRegBuilder:
    def test_add_p_smooth(self, X: jnp.ndarray) -> None:
        """Tests basic plausibility of adding a parametric smooth."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = jnp.zeros(p)
        s = jnp.array([10.0] * (p))
        name = "linreg"

        group = drb.add_p_smooth(X, m=m, s=s, predictor="loc", name=name)  # noqa: F841
        expected_names = [name + x for x in ["_X", "_m", "_s", "_beta"]]
        node_names = [node.name for node in drb.vars]

        # assert group is drb.groups[0]
        # assert len(drb.groups) == 1
        assert len(drb.vars) == 5
        # assert drb.groups[0].name == name
        assert all([name in node_names for name in expected_names])

    def test_add_p_smooth_no_name(self, X: jnp.ndarray) -> None:
        """If no name is provided, a name should be generated automatically."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = jnp.zeros(p)
        s = jnp.array([10.0] * (p))

        drb.add_p_smooth(X, m=m, s=s, predictor="loc")
        # assert drb.groups[0].name == "loc_p0"

    def test_add_p_smooth_two_smooths(self, X: jnp.ndarray) -> None:
        """It should be possible to add two smooths to the DistRegBuilder."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = jnp.zeros(p)
        s = jnp.array([10.0] * (p))

        group1 = drb.add_p_smooth(X, m=m, s=s, predictor="loc")  # noqa: F841
        group2 = drb.add_p_smooth(X, m=m, s=s, predictor="loc")  # noqa: F841

        # assert group1 is not group2
        # assert drb.groups[0].name != drb.groups[1].name
        assert len(drb.vars) == 10

    def test_add_p_smooth_two_smooths_equal_names(self, X: jnp.ndarray) -> None:
        """It should not be possible to add two smooths with equal names."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = jnp.zeros(p)
        s = jnp.array([10.0] * (p))

        drb.add_p_smooth(X, m=m, s=s, predictor="loc", name="test")

        with pytest.raises(RuntimeError):
            drb.add_p_smooth(X, m=m, s=s, predictor="loc", name="test")

        len_before_build = len(drb.vars)
        drb.build_model()
        len_after_build = len(drb.vars)

        assert len_before_build == 5
        assert len_after_build == 0

    def test_add_predictor(self, X: jnp.ndarray) -> None:
        """Tests basic plausibility of adding a predictor corresponding to a smooth."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = jnp.zeros(p)
        s = jnp.array([10.0] * (p))
        name = "linreg"

        drb.add_p_smooth(X, m=m, s=s, predictor="loc", name=name)
        drb_return = drb.add_predictor("loc", tfb.Identity)

        node_names = [node.name for node in drb.vars]

        assert drb_return is drb
        assert len(drb.vars) == 7
        assert "loc_pdt" in node_names
        assert "loc" in node_names

    def test_add_predictor_wrong_name(self, X: jnp.ndarray) -> None:
        """
        Adding a predictor with a name that is not already associated with a smooth
        should lead to an error.
        """
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = jnp.zeros(p)
        s = jnp.array([10.0] * (p))
        name = "linreg"

        drb.add_p_smooth(X, m=m, s=s, predictor="loc", name=name)

        with pytest.raises(RuntimeError):
            drb.add_predictor("wrong_name", tfb.Identity)

    def test_add_np_smooth(self, X: jnp.ndarray) -> None:
        """
        Tests basic plausibility of adding a non-parametric smooth.

        Note: This test does not include a valid matrix of basis function evaluations,
        it simply reuses the design matrix X. For the purpose of this test, that should
        be fine.
        """
        drb = dr.DistRegBuilder()

        K = jnp.eye(X.shape[1])
        name = "np_smooth_test"
        group = drb.add_np_smooth(  # noqa: F841
            X, K, a=1.0, b=0.001, predictor="loc", name=name
        )

        expected_names = [name + x for x in ["_X", "_K", "_a", "_b", "_tau2", "_beta"]]
        node_names = [node.name for node in drb.vars]

        # assert len(drb.groups) == 1
        # assert drb.groups[0] is group
        assert all([name in node_names for name in expected_names])

    def test_add_np_smooth_beta(self, X: jnp.ndarray) -> None:
        """
        Asserts that beta is initialized with zeros and has the expected inputs.

        Note: This test does not include a valid matrix of basis function evaluations,
        it simply reuses the design matrix X. For the purpose of this test, that should
        be fine.
        """
        drb = dr.DistRegBuilder()

        K = jnp.eye(X.shape[1])
        name = "np_smooth_test"
        drb.add_np_smooth(X, K, a=1.0, b=0.001, predictor="loc", name=name)

        beta = [n for n in drb.vars if n.name == name + "_beta"][0]
        expected_input_names = [
            name + x for x in ["_K_var_value", "_rank_var_value", "_tau2_var_value"]
        ]
        input_names = [ip.name for ip in list(beta.all_input_nodes())]

        assert all(beta.value == jnp.zeros(jnp.shape(X)[1]))
        assert all([name in input_names for name in expected_input_names])

    def test_add_np_smooth_tau2(self, X: jnp.ndarray) -> None:
        """Asserts that `tau2` is initialized with a value of 10000."""
        drb = dr.DistRegBuilder()

        K = jnp.eye(X.shape[1])
        name = "np_smooth_test"
        drb.add_np_smooth(X, K, a=1.0, b=0.001, predictor="loc", name=name)

        tau2 = [n for n in drb.vars if n.name == name + "_tau2"][0]

        assert tau2.value == 10_000.0

    def test_add_response(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """Tests basic plausibility of adding a response."""
        drb = dr.DistRegBuilder()
        m = jnp.zeros(X.shape[1])
        s = jnp.array([10.0] * (X.shape[1]))

        drb.add_p_smooth(X, m=m, s=s, predictor="loc")
        drb.add_p_smooth(X, m=m, s=s, predictor="scale")

        drb.add_predictor("loc", tfb.Identity)
        drb.add_predictor("scale", tfb.Exp)

        drb.add_response(y, tfd.Normal)

        assert drb.response
        assert drb.build_model()

    def test_add_response_first(self, y: jnp.ndarray) -> None:
        """
        Adding a response before the smooths and predictors are set leads to an error.
        """
        drb = dr.DistRegBuilder()
        with pytest.raises(RuntimeError):
            drb.add_response(y, tfd.Normal)

    def test_build_empty(self, local_caplog) -> None:
        """An empty model can be built with a warning."""
        with local_caplog() as caplog:
            drb = dr.DistRegBuilder()
            model = drb.build_model()
            assert len(caplog.records) == 1
            assert caplog.records[0].levelname == "WARNING"
            assert model


class TestTau2GibbsKernel:
    def test_return_gibbs_kernel(self, drb_np_smooth: dr.DistRegBuilder) -> None:
        """The function should successfully return a Gibbs kernel."""
        g = drb_np_smooth.build_model().groups()["np_smooth_test"]
        kernel = dr.tau2_gibbs_kernel(g)
        assert isinstance(kernel, gs.GibbsKernel)

    def test_plausible_transition(self, drb_np_smooth) -> None:
        """
        The transition outcome of a single Gibbs transition should be plausible, i.e.
        the drawn value should be within the support of the inverse gamma distribution.
        """
        g = drb_np_smooth.build_model().groups()["np_smooth_test"]
        kernel = dr.tau2_gibbs_kernel(g)

        def log_prob(model_state):
            return 0.0

        kernel.set_model(gs.DictModel(log_prob))
        model_state = construct_model_state_from_np_smooth(g)

        epoch_config = gs.EpochConfig(
            gs.EpochType.POSTERIOR, duration=1, thinning=1, optional=None
        )

        epoch_state = epoch_config.to_state(nth_epoch=0, time_before_epoch=0)

        outcome = kernel.transition(
            prng_key, kernel_state={}, model_state=model_state, epoch=epoch_state
        )

        assert outcome.model_state[g["tau2"].name] > 0.0


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
        When used with a DistRegBuilder that includes one parametric and one
        non-parametric smooth, the function returns an engine builder with two
        IWLS kernels and a Gibbs kernel.
        """
        drb = drb_np_smooth
        model = drb.build_model()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        assert len(ebuilder.kernels) == 3
        assert sum(isinstance(k, gs.IWLSKernel) for k in ebuilder.kernels) == 2
        assert sum(isinstance(k, gs.GibbsKernel) for k in ebuilder.kernels) == 1

    def test_dist_reg_mcmc_build_np_smooth(
        self, drb_np_smooth: dr.DistRegBuilder
    ) -> None:
        """
        When used with a DistRegBuilder that includes one parametric and one
        non-parametric smooth, the engine builder successfully builds an engine.
        """
        drb = drb_np_smooth
        model = drb.build_model()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        ebuilder.set_duration(1000, 100)
        engine = ebuilder.build()
        assert engine
