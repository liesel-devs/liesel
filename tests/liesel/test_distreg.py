import copy

import numpy as np
import pytest
from jax import random as jrd

import liesel.goose as gs
import liesel.liesel.distreg as dr

rng = np.random.default_rng(1337)
prng_key = jrd.PRNGKey(42)
n = 30
p = 2


@pytest.fixture
def X():
    yield np.column_stack([np.ones(n), rng.uniform(size=[n, p - 1])])


@pytest.fixture
def y(X):
    beta = np.ones(X.shape[1])
    gamma = np.array([0.1] * X.shape[1])

    sigma = np.exp(X @ gamma)
    y = rng.normal(X @ beta, sigma, size=n)

    yield y


@pytest.fixture
def drb(X, y):
    drb = dr.DistRegBuilder()
    m = np.zeros(X.shape[1])
    s = np.array([10] * (X.shape[1]))

    drb.add_p_smooth(X, m=m, s=s, predictor="loc")
    drb.add_p_smooth(X, m=m, s=s, predictor="scale")

    drb.add_predictor("loc", "Identity")
    drb.add_predictor("scale", "Exp")

    drb.add_response(y, "Normal")

    yield drb


@pytest.fixture
def drb_np_smooth(X, y):
    m = np.zeros(X.shape[1])
    s = np.array([10] * (X.shape[1]))

    drb = dr.DistRegBuilder()

    K = np.eye(X.shape[1])
    name = "np_smooth_test"
    drb.add_np_smooth(X, K, a=1, b=0.001, predictor="loc", name=name)
    drb.add_p_smooth(X, m=m, s=s, predictor="scale")

    drb.add_predictor("loc", "Identity")
    drb.add_predictor("scale", "Exp")

    drb.add_response(y, "Normal")

    yield drb


def construct_model_state_from_np_smooth(group: dr.NPSmoothGroup) -> dict:
    model_state = {}
    model_state[group["a"].name] = group["a"]
    model_state[group["rank"].name] = group["rank"]
    model_state[group["b"].name] = group["b"]
    model_state[group["beta"].name] = group["beta"]
    model_state[group["K"].name] = group["K"]
    return model_state


class TestDistRegBuilder:
    def test_add_p_smooth(self, X):
        """Tests basic plausibility of adding a parametric smooth."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = np.zeros(p)
        s = np.array([10] * (p))
        name = "linreg"

        group = drb.add_p_smooth(X, m=m, s=s, predictor="loc", name=name)
        expected_names = [name + x for x in ["_X", "_m", "_s", "_beta"]]
        node_names = [node.name for node in drb.all_nodes()]

        assert group is drb.groups[0]
        assert len(drb.groups) == 1
        assert len(drb.all_nodes()) == 5
        assert drb.groups[0].name == name
        assert all([name in node_names for name in expected_names])

    def test_add_p_smooth_no_name(self, X):
        """If no name is provided, a name should be generated automatically."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = np.zeros(p)
        s = np.array([10] * (p))

        drb.add_p_smooth(X, m=m, s=s, predictor="loc")
        assert drb.groups[0].name == "loc_p0"

    def test_add_p_smooth_two_smooths(self, X):
        """It should be possible to add two smooths to the DistRegBuilder."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = np.zeros(p)
        s = np.array([10] * (p))

        group1 = drb.add_p_smooth(X, m=m, s=s, predictor="loc")
        group2 = drb.add_p_smooth(X, m=m, s=s, predictor="loc")

        assert group1 is not group2
        assert drb.groups[0].name != drb.groups[1].name
        assert len(drb.all_nodes()) == 10

    def test_add_p_smooth_two_smooths_equal_names(self, X):
        """It should not be possible to add two smooths with equal names."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = np.zeros(p)
        s = np.array([10] * (p))

        drb.add_p_smooth(X, m=m, s=s, predictor="loc", name="test")

        with pytest.raises(RuntimeError):
            drb.add_p_smooth(X, m=m, s=s, predictor="loc", name="test")

        len_before_build = len(drb.all_nodes())
        drb.build()
        len_after_build = len(drb.all_nodes())

        assert len_before_build == len_after_build

    def test_add_predictor(self, X):
        """Tests basic plausibility of adding a predictor corresponding to a smooth."""
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = np.zeros(p)
        s = np.array([10] * (p))
        name = "linreg"

        drb.add_p_smooth(X, m=m, s=s, predictor="loc", name=name)
        drb_return = drb.add_predictor("loc", "Identity")

        node_names = [node.name for node in drb.all_nodes()]

        assert drb_return is drb
        assert len(drb.all_nodes()) == 7
        assert "loc_pdt" in node_names
        assert "loc" in node_names

    def test_add_predictor_wrong_name(self, X):
        """
        Adding a predictor with a name that is not already associated with a smooth
        should lead to an error.
        """
        p = X.shape[1]
        drb = dr.DistRegBuilder()

        m = np.zeros(p)
        s = np.array([10] * (p))
        name = "linreg"

        drb.add_p_smooth(X, m=m, s=s, predictor="loc", name=name)

        with pytest.raises(RuntimeError):
            drb.add_predictor("wrong_name", "Identity")

    def test_add_np_smooth(self, X):
        """
        Tests basic plausibility of adding a non-parametric smooth.

        Note: This test does not include a valid matrix of basis function evaluations,
        it simply reuses the design matrix X. For the purpose of this test, that should
        be fine.
        """
        drb = dr.DistRegBuilder()

        K = np.eye(X.shape[1])
        name = "np_smooth_test"
        group = drb.add_np_smooth(X, K, a=1, b=0.001, predictor="loc", name=name)

        expected_names = [name + x for x in ["_X", "_K", "_a", "_b", "_tau2", "_beta"]]
        node_names = [node.name for node in drb.all_nodes()]

        assert len(drb.groups) == 1
        assert drb.groups[0] is group
        assert all([name in node_names for name in expected_names])

    def test_add_np_smooth_beta(self, X):
        """
        Asserts that beta is initialized with zeros and has the expected inputs.

        Note: This test does not include a valid matrix of basis function evaluations,
        it simply reuses the design matrix X. For the purpose of this test, that should
        be fine.
        """
        drb = dr.DistRegBuilder()

        K = np.eye(X.shape[1])
        name = "np_smooth_test"
        drb.add_np_smooth(X, K, a=1, b=0.001, predictor="loc", name=name)

        beta = [n for n in drb.all_nodes() if n.name == name + "_beta"][0]
        expected_input_names = [name + x for x in ["_K", "_rank", "_tau2"]]
        input_names = [ip.name for ip in list(beta.inputs)]

        assert all(beta.value == np.zeros(np.shape(X)[1], np.float32))
        assert all([name in input_names for name in expected_input_names])

    def test_add_np_smooth_tau2(self, X):
        """Asserts that `tau2` is initialized with a value of 10000."""
        drb = dr.DistRegBuilder()

        K = np.eye(X.shape[1])
        name = "np_smooth_test"
        drb.add_np_smooth(X, K, a=1, b=0.001, predictor="loc", name=name)

        tau2 = [n for n in drb.all_nodes() if n.name == name + "_tau2"][0]

        assert tau2.value == 10_000.0

    def test_add_response(self, X, y):
        """Tests basic plausibility of adding a response."""
        drb = dr.DistRegBuilder()
        m = np.zeros(X.shape[1])
        s = np.array([10] * (X.shape[1]))

        drb.add_p_smooth(X, m=m, s=s, predictor="loc")
        drb.add_p_smooth(X, m=m, s=s, predictor="scale")

        drb.add_predictor("loc", "Identity")
        drb.add_predictor("scale", "Exp")

        drb.add_response(y, "Normal")

        assert drb.response
        assert drb.build()

    def test_add_response_first(self, y):
        """
        Adding a response before the smooths and predictors are set leads to an error.
        """
        drb = dr.DistRegBuilder()
        with pytest.raises(RuntimeError):
            drb.add_response(y, "Normal")

    def test_build_empty(self, local_caplog):
        """An empty model can be built with a warning."""
        with local_caplog():
            drb = dr.DistRegBuilder()
            model = drb.build()
            assert len(local_caplog.records) == 1
            assert local_caplog.records[0].levelname == "WARNING"
            assert model


class TestCopRegBuilder:
    def test_init_copula_builder(self, drb):
        crb = dr.CopRegBuilder(drb, drb)

        assert len(crb.groups) == 2 * len(drb.groups)
        assert len(crb.nodes) == 2 * len(drb.nodes)
        assert len(crb.all_nodes()) == len(drb.all_nodes())

    def test_init_copula_builder_with_models(self, drb):
        """
        The CopRegBuilder must be initialized with DistRegBuilders, not with models.
        """
        model = drb.build()
        with pytest.raises(RuntimeError):
            dr.CopRegBuilder(model, model)

    def test_add_copula(self, drb):
        """
        Tests that the `add_copula` method runs and adds a plausible number of nodes.
        """
        drb2 = copy.deepcopy(drb)
        crb = dr.CopRegBuilder(drb, drb2)
        n_nodes_before = len(crb.all_nodes())

        X = np.ones((drb.response.value.shape[0], 1))
        crb.add_p_smooth(X, m=0.0, s=100.0, predictor="dependence")
        crb.add_predictor("dependence", "AlgebraicSigmoid")
        crb.add_copula("GaussianCopula")

        assert len(crb.all_nodes()) == n_nodes_before + 10


class TestTau2GibbsKernel:
    def test_return_gibbs_kernel(self, drb_np_smooth):
        """The function should successfully return a Gibbs kernel."""
        g = drb_np_smooth.groups[0]
        kernel = dr.tau2_gibbs_kernel(g)
        assert isinstance(kernel, gs.GibbsKernel)

    def test_plausible_transition(self, drb_np_smooth):
        """
        The transition outcome of a single Gibbs transition should be plausible, i.e.
        the drawn value should be within the support of the inverse gamma distribution.
        """
        g = drb_np_smooth.groups[0]
        kernel = dr.tau2_gibbs_kernel(g)

        def log_prob(model_state):
            return 0

        kernel.set_model(gs.DictModel(log_prob))
        model_state = construct_model_state_from_np_smooth(g)

        outcome = kernel.transition(
            prng_key, kernel_state={}, model_state=model_state, epoch=None
        )

        assert outcome.model_state[g["tau2"].name] > 0


class TestDistRegMCMC:
    def test_dist_reg_mcmc_p_smooth(self, drb):
        """
        When used with a DistRegBuilder that includes only parametric smooths,
        the function returns an engine builder with two IWLS kernels.
        """
        model = drb.build()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        assert len(ebuilder.kernels) == 2
        assert all([isinstance(k, gs.IWLSKernel) for k in ebuilder.kernels])

    def test_dist_reg_mcmc_build_p_smooth(self, drb):
        """
        When used with a DistRegBuilder that includes only parametric smooths,
        the returned engine builder successfully builds an engine.
        """
        model = drb.build()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        ebuilder.set_duration(1000, 100)
        engine = ebuilder.build()
        assert engine

        engine.sample_next_epoch()
        assert engine.get_results()

    def test_dist_reg_mcmc_np_smooth(self, drb_np_smooth):
        """
        When used with a DistRegBuilder that includes one parametric and one
        non-parametric smooth, the function returns an engine builder with two
        IWLS kernels and a Gibbs kernel.
        """
        drb = drb_np_smooth
        model = drb.build()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        assert len(ebuilder.kernels) == 3
        assert sum(isinstance(k, gs.IWLSKernel) for k in ebuilder.kernels) == 2
        assert sum(isinstance(k, gs.GibbsKernel) for k in ebuilder.kernels) == 1

    def test_dist_reg_mcmc_build_np_smooth(self, drb_np_smooth):
        """
        When used with a DistRegBuilder that includes one parametric and one
        non-parametric smooth, the engine builder successfully builds an engine.
        """
        drb = drb_np_smooth
        model = drb.build()
        ebuilder = dr.dist_reg_mcmc(model, 42, 2)
        ebuilder.set_duration(1000, 100)
        engine = ebuilder.build()
        assert engine
