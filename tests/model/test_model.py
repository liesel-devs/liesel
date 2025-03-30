import tempfile
import typing
import warnings
from collections.abc import Generator
from itertools import combinations
from types import MappingProxyType

import jax.numpy as jnp
import jax.random as rnd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
from liesel.model.model import GraphBuilder, Model, save_model
from liesel.model.nodes import Calc, Dist, Group, TransientNode, Value, Var


@pytest.fixture
def data() -> Generator:
    key = rnd.PRNGKey(13)
    n = 500
    true_beta = jnp.array([1.0, 2.0])
    true_sigma = 1.0

    key_x, key_y = rnd.split(key, 2)

    x0 = tfd.Uniform().sample(seed=key_x, sample_shape=n)
    X = jnp.column_stack([jnp.ones(n), x0])
    y = tfd.Normal(loc=X @ true_beta, scale=true_sigma).sample(seed=key_y)
    yield X, y


@pytest.fixture
def beta() -> Generator:
    beta_prior_loc = Var(0.0, name="beta_loc")
    beta_prior_scale = Var(100.0, name="beta_scale")
    beta_prior = Dist(tfd.Normal, loc=beta_prior_loc, scale=beta_prior_scale)

    beta_hat = Var.new_param(
        value=jnp.array([0.0, 0.0]), distribution=beta_prior, name="beta_hat"
    )
    yield beta_hat


@pytest.fixture
def sigma() -> Generator:
    sigma_prior_concentration = Var(0.01, name="concentration")
    sigma_prior_scale = Var(0.01, name="scale")

    sigma_prior = Dist(
        tfd.InverseGamma,
        concentration=sigma_prior_concentration,
        scale=sigma_prior_scale,
    )

    sigma_hat = Var.new_param(
        value=10.0,
        distribution=sigma_prior,
        name="sigma_hat",
    )

    yield sigma_hat


@pytest.fixture
def y_var(data, beta, sigma) -> Generator:
    x, y = data
    x_var = Var.new_obs(x, name="X")

    mu_calc = Calc(lambda X, beta: X @ beta, x_var, beta)
    mu_hat = Var(mu_calc, name="mu")

    likelihood = Dist(tfd.Normal, loc=mu_hat, scale=sigma)
    y_var = Var.new_obs(value=y, distribution=likelihood, name="y_var")

    Group("loc", X=x_var, beta=beta)
    Group("scale", scale=sigma)

    yield y_var


@pytest.fixture
def model(y_var) -> Generator:
    """
    A simple linear model with an additional unconnected data node for testing purposes.
    """
    data = Value(2.5, "z")
    yield Model([y_var, data])


@pytest.fixture
def model_nodes() -> Generator:
    log_prior = Value(3.0, _name="_model_log_prior")
    log_lik = Value(4.0, _name="_model_log_lik")
    log_prob = Value(5.0, _name="_model_log_prob")
    model_nodes = [log_prob, log_prior, log_lik]
    yield model_nodes


class TestModel:
    def test_copy_length(self, model: Model) -> None:
        """
        Verifies the correct length of copied var and node dicts.

        The output of ``model.copy_nodes()`` has 3 elements less than ``model.nodes``.
        This is, because ``model.nodes`` includes model-specific nodes for the log-
        likelihood, -prior, and -prob.
        """
        nodes_and_vars = model.copy_nodes_and_vars()

        assert len(nodes_and_vars) == 2
        assert len(nodes_and_vars[0]) == len(model.nodes) - 3
        assert len(nodes_and_vars[1]) == len(model.vars)

    def test_copy_unfrozen(self, model: Model) -> None:
        """Verifies that the vars and nodes are unfrozen."""
        nodes_and_vars = model.copy_nodes_and_vars()

        assert all([not node.model for node in nodes_and_vars[0].values()])
        assert all([not var.model for var in nodes_and_vars[1].values()])

    def test_copy_computational_model(self, model: Model) -> None:
        model._copy_computational_model()
        # TODO: Fill this test. At the moment, I am uncertain as to what this method
        # is supposed to achieve exactly.

    def test_groups(self, model: Model) -> None:
        """
        Verifies that the groups defined for the testing model are returned correctly.
        """
        groups = model.groups()

        assert len(groups["loc"].nodes_and_vars) == 2
        assert len(groups["scale"].nodes_and_vars) == 1

        assert "X" in groups["loc"]
        assert "beta" in groups["loc"]
        assert "scale" in groups["scale"]

    def test_set_seed_value_errors(self, model: Model) -> None:
        """
        Mypy would complain about the input types here, as it should.
        """

        with pytest.raises(TypeError):
            model.set_seed(jnp.array([0, 123]))  # type: ignore

        with pytest.raises(TypeError):
            model.set_seed(jnp.array([123]))  # type: ignore

        with pytest.raises(TypeError):
            model.set_seed(123)  # type: ignore

    def test_set_seed(self, model: Model) -> None:
        """Verifies that seed nodes are created by set_seed."""
        _, vars = model.copy_nodes_and_vars()
        vars["beta_loc"].value_node.needs_seed = True

        model = Model(vars.values())
        key = rnd.PRNGKey(123)
        model.set_seed(key)

        assert model._seed_nodes

    def test_set_seed_with_no_stochastic_nodes(self, model: Model) -> None:
        key = rnd.PRNGKey(123)
        model.set_seed(key)
        assert not model._seed_nodes

    def test_update_changes_log_prob(self, model: Model) -> None:
        model.auto_update = False
        log_prob_before = model.log_prob
        beta = model.vars["beta_hat"]
        beta.value = jnp.array([-10.0, 10.0])
        assert any([node.outdated for node in model.nodes.values()])

        model.update()
        assert not any([node.outdated for node in model.nodes.values()])
        assert log_prob_before != pytest.approx(model.log_prob)

    def test_update_uses_correct_order(self, model: Model) -> None:
        """
        Verifies that Model.update() updates the model in the correct topological order.
        """
        ...

    def test_log_probs(self, model: Model) -> None:
        assert model.log_prior.shape == ()
        assert model.log_lik.shape == ()
        assert model.log_prob.shape == ()

        assert model.log_prob == pytest.approx(model.log_lik + model.log_prior)

    def test_var_graph(self, model: Model) -> None:
        """
        Verifies that all model vars are present in the graph and vice versa.
        """

        vars = list(model.vars.values())
        assert all([node in vars for node in model.var_graph.nodes])
        assert all([node in model.var_graph.nodes for node in vars])

    def test_node_graph(self, model: Model) -> None:
        """
        Verifies that all model nodes are present in the graph and vice versa.
        """

        nodes = list(model.nodes.values())
        assert all([node in nodes for node in model.node_graph.nodes])
        assert all([node in model.node_graph.nodes for node in nodes])

    def test_nodes_len(self, model: Model) -> None:
        assert isinstance(model.nodes, MappingProxyType)
        print(list(model.vars.keys()))
        print(list(model.nodes.keys()))
        # this is a bit surprising since variables have always 2 value and 1
        # data node
        assert len(model.nodes) == 25

    def test_vars_len(self, model: Model) -> None:
        assert isinstance(model.vars, MappingProxyType)
        assert len(model.vars) == 9

    def test_vars(self, model: Model) -> None:
        """
        Verifies that all necessary vars are present in Model.vars.
        """
        assert "beta_loc" in model.vars
        assert "beta_scale" in model.vars
        assert "beta_hat" in model.vars

        assert "concentration" in model.vars
        assert "scale" in model.vars
        assert "sigma_hat" in model.vars

        assert "X" in model.vars
        assert "mu" in model.vars
        assert "y_var" in model.vars

    def test_nodes(self, model: Model) -> None:
        """
        Verifies that all necessary nodes are present in Model.nodes.
        """
        assert "z" in model.nodes

        assert "beta_loc_value" in model.nodes
        assert "beta_scale_value" in model.nodes
        assert "beta_hat_value" in model.nodes
        assert "beta_hat_log_prob" in model.nodes

        assert "concentration_value" in model.nodes
        assert "scale_value" in model.nodes
        assert "sigma_hat_value" in model.nodes
        assert "sigma_hat_log_prob" in model.nodes

        assert "X_value" in model.nodes
        assert "mu_value" in model.nodes

        assert "y_var_value" in model.nodes
        assert "y_var_log_prob" in model.nodes

        assert "_model_log_lik" in model.nodes
        assert "_model_log_prior" in model.nodes
        assert "_model_log_prob" in model.nodes

    def test_state(self, model: Model) -> None:
        """
        Verifies that the initial state of the model agrees with the initial values
        of the model nodes.
        """
        for name, node in model.nodes.items():
            if not isinstance(node, TransientNode):
                assert jnp.allclose(model.state[name].value, node.value)
                assert not model.state[name].outdated

    def test_move_vars_and_nodes(self, model: Model) -> None:
        n_vars = len(model.vars)
        n_nodes = len(model.nodes)

        # no model nodes
        nodes, vars = model.pop_nodes_and_vars()
        assert len(vars) == n_vars
        assert len(nodes) == n_nodes - 3
        assert not model.vars
        assert not model.nodes

    def test_unique_groups(self) -> None:
        v1 = Var(0.0, name="v1")
        v2 = Var(0.0, name="v2")
        g1 = Group("g1", var1=v1, var2=v2)
        m = Model([v1, v2])

        assert g1 in m.groups().values()

        m.pop_nodes_and_vars()
        g2 = Group("g2", var1=v1, var2=v2)
        m = Model([v1, v2])

        assert g1 in m.groups().values()
        assert g2 in m.groups().values()

        m.pop_nodes_and_vars()
        v3 = Var(0.0, name="v3")
        Group("g1", var3=v3)

        with pytest.raises(RuntimeError):
            Model([v1, v2, v3])

    def test_transform_vars(self) -> None:
        lmbd = Var(1.0, name="lambda")
        dist = Dist(tfd.Exponential, lmbd)
        x = Var(1.0, dist, name="x")
        x.auto_transform = True

        model = Model([x])

        assert "x_transformed" in model.vars
        assert "lambda_transformed" not in model.vars
        assert model.vars["x_transformed"].value == pytest.approx(0.54132485)

    def test_transform_vars_weak(self) -> None:
        x = Var.new_calc(lambda x: x, name="x")
        x.auto_transform = True

        with pytest.raises(RuntimeError, match="has no distribution"):
            Model([x])

    def test_transform_vars_no_dist(self) -> None:
        x = Var(1.0, name="x")
        x.auto_transform = True

        with pytest.raises(RuntimeError, match="has no distribution"):
            Model([x])

    def test_build_after_transform(self) -> None:
        lmbd = Var(1.0, name="lambda")
        dist = Dist(tfd.Exponential, lmbd)
        x = Var(1.0, dist, name="x")

        gb = GraphBuilder()

        model = gb.add(x).build_model()

        _, vars = model.copy_nodes_and_vars()

        gb.add(*vars.values())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (FutureWarning))
            gb.transform(vars["x"])
        new_model = gb.build_model()

        assert "x_transformed" in new_model.vars
        assert new_model.vars["x_transformed"].value == pytest.approx(0.54132485)

    def test_mcmc_kernels(self):
        mu = Var(
            value=0.0,
            distribution=Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="mu",
            mcmc_kernel=gs.NUTSKernel,
        )

        model = Model([mu])

        kernels = model.mcmc_kernels()

        assert len(kernels) == 1
        assert isinstance(kernels["mu"], gs.NUTSKernel)

        mu = Var(
            value=0.0,
            distribution=Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="mu",
            mcmc_kernel=gs.NUTSKernel(["mu"]),
        )

        model = Model([mu])

        kernels = model.mcmc_kernels()

        assert len(kernels) == 1
        assert isinstance(kernels["mu"], gs.NUTSKernel)


@pytest.mark.xfail
class TestUserDefinedModelNodes:
    @typing.no_type_check
    def test_require_model_nodes(self, y_var: Var, model_nodes: list[Value]) -> None:
        """
        Verifies that the model raises exceptions if any required model nodes are
        missing.
        """
        # Initialization without model nodes fails
        with pytest.raises(RuntimeError):
            Model(variables=[y_var], user_defined_model_nodes=True)

        # Initialization with any single model node fails
        for node in model_nodes:
            with pytest.raises(RuntimeError):
                Model(variables=[y_var], nodes=[node], user_defined_model_nodes=True)

        # Initialization with any two model nodes fails
        for comb in combinations(model_nodes, 2):
            with pytest.raises(RuntimeError):
                Model(variables=[y_var], nodes=comb, user_defined_model_nodes=True)

    @typing.no_type_check
    def test_provide_model_nodes(self, y_var: Var, model_nodes: list[Value]) -> None:
        """
        Verifies that the model nodes are accepted if provided correctly.
        """
        model = Model(
            variables=[y_var], nodes=model_nodes, user_defined_model_nodes=True
        )

        assert model.log_prior == pytest.approx(3.0)
        assert model.log_lik == pytest.approx(4.0)
        assert model.log_prob == pytest.approx(5.0)

    @typing.no_type_check
    def test_user_defined_false(self, y_var: Var, model_nodes: list[Value]) -> None:
        """
        Verifies that the model does not accept nodes with reserved names, if the
        flag user_defnied_model_nodes is set to False (default).
        """
        with pytest.raises(RuntimeError):
            Model(variables=[y_var], nodes=model_nodes)

        for node in model_nodes:
            with pytest.raises(RuntimeError):
                Model(variables=[y_var], nodes=[node])

        for comb in combinations(model_nodes, 2):
            with pytest.raises(RuntimeError):
                Model(variables=[y_var], nodes=comb)


class TestSimulate:
    @pytest.fixture
    def model(self) -> Model:
        mu = Var(0.0, name="mu")
        sigma = Var(1.0, name="sigma")
        x = Var(0.0, Dist(tfd.Normal, mu, sigma), name="x")
        return Model([x])

    def test_simulate_scalar(self, model):
        model.simulate(rnd.PRNGKey(42))
        assert jnp.all(model.vars["x"].value != 0.0)
        assert model.vars["x"].value.shape == ()

    def test_simulate_vector(self, model):
        model.vars["x"].value = jnp.zeros(5)

        model.simulate(rnd.PRNGKey(42))
        assert jnp.all(model.vars["x"].value != 0.0)
        assert model.vars["x"].value.shape == (5,)

    def test_simulate_matrix(self, model):
        model.vars["x"].value = jnp.zeros((5, 5))

        model.simulate(rnd.PRNGKey(42))
        assert jnp.all(model.vars["x"].value != 0.0)
        assert model.vars["x"].value.shape == (5, 5)

    def test_simulate_with_skipped_dist(self, model):
        model.simulate(rnd.PRNGKey(42), ["x_log_prob"])
        assert model.vars["x"].value == 0.0

    def test_simulate_with_skipped_at(self, model):
        model.simulate(rnd.PRNGKey(42), ["x_var_value"])
        assert model.vars["x"].value == 0.0

    def test_simulate_with_skipped_var(self, model):
        model.simulate(rnd.PRNGKey(42), ["x"])
        assert model.vars["x"].value == 0.0

    def test_simulate_with_empty_model(self):
        Model([]).simulate(rnd.PRNGKey(42))

    def test_simulate_with_unsettable_value(self):
        mu = Var(0.0, name="mu")
        sigma = Var(1.0, name="sigma")
        x = Var(Calc(lambda: 0.0), Dist(tfd.Normal, mu, sigma), name="x")
        model = Model([x])

        with pytest.raises(AttributeError, match="Cannot set value of Calc"):
            model.simulate(rnd.PRNGKey(42))

    def test_simulate_hierarchical_model(self):
        mu = Var(0.0, Dist(tfd.Normal, loc=2.0, scale=1.0), name="mu")
        sigma = Var(1.0, name="sigma")
        x = Var(0.0, Dist(tfd.Normal, mu, sigma), name="x")
        model = Model([x])

        model.simulate(rnd.PRNGKey(42))
        assert jnp.all(model.vars["mu"].value != 0.0)
        assert jnp.all(model.vars["x"].value != 0.0)


def test_save_model() -> None:
    x = Var(1.0, name="x")
    model = Model([x])

    fh = tempfile.TemporaryFile()
    save_model(model, fh)
    fh.close()


class TestJittering:
    def test_jitter(self):
        mu = Var(1.0, name="mu", jitter_dist=tfd.Uniform(low=-1.0, high=1.0))
        jittered_value = mu.apply_jitter(seed=rnd.key(1))

        assert mu.value != pytest.approx(jittered_value)

    def test_model_jitter(self):
        mu = Var(
            jnp.array([1.0, 2.0]),
            name="mu",
            jitter_dist=tfd.Uniform(low=-1.0, high=1.0),
        )

        sigma = Var(1.0, name="sigma", jitter_dist=tfd.Uniform(low=-0.5, high=1.0))

        y = Var(jnp.array([2.0, 3.0]), Dist(tfd.Normal, loc=mu, scale=sigma), name="y")

        model = Model([y])
        model.apply_jitter(rnd.key(1))

        assert not jnp.allclose(mu.value, jnp.array([1.0, 2.0]))
        assert not jnp.allclose(sigma.value, 1.0)

        assert jnp.allclose(y.value, jnp.array([2.0, 3.0]))

    def test_engine_jitter(self):
        mu = Var(
            jnp.array([1.0, 2.0]),
            name="mu",
            jitter_dist=tfd.Uniform(low=-1.0, high=1.0),
        )

        sigma = Var(1.0, name="sigma", jitter_dist=tfd.Uniform(low=-0.5, high=1.0))

        y = Var(jnp.array([2.0, 3.0]), Dist(tfd.Normal, loc=mu, scale=sigma), name="y")

        model = Model([y])

        eb = gs.EngineBuilder(seed=2, num_chains=2)
        eb.set_model(gs.LieselInterface(model))
        eb.set_initial_values(model.state)
        eb.set_duration(warmup_duration=200, posterior_duration=20)
        eb.set_jitter_fns(model.jitter_functions())

        eb.add_kernel(gs.NUTSKernel(["mu"]))
        eb.add_kernel(gs.NUTSKernel(["sigma"]))

        engine = eb.build()

        assert not jnp.allclose(mu.value, engine._model_states["mu_value"][0][0])
        assert not jnp.allclose(mu.value, engine._model_states["mu_value"][0][1])
        assert not jnp.allclose(
            engine._model_states["mu_value"][0][0],
            engine._model_states["mu_value"][0][1],
        )

        assert not jnp.allclose(sigma.value, engine._model_states["sigma_value"][0][0])
        assert not jnp.allclose(sigma.value, engine._model_states["sigma_value"][0][1])
        assert not jnp.allclose(
            engine._model_states["sigma_value"][0][0],
            engine._model_states["sigma_value"][0][1],
        )
