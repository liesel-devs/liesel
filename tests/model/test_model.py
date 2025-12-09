import tempfile
import typing
from collections.abc import Generator
from itertools import combinations
from types import MappingProxyType

import jax
import jax.numpy as jnp
import jax.random as rnd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

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

    @pytest.mark.xfail
    def test_copy_model(self, model: Model) -> None:
        vars_ = list(model.vars.values())
        model_copy = Model(vars_, copy=True)
        assert len(model_copy.vars) == len(model.vars)

        with pytest.raises(RuntimeError):
            Model(vars_, copy=False)

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

    def test_parameters(self, model: Model) -> None:
        assert len(model.parameters) == 2
        assert "sigma_hat" in model.parameters
        assert "beta_hat" in model.parameters

        parameters_log_prob = [var.log_prob.sum() for var in model.parameters.values()]
        assert sum(parameters_log_prob) == pytest.approx(model.log_prior)

    def test_observed(self, model: Model) -> None:
        assert len(model.observed) == 2
        assert "y_var" in model.observed
        assert "X" in model.observed

        observed_log_prob = [
            jnp.atleast_1d(var.log_prob).sum() for var in model.observed.values()
        ]
        assert sum(observed_log_prob) == pytest.approx(model.log_lik)

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
        vars["x"].transform()

        new_model = gb.build_model()

        assert "x_transformed" in new_model.vars
        assert new_model.vars["x_transformed"].value == pytest.approx(0.54132485)

    def test_extract_position(self, model) -> None:
        pos = model.extract_position(["z"])
        assert pos["z"] == pytest.approx(model.nodes["z"].value)

    def test_update_state(self, model) -> None:
        pos = {"z": 3.0}
        state = model.update_state(pos, inplace=False)
        assert model.extract_position(["z"], model_state=state)["z"] == pytest.approx(
            3.0
        )
        assert model.nodes["z"].value != pytest.approx(3.0)

        # updating the state from above
        pos = {"sigma_hat": 20.0}
        state = model.update_state(pos, inplace=False, model_state=state)

        # extracted position from updated state should contain the updated values
        extracted_pos = model.extract_position(["z", "sigma_hat"], model_state=state)
        assert extracted_pos["z"] == pytest.approx(3.0)
        assert extracted_pos["sigma_hat"] == pytest.approx(20.0)

        # original model state should be unchanged
        assert model.nodes["z"] != pytest.approx(3.0)
        assert model.vars["sigma_hat"].value != pytest.approx(20.0)

        pos = {"z": 3.0}
        state = model.update_state(pos, inplace=True)
        assert model.extract_position(["z"], model_state=state)["z"] == pytest.approx(
            3.0
        )
        assert model.nodes["z"].value == pytest.approx(3.0)


class TestPredictions:
    def test_predict_no_batching_dim(self, model) -> None:
        position = model.extract_position(["sigma_hat", "beta_hat"])

        # predictions at current values for all vars
        pred = model.predict(samples=position)
        assert pred["mu"].shape == (500,)
        assert len(pred) == len(model.vars)

    def test_predict_one_batching_dim(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((3, 2), rnd.PRNGKey(6)),
        }

        # manual prediction
        manual_pred = jnp.einsum(
            "nk,...k->...n", model.vars["X"].value, samples["beta_hat"]
        )

        pred = model.predict(samples=samples)
        assert jnp.allclose(pred["mu"], manual_pred)
        assert pred["mu"].shape == (3, 500)
        assert len(pred) == len(model.vars)

    def test_predict_at_current_state(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((4, 3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
        }

        # manual prediction
        manual_pred = jnp.einsum(
            "nk,...k->...n", model.vars["X"].value, samples["beta_hat"]
        )

        # predictions at current values for all vars
        pred = model.predict(samples=samples)
        assert jnp.allclose(pred["mu"], manual_pred)
        assert pred["mu"].shape == (4, 3, 500)
        assert len(pred) == len(model.vars)

    def test_predict_with_unused_samples(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((4, 3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
            "unused": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
        }

        # manual prediction
        manual_pred = jnp.einsum(
            "nk,...k->...n", model.vars["X"].value, samples["beta_hat"]
        )

        # predictions at current values for all vars
        pred = model.predict(samples=samples)
        assert jnp.allclose(pred["mu"], manual_pred)
        assert pred["mu"].shape == (4, 3, 500)
        assert len(pred) == len(model.vars)

    def test_predict_for_specific_var(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((4, 3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
        }

        # manual prediction
        manual_pred = jnp.einsum(
            "nk,...k->...n", model.vars["X"].value, samples["beta_hat"]
        )

        # predictions at current values for mu
        pred = model.predict(samples=samples, predict=["mu"])

        assert jnp.allclose(pred["mu"], manual_pred)
        assert pred["mu"].shape == (4, 3, 500)
        assert len(pred) == 1

    def test_predict_at_newdata(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((4, 3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
        }

        # predictions at new values for X
        xnew = tfd.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=model.vars["X"].value.shape, seed=rnd.PRNGKey(7)
        )

        assert not jnp.allclose(xnew, model.vars["X"].value)

        manual_pred = jnp.einsum("nk,...k->...n", xnew, samples["beta_hat"])

        pred = model.predict(samples=samples, predict=["mu"], newdata={"X": xnew})
        assert jnp.allclose(pred["mu"], manual_pred)
        assert pred["mu"].shape == (4, 3, 500)

    def test_predict_when_newdata_and_samples_overlap(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((4, 3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
        }

        # predictions at new values for X
        xnew = tfd.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=model.vars["X"].value.shape, seed=rnd.PRNGKey(7)
        )

        assert not jnp.allclose(xnew, model.vars["X"].value)

        with pytest.raises(RuntimeError):
            model.predict(
                samples=samples,
                predict=["mu"],
                newdata={"X": xnew, "beta_hat": samples["beta_hat"][0, 0, :]},
            )

    def test_predict_at_newdata_not_in_the_model(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((4, 3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
        }

        # predictions at new values for X
        xnew = tfd.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=model.vars["X"].value.shape, seed=rnd.PRNGKey(7)
        )

        with pytest.raises(KeyError):
            model.predict(samples=samples, predict=["mu"], newdata={"Z": xnew})

    def test_predict_at_newdata_not_needed(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((4, 3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
        }

        # predictions at new values for X
        xnew = tfd.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=model.vars["X"].value.shape, seed=rnd.PRNGKey(7)
        )

        model.predict(samples=samples, predict=["sigma_hat"], newdata={"X": xnew})

    def test_predict_at_newdata_with_new_shape(self, model) -> None:
        samples = {
            "sigma_hat": tfd.Uniform().sample((4, 3), rnd.PRNGKey(6)),
            "beta_hat": tfd.Uniform().sample((4, 3, 2), rnd.PRNGKey(6)),
        }

        # predictions at new values for X with different N
        xnew = tfd.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=(10, 2), seed=rnd.PRNGKey(7)
        )

        manual_pred = jnp.einsum("nk,...k->...n", xnew, samples["beta_hat"])

        pred = model.predict(samples=samples, predict=["mu"], newdata={"X": xnew})
        assert jnp.allclose(pred["mu"], manual_pred)
        assert pred["mu"].shape == (4, 3, 10)

        # if the newdata shape does not work with some required shapes downstream,
        # we run into a typerror
        with pytest.raises(TypeError):
            model.predict(samples=samples, newdata={"X": xnew})

    def test_predict_multiple_vars_new_shapes_issue_291(self) -> None:
        # create model with variables of shape (3,)
        x1 = Var.new_obs(jnp.ones(3), name="x1")
        x2 = Var.new_obs(jnp.ones(3), name="x2")

        # needed for provide samples
        dummy_param = Var.new_param(jnp.array(0.0), name="dummy")

        # create a calculation that depends on compatible shapes
        calc_sum = Var.new_calc(
            lambda y0, y1, y2: (y0 + y1).sum() + y2,
            x1,
            x2,
            dummy_param,
            name="calc_sum",
        )
        model = GraphBuilder().add(x1, x2, calc_sum, dummy_param).build_model()

        # update with variables of different but compatible shapes
        pred = model.predict(
            samples={"dummy": jnp.array([[0.0], [0.0]])},
            predict=["calc_sum"],
            newdata={"x1": jnp.ones(5), "x2": jnp.ones(5)},
        )

        # verify prediction works - calc_sum should be 10.0 (5 + 5 + 0 = 10)
        assert pred["calc_sum"].shape == (2, 1)
        assert jnp.allclose(pred["calc_sum"], 10.0)


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


@pytest.fixture
def linreg():
    X = Var(
        value=tfd.Uniform(low=-1.0, high=1.0).sample((100, 2), rnd.key(3)), name="X"
    )
    b = Var(
        value=jnp.zeros(2),
        distribution=Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="b",
    )
    mu = Var.new_calc(jnp.dot, X, b, name="mu")

    sigma = Var(1.0, Dist(tfd.InverseGamma, concentration=5.0, scale=0.5), name="sigma")
    y = Var(jnp.zeros(X.value.shape[0]), Dist(tfd.Normal, mu, sigma), name="y")
    model = Model([y])
    yield model


class TestSample:
    def test_sample_prior(self):
        """
        Test that the function runs and shows expected behavior in a minimal example.
        """
        mu = Var(0.0, Dist(tfd.Normal, loc=2.0, scale=1.0), name="mu")
        sigma = Var(
            1.0, Dist(tfd.InverseGamma, concentration=5.0, scale=0.5), name="sigma"
        )
        x = Var(0.0, Dist(tfd.Normal, mu, sigma), name="x")
        model = Model([x])

        # sample with x being fixed
        # so x will not be sampled
        samples = model.sample(shape=(1, 100), seed=rnd.key(1), fixed=["x"])

        assert "mu" in samples  # mu should be sampled
        assert "sigma" in samples  # sigma should be sampled
        assert "x" not in samples  # x should not be sampled

        assert samples["mu"].shape == (1, 100)  # verify correct shape

        # basic plausibility checks for sampling from the correct distribution
        # this is not a tough check though.
        assert samples["mu"].mean() == pytest.approx(2.0, abs=0.3)
        assert samples["mu"].std() == pytest.approx(1.0, abs=0.5)

        assert samples["sigma"].shape == (1, 100)  # verify correct shape of sigma

        # basic plausibility checks for sampling from the correct distribution
        # this is not a tough check though.
        sigma_mean = sigma.dist_node.init_dist().mean()
        sigma_std = sigma.dist_node.init_dist().stddev()
        assert samples["sigma"].mean() == pytest.approx(sigma_mean, abs=0.1)
        assert samples["sigma"].std() == pytest.approx(sigma_std, abs=0.1)

        # now sample all variables, including x
        samples = model.sample(shape=(1, 100), seed=rnd.key(1))

        assert "x" in samples  # verify that x is in samples
        assert samples["x"].shape == (1, 100)  # verify shape

    def test_sample_prior_linreg(self, linreg: Model):
        """
        Test that the function runs and shows expected behavior in a slightly more
        elaborate example (linear regression).
        """
        model = linreg

        # sample with y fixed; i.e. y will not be sampled
        samples = model.sample(shape=(1, 100), seed=rnd.key(1), fixed=["y"])

        assert "b" in samples  # verify that b has been sampled
        assert "sigma" in samples  # verify that sigma has been sampled
        assert "y" not in samples  # verify that y has NOT ben sampled

        assert samples["b"].shape == (1, 100, 2)  # verify shape of b samples

        # basic plausibility checks for sampling from the correct distribution
        # this is not a tough check though.
        assert samples["b"].mean(axis=(0, 1)) == pytest.approx(0.0, abs=0.5)
        assert samples["b"].std(axis=(0, 1)) == pytest.approx(1.0, abs=0.5)

        assert samples["sigma"].shape == (1, 100)  # verify shape of sigma samples
        # basic plausibility checks for sampling from the correct distribution
        # this is not a tough check though.
        sigma = model.vars["sigma"]
        sigma_mean = sigma.dist_node.init_dist().mean()  # type: ignore
        sigma_std = sigma.dist_node.init_dist().stddev()  # type: ignore
        assert samples["sigma"].mean() == pytest.approx(sigma_mean, abs=0.1)
        assert samples["sigma"].std() == pytest.approx(sigma_std, abs=0.1)

        # now sample all nodes, including y
        samples = model.sample(shape=(1, 80), seed=rnd.key(1))

        assert "y" in samples  # verify that y has been sampled
        # verify shape of y samples
        # because we have (1, 80) samples of regression coefficients and
        # (100,) covariate observations, the expected shape for y is (1, 80, 100),
        # where the shape is organized as (sample_shape, event_shape)
        assert samples["y"].shape == (1, 80, 100)

        # basic plausibility checks for sampling from the correct distribution
        # this is not a tough check though.
        y_samples_mean = samples["y"].mean(axis=(0, 1))
        assert jnp.allclose(y_samples_mean, 0.0, atol=0.5)

    def test_sample_prior_linreg_jit(self, linreg: Model):
        model = linreg

        jitted_sample = jax.jit(
            model.sample,
            static_argnames=["shape", "fixed", "dists"],
        )

        jitted_sample(shape=(1, 100), seed=rnd.key(1))

        x_shape = model.vars["X"].value.shape
        x_new = tfd.Uniform(low=10.0, high=11.0).sample(x_shape, seed=rnd.key(9))
        jitted_sample(shape=(1, 100), seed=rnd.key(1), newdata={"X": x_new})
        jitted_sample(
            shape=(1, 100), seed=rnd.key(1), newdata={"X": x_new}, fixed=("y")
        )

    def test_sample_from_custom_dist(self, linreg: Model):
        model = linreg

        # sample with y fixed; i.e. y will not be sampled
        samples = model.sample(shape=(1, 100), seed=rnd.key(1), fixed=["y"])

        assert "b" in samples  # verify that b has been sampled
        assert "sigma" in samples  # verify that sigma has been sampled
        assert "y" not in samples  # verify that y has NOT ben sampled

        samples2 = model.sample(
            shape=(1, 100),
            seed=rnd.key(1),
            fixed=["y"],
            dists={"b": Dist(tfd.Uniform, low=0.1, high=0.2)},
        )

        assert "b" in samples2
        assert not jnp.allclose(samples["b"], samples2["b"])
        assert jnp.all(samples2["b"] <= 0.2)
        assert jnp.all(samples2["b"] >= 0.1)

    def test_sample_from_custom_dist_with_variable_dependent_param(self):
        min_ = Var.new_param(0.1, Dist(tfd.Uniform, low=0.1, high=0.2), name="min")
        max_ = Var.new_calc(lambda x: x + 0.1, x=min_, name="max")

        m = Var.new_param(0.0, name="m")
        y = Var.new_obs(0.0, Dist(tfd.Normal, loc=m, scale=1.0), name="y")

        model = Model([y, min_, max_])

        samples1 = model.sample(shape=(1, 100), seed=rnd.key(1))
        assert "m" not in samples1

        samples2 = model.sample(
            shape=(1, 100),
            seed=rnd.key(1),
            dists={"m": Dist(tfd.Uniform, low=min_, high=max_)},
        )

        assert "m" in samples2

        max_samples = max_.predict(samples2)

        assert jnp.all(samples2["m"] > samples2["min"])
        assert jnp.all(samples2["m"] < max_samples)

    def test_sample_from_custom_dist_var_not_found(self):
        min_ = Var.new_param(0.1, Dist(tfd.Uniform, low=0.1, high=0.2), name="min")
        max_ = Var.new_calc(lambda x: x + 0.1, x=min_, name="max")

        m = Var.new_param(0.0, name="m")
        y = Var.new_obs(0.0, Dist(tfd.Normal, loc=m, scale=1.0), name="y")

        model = Model([y, min_, max_])

        with pytest.raises(ValueError):
            model.sample(
                shape=(1, 100),
                seed=rnd.key(1),
                dists={"s": Dist(tfd.Uniform, low=min_, high=max_)},
            )

    def test_sample_from_custom_dist_weak_var(self):
        min_ = Var.new_param(0.1, Dist(tfd.Uniform, low=0.1, high=0.2), name="min")
        max_ = Var.new_calc(lambda x: x + 0.1, x=min_, name="max")

        m = Var.new_param(0.0, name="m")
        y = Var.new_obs(0.0, Dist(tfd.Normal, loc=m, scale=1.0), name="y")

        model = Model([y, min_, max_])

        with pytest.raises(ValueError):
            model.sample(
                shape=(1, 100),
                seed=rnd.key(1),
                dists={"max": Dist(tfd.Uniform, low=min_, high=max_)},
            )

    def test_sample_from_custom_dist_with_pseudo_circular_graph(self):
        """
        This is an interesting edge case.
        In the model below, we have

            m ~ U(min, max)

        and when sampling, I change the distribution for min:

            min ~ U(0.1, 0.2)  ->  min ~ U(0.48, m)

        This is circular. During sampling however, the circularity does not cause an
        error. Instead, the value of m in the current state of the model gets inserted
        in the distribution for min, such that we have

            min ~ U(0.48, 0.5).

        """
        min_ = Var.new_param(0.1, Dist(tfd.Uniform, low=0.1, high=0.2), name="min")
        max_ = Var.new_calc(lambda x: x + 0.1, x=min_, name="max")

        m = Var.new_param(0.5, Dist(tfd.Uniform, low=min_, high=max_), name="m")
        y = Var.new_obs(0.0, Dist(tfd.Normal, loc=m, scale=1.0), name="y")

        model = Model([y, min_, max_])

        samples = model.sample(shape=(1, 100), seed=rnd.key(1))

        max_samples = max_.predict(samples)

        assert jnp.all(samples["m"] > samples["min"])
        assert jnp.all(samples["m"] < max_samples)

        samples2 = model.sample(
            shape=(1, 100),
            seed=rnd.key(1),
            dists={"min": Dist(tfd.Uniform, low=0.48, high=m)},
        )

        assert "m" in samples2
        assert jnp.all(samples2["min"] < 0.5)
        assert jnp.all(samples2["min"] > 0.48)

        max_samples = max_.predict(samples2)

        assert jnp.all(samples2["m"] > samples2["min"])
        assert jnp.all(samples2["m"] < max_samples)

    def test_sample_posterior(self, linreg: Model):
        model = linreg

        samples1 = model.sample(shape=(2, 8), seed=rnd.key(7), fixed=["y"])
        samples2 = model.sample(
            shape=(11,), seed=rnd.key(8), posterior_samples=samples1
        )

        assert "y" not in samples1  # verify that y was not sampled in samples1

        # in samples2, the variables for which we provided samples are not sampled again
        # this leaves only 'y' as a variable with a probability distribution to be
        # sampled. So len(samples2) should be 1.
        assert len(samples2) == 1
        assert "y" in samples2  # verify that the sampled variable is y

        # the shape of the new sample for y is now
        # (sample_shape, chain, iter, event_shape)
        # in this case: sample_shape = 11
        # with (chain, iter), I use MCMC terminology here. In this example, they refer
        # to the elements of the sample shape for sample1
        assert samples2["y"].shape == (11, 2, 8, 100)

    def test_sample_at_newdata(self, linreg: Model):
        model = linreg

        samples1 = model.sample(
            shape=(2, 8),
            seed=rnd.key(7),
        )

        x_shape = model.vars["X"].value.shape
        x_new = tfd.Uniform(low=10.0, high=11.0).sample(x_shape, seed=rnd.key(9))
        samples2 = model.sample(
            shape=(2, 8),
            seed=rnd.key(8),
            newdata={"X": x_new},
        )

        assert not jnp.allclose(samples1["y"], samples2["y"])

    def test_sample_posterior_keys_overlap_with_posterior_samples(self, linreg: Model):
        model = linreg

        samples1 = model.sample(
            shape=(2, 8),
            seed=rnd.key(7),
        )

        x_shape = model.vars["X"].value.shape
        x_new = tfd.Uniform(low=10.0, high=11.0).sample(x_shape, seed=rnd.key(9))
        with pytest.raises(RuntimeError):
            model.sample(
                shape=(2, 8),
                seed=rnd.key(8),
                posterior_samples=samples1,
                newdata={"X": x_new, "b": samples1["b"][0, 0, :]},
            )

    def test_sample_posterior_shape_of_posterior_samples(self, linreg: Model):
        model = linreg
        # the values in posterior_samples *have* to have leading (chain, iter) axes
        # if one of them is missing, the function errors
        samples1 = model.sample(shape=(2,), seed=rnd.key(7), fixed=["y"])
        with pytest.raises(ValueError):
            model.sample(shape=(11,), seed=rnd.key(8), posterior_samples=samples1)

        # *too many* leading axes also cause errors
        samples1 = model.sample(shape=(3, 2, 8), seed=rnd.key(7), fixed=["y"])
        with pytest.raises(RuntimeError):
            model.sample(shape=(11,), seed=rnd.key(8), posterior_samples=samples1)

    def test_sample_posterior_consistency_of_fixed(self, linreg: Model):
        model = linreg
        # If a variable name that is given in 'fixed' is also included in
        # 'posterior_samples', the function raises an error.
        samples1 = model.sample(shape=(2, 8), seed=rnd.key(7), fixed=["y"])
        with pytest.raises(ValueError):
            model.sample(
                shape=(11,), seed=rnd.key(8), posterior_samples=samples1, fixed=["b"]
            )
