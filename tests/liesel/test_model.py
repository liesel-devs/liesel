import copy

import jax
import numpy as np
import pytest
import scipy

from liesel.liesel import ModelBuilder, Node, NodeDistribution, NodeGroup, Parameter
from liesel.liesel.goose import (
    GooseModel,
    get_log_prob,
    get_position,
    make_log_prob_fn,
    make_update_state_fn,
    update_state,
)


@pytest.fixture
def sigma():
    yield Node(value=1.0, name="sigma")


@pytest.fixture
def loc():
    yield Node(value=0.0, name="mu")


@pytest.fixture
def node(sigma, loc):
    node = Parameter(
        value=np.asarray((0.0, 0.0, 0.0)),
        name="foo",
        distribution=NodeDistribution("Normal", loc=loc, scale=sigma),
    )
    yield node


@pytest.fixture
def model(sigma, loc, node):
    """Provides a model for testing."""
    mb = ModelBuilder()
    mb.add_nodes(node, loc, sigma)

    model = mb.build()
    yield model


class TestModel:
    def test_implicit_no_distribution(self):
        """
        Expected behavior: If a parameter without a distribution is added to a model,
        the model can still be built.

        In this case, the parameter is defined to have no distribution simply by
        *omitting* the distribution argument upon initialization.
        """
        node = Parameter(0.0, name="foo")
        mb = ModelBuilder()
        mb.add_nodes(node)
        model = mb.build()

        model_node = list(model.nodes.values())[0]
        assert model_node.value == 0.0
        assert model.log_prob == 0.0

    def test_explicit_no_distribution(self):
        """
        Expected behavior: If a parameter without a distribution is added to a model,
        the model can still be built.

        In this case, the parameter is defined to have no distribution explicitly.
        """
        node = Parameter(0.0, name="foo", distribution=None)
        mb = ModelBuilder()
        mb.add_nodes(node)
        model = mb.build()

        model_node = list(model.nodes.values())[0]
        assert model_node.value == 0.0
        assert model.log_prob == 0.0

    def test_build_model_without_node_name(self):
        """
        Expected behavior: If a node without a name is added to a model, the model can
        still be built.
        """
        node = Node(0.0)
        mb = ModelBuilder()
        mb.add_nodes(node)
        model = mb.build()

        model_node = list(model.nodes.values())[0]
        assert model_node.value == 0.0
        assert model_node == node

    def test_no_distribution(self):
        """
        Expected behavior: If a node without a distribution is added to a model, the
        model can still be built.

        In this case, the node is defined to have no distribution simply by *omitting*
        the distribution argument upon initialization.
        """
        node = Node(0.0, name="foo")
        mb = ModelBuilder()
        mb.add_nodes([node])
        model = mb.build()

        assert model.log_prob == 0.0

    def test_logprob(self, model):
        """
        The model's log_prob attribute should be consistent with the log probability
        returned by scipy's logpdf function.
        """
        expected = float(3 * scipy.stats.norm.logpdf(0.0, 0.0, 1.0))
        assert float(model.log_prob) == pytest.approx(expected)

    def test_jaxify(self, model):
        """
        The model's log probability should not be changed by jaxifying the model (i.e.
        all nodes in the model).
        """
        logp = float(model.log_prob)
        model.jaxify()
        j_logp = float(model.log_prob)

        assert float(j_logp) == pytest.approx(logp)

    def test_unjaxify(self, model):
        """
        The model's log probability should not be changed by jaxifying and subsequently
        unjaxifying the model (i.e. all nodes in the model).
        """
        logp = float(model.log_prob)
        model.jaxify()
        model.unjaxify()
        j_logp = float(model.log_prob)

        assert not model.jaxified
        assert float(j_logp) == pytest.approx(logp)

    def test_empty_copy(self, model):
        """
        Copying should return a version of the model with an empty state and leave the
        original object untouched.
        """
        model_before = copy.deepcopy(model)
        model_copy = model.empty_copy()

        assert model_copy.log_prob == 0.0
        assert not model.log_prob == 0.0
        assert model.log_prob == model_before.log_prob

    def test_get_nodes_by_class(self, model):
        """
        Nodes should be correctly retrieved based on their class. Specifying a parent
        class should lead to retrieving nodes of children classes.
        """
        params = model.get_nodes_by_class(Parameter)
        assert len(params) == 1
        assert params["foo"]

        nodes = model.get_nodes_by_class(Node)
        assert len(nodes) == 3

    def test_get_nodes_by_regex(self, model):
        """
        Nodes should be retrieved by matching regular expressions to their names.
        """
        params = model.get_nodes_by_regex("foo")
        assert len(params) == 1
        assert params["foo"]

        only_hyperparams = model.get_nodes_by_regex("(mu)|(sigma)")
        assert len(only_hyperparams) == 2
        assert "foo" not in only_hyperparams

    def test_update(self, model):
        """
        The model's update method should successfully control its nodes' update methods.
        """
        foo = model.get_nodes_by_regex("foo")["foo"]

        foo.set_value(np.array([10, 10, 10]), update=False)
        assert foo.outdated

        model.update()
        assert not foo.outdated


class TestModelBuilder:
    def test_add_nodes_individually(self, sigma, loc, node):
        mb = ModelBuilder()
        mb.add_nodes(node, loc, sigma)
        assert len(mb.nodes) == 3
        assert len(mb.all_nodes()) == 3

    def test_add_nodes_iterable(self, sigma, loc, node):
        mb = ModelBuilder()
        mb.add_nodes([node, loc, sigma])
        assert len(mb.nodes) == 3
        assert len(mb.all_nodes()) == 3

    def test_add_nodes_mixed(self, sigma, loc, node):
        mb = ModelBuilder()
        mb.add_nodes([node, loc], sigma)
        assert len(mb.nodes) == 3
        assert len(mb.all_nodes()) == 3

    def test_add_nodes_duplicate(self, sigma):
        """
        Duplicate nodes are removed when a model is built, so the ModelBuilder allows
        adding duplicates.
        """
        mb = ModelBuilder()
        mb.add_nodes(sigma, sigma)
        assert len(mb.nodes) == 2
        assert len(mb.all_nodes()) == 1

    def test_add_nodes_wrong_type(self, sigma):
        group = NodeGroup(sigma=sigma)
        mb = ModelBuilder()

        with pytest.raises(RecursionError):
            mb.add_nodes("foo")

        with pytest.raises(TypeError):
            mb.add_nodes(1)
            mb.add_nodes(1.0)
            mb.add_nodes(group)

    def test_add_groups_individually(self, sigma, loc, node):
        group = NodeGroup(sigma=sigma, loc=loc, node=node)
        mb = ModelBuilder()
        mb.add_groups(group)
        assert len(mb.all_nodes()) == 3
        assert len(mb.groups) == 1

    def test_add_groups_iterable(self, sigma, loc, node):
        group = NodeGroup(sigma=sigma, loc=loc, node=node)
        mb = ModelBuilder()
        mb.add_groups([group])
        assert len(mb.all_nodes()) == 3
        assert len(mb.groups) == 1

    def test_add_groups_wrong_types(self, sigma):
        mb = ModelBuilder()

        with pytest.raises(RecursionError):
            mb.add_groups("foo")

        with pytest.raises(TypeError):
            mb.add_groups(1)
            mb.add_groups(1.0)
            mb.add_groups(sigma)

    def test_build_model(self, sigma, loc, node):
        mb = ModelBuilder()
        mb.add_nodes(sigma, loc, node)
        model = mb.build()

        assert len(model.sorted_nodes) == 3

    def test_build_model_with_duplicates(self, sigma):
        mb = ModelBuilder()
        mb.add_nodes(sigma, sigma)
        model = mb.build()

        assert len(model.sorted_nodes) == 1

    def test_build_model_with_group(self, sigma, loc, node):
        group = NodeGroup(sigma=sigma, loc=loc, node=node)
        mb = ModelBuilder()
        mb.add_groups([group])

        model = mb.build()
        assert len(model.groups) == 1
        assert len(model.sorted_nodes) == 3


class TestPureModelFunctions:
    def test_jitted_logprob(self, model):
        """
        The compiled "pure" log probability function should return a log probability
        consistent with the value returned by scipy's logpdf function.
        """
        model.jaxify()
        jlog_prob_fn = jax.jit(get_log_prob)

        j_logp = float(jlog_prob_fn(model.state))

        expected = float(3 * scipy.stats.norm.logpdf(0.0, 0.0, 1.0))
        assert j_logp == pytest.approx(expected)

    def test_get_one_position(self, model):
        position = get_position(["foo", "mu"], model.state)
        assert "foo" in position
        assert "mu" in position
        assert len(position) == 2

    def test_update_state(self, model):
        model.jaxify()
        position = get_position(["foo", "mu"], model.state)
        position["foo"] = position["foo"].at[0].set(10)

        updated_state = update_state(position, model.state, model)
        assert get_position(["foo"], updated_state)["foo"][0] == 10
        assert get_position(["foo"], model.state)["foo"][0] == 10

    def test_make_update_state_fn(self, model):
        """
        Note that the function returned by make_update_state_fn does not modify the
        model in place, different from the ordinary update_state function.
        """
        update_state_fn = jax.jit(make_update_state_fn(model))

        model.jaxify()
        position = get_position(["foo", "mu"], model.state)
        position["foo"] = position["foo"].at[0].set(10)
        updated_state = update_state_fn(position, model.state)
        assert get_position(["foo"], updated_state)["foo"][0] == 10
        assert get_position(["foo"], model.state)["foo"][0] == 0

    def test_make_log_prob_fn(self, model):
        get_log_prob_fn = jax.jit(make_log_prob_fn(model))

        model.jaxify()
        position = get_position(["foo", "mu", "sigma"], model.state)
        log_prob = get_log_prob_fn(position, model.state)

        assert log_prob == get_log_prob(model.state)

    def test_goose_log_prob(self, model):
        gm = GooseModel(model)
        log_prob = gm.log_prob(model.state)

        assert log_prob == get_log_prob(model.state)

    def test_goose_update_state(self, model):
        model.jaxify()
        position = get_position(["foo", "mu"], model.state)
        position["foo"] = position["foo"].at[0].set(10)

        gm = GooseModel(model)
        updated_state = gm.update_state(position, model.state)
        assert get_position(["foo"], updated_state)["foo"][0] == 10

    def test_goose_extract_position(self, model):
        gm = GooseModel(model)
        position = gm.extract_position(["foo"], model.state)
        assert "foo" in position
        assert len(position) == 1
