from typing import Any, assert_type

import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.model import Dist, Group, Model, Node, Var


class CustomVar(Var):
    def custom_method(self) -> float:
        return 42.0


def test_dynamic_variable_lookups() -> None:
    """Named lookups support subclass APIs at runtime and under type checking."""
    x = CustomVar(0.0, name="x")
    x.parameter = True
    y = Var.new_obs(1.0, Dist(tfd.Normal, loc=x, scale=1.0), name="y")
    group = Group("example", x=x, y=y)
    model = Model(y)

    assert_type(model.vars["x"], Any)
    assert_type(model.parameters["x"], Any)
    assert_type(model.observed["y"], Any)
    assert_type(model.vars.copy()["x"], Any)
    assert_type(model.nodes[x.value_node.name], Node)
    assert_type(group.vars["x"], Any)
    assert_type(group.nodes_and_vars["x"], Any)
    assert_type(group["x"], Any)

    assert model.vars["x"] is x
    assert model.vars["x"].custom_method() == 42.0
    assert model.parameters["x"].custom_method() == 42.0
    assert group.vars["x"].custom_method() == 42.0
    assert group.nodes_and_vars["x"].custom_method() == 42.0
    assert group["x"].custom_method() == 42.0

    copied = model.copy_vars()
    nodes, variables = model.copy_nodes_and_vars()
    assert_type(copied["x"], Any)
    assert_type(variables["x"], Any)
    assert_type(nodes[x.value_node.name], Node)
    assert copied["x"] is not x
    assert copied["x"].custom_method() == 42.0
    assert variables["x"].custom_method() == 42.0
    copied["y"].dist_node["loc"] = Var(2.0)
    assert copied["y"].dist_node["loc"].value == 2.0
    assert model.observed["y"].dist_node["loc"].value == 0.0
