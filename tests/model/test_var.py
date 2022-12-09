import typing

import pytest
import tensorflow_probability.substrates.jax as tfp

import liesel.model.model as lmodel
import liesel.model.nodes as lnodes


def test_initialization() -> None:

    # simple data variable
    var0 = lnodes.Var(0, None, "")
    assert var0.value == 0
    assert var0.dist_node is None
    assert var0.name == ""
    assert isinstance(var0.value_node, lnodes.Data)

    # simple data variable with specified nodes
    dat = lnodes.Data(1)
    dist = lnodes.Dist(lnodes.NoDistribution())
    var1 = lnodes.Var(dat, dist, "foo")
    assert id(var1.value_node) == id(dat)
    assert id(var1.dist_node) == id(dist)
    assert var1.value == 1
    assert var1.name == "foo"


def test_default_values() -> None:
    # simple data variable
    var = lnodes.Var(0, None)
    assert var.name == ""
    assert var.dist_node is None
    assert var.role == ""
    assert var.parameter is False
    assert var.observed is False
    assert not var.groups
    assert not var.info
    assert not var.model


def test_name() -> None:
    var = lnodes.Var(0.0, None, "foo")

    assert var.name == "foo"

    var.name = "bar"
    assert var.name == "bar"

    with pytest.raises(RuntimeError):
        _ = lmodel.Model([var])
        var.name = "foo"


def test_role() -> None:
    var = lnodes.Var(0.0, None)

    assert var.role == ""

    var.role = "bar"
    assert var.role == "bar"

    _ = lmodel.Model([var])
    var.role = "foo"


def test_observed() -> None:
    var = lnodes.Var(0.0, None)

    assert not var.observed

    var.observed = True
    assert var.observed

    with pytest.raises(RuntimeError):
        _ = lmodel.Model([var])
        var.observed = False


def test_parameter() -> None:
    var = lnodes.Var(0.0, None)

    assert not var.parameter

    var.parameter = True
    assert var.parameter

    with pytest.raises(RuntimeError):
        _ = lmodel.Model([var])
        var.parameter = False


def test_groups() -> None:
    var = lnodes.Var(0.0, None)

    assert not var.groups

    grp = {
        ("grp", "role"),
    }
    var.groups = grp
    assert var.groups is grp

    _ = lmodel.Model([var])
    var.groups = set()


def test_info() -> None:
    var = lnodes.Var(0.0, None)

    info0 = {"foo": 1.0, "bar": "baz"}
    info1 = {"foo": 1.0}

    assert not var.info

    var.info = info0
    assert var.info is info0
    assert var.info is not info1

    # infos are always changable
    _ = lmodel.Model([var])
    var.info = info1
    assert var.info is info1
    assert var.info is not info0


def test_value_node() -> None:
    node0 = lnodes.Data(0.0)
    node1 = lnodes.Data(0.0)
    var = lnodes.Var(node0, None)

    assert var.value_node is node0
    assert var.value_node is not node1

    var.value_node = node1
    assert var.value_node is not node0
    assert var.value_node is node1

    with pytest.raises(RuntimeError):
        _ = lmodel.Model([var])
        var.value_node = node0


def test_dist_node() -> None:
    node0 = lnodes.Dist(lnodes.NoDistribution)
    node1 = lnodes.Dist(lnodes.NoDistribution)
    var = lnodes.Var(0.0, node0)

    assert var.dist_node is node0
    assert var.dist_node is not node1

    var.dist_node = node1
    assert var.dist_node is not node0
    assert var.dist_node is node1

    with pytest.raises(RuntimeError):
        _ = lmodel.Model([var])
        var.dist_node = node0


def test_property_strong_node() -> None:
    var = lnodes.Var(0.0, None)

    assert var.strong
    assert not var.weak


def test_property_weak_node() -> None:
    in0 = lnodes.Data(0.0)
    calc = lnodes.Calc(lambda x: x, in0)
    var = lnodes.Var(calc, None)

    assert not var.strong
    assert var.weak


@typing.no_type_check
def test_writing_weak_strong_fails() -> None:
    var = lnodes.Var(0.0, None)

    with pytest.raises(AttributeError):
        var.strong = False

    with pytest.raises(AttributeError):
        var.weak = False


def test_read_value() -> None:
    # strong node
    var0 = lnodes.Var(0, None)
    assert var0.value == 0

    # weak node
    in0 = lnodes.Data(0)
    calc = lnodes.Calc(lambda x: x + 1, in0)
    var1 = lnodes.Var(calc, None)

    # this value might be not 1 since
    # var.update() was not called
    assert var1.value is calc.value


def test_write_value() -> None:
    # strong node
    var0 = lnodes.Var(0, None)
    var0.value = 1
    assert var0.value == 1

    # weak node
    in0 = lnodes.Data(0)
    calc = lnodes.Calc(lambda x: x + 1, in0)
    var1 = lnodes.Var(calc, None)
    with pytest.raises(RuntimeError):
        var1.value = 2


def test_property_model() -> None:
    var = lnodes.Var(0.0)

    assert not var.model

    model = lmodel.Model([var])
    assert var.model

    model.pop_nodes_and_vars()
    assert not var.model


def test_method_nodes() -> None:
    in0 = lnodes.Data(0)
    calc = lnodes.Calc(lambda x: x + 1, in0)
    dist = lnodes.Dist(lnodes.NoDistribution())
    var0 = lnodes.Var(calc, dist)

    assert len(var0.nodes) == 3
    assert calc in var0.nodes
    assert dist in var0.nodes

    var1 = lnodes.Var(in0, None)
    assert len(var1.nodes) == 2
    assert in0 in var1.nodes
    assert calc not in var1.nodes
    assert dist not in var1.nodes


def test_update_value_strong():
    var = lnodes.Var(1)
    var.update()
    assert var.value == 1


def test_update_value_unfrozen_strong():
    var = lnodes.Var(1)
    var.update()


def test_update_value_weak():
    var0 = lnodes.Var(1, name="in")
    var1 = lnodes.Var(lnodes.Calc(lambda x: x + 1, var0.value_node), name="out")

    var1.update()
    assert var1.value == 2


def test_update_value_unfrozen_weak():
    var0 = lnodes.Var(1, name="in")
    var1 = lnodes.Var(lnodes.Calc(lambda x: x + 1, var0.value_node), name="out")
    var1.update()


# ------------- test all_inputs_* / all_outputs_* ------------------------


def test_all_input_nodes_strong_no_dist():
    var = lnodes.Var(0)
    assert len(var.all_input_nodes()) == 0


def test_all_input_nodes_weak_no_dist():
    x = lnodes.Data(1)
    var = lnodes.Var(lnodes.Calc(lambda x: x + 1, x))
    assert len(var.all_input_nodes()) == 1


def test_all_input_nodes_weak_no_dist_2():
    x = lnodes.Data(1)
    var = lnodes.Var(
        lnodes.Calc(
            lambda x, y: x + y,
            x,
            x,
        )
    )
    assert len(var.all_input_nodes()) == 1


def test_all_input_nodes_weak_no_dist_3():
    x = lnodes.Data(1)
    y = lnodes.Data(2)
    var = lnodes.Var(
        lnodes.Calc(
            lambda x, y: x + y,
            x,
            y,
        )
    )
    assert len(var.all_input_nodes()) == 2


def test_all_input_nodes_strong_w_dist():
    dist = lnodes.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0)

    var = lnodes.Var(0.0, dist)
    assert len(var.all_input_nodes()) == 3


def test_all_input_nodes_weak_w_dist():
    dist = lnodes.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0)
    x = lnodes.Data(1)
    var = lnodes.Var(lnodes.Calc(lambda x: x + 1, x), dist)
    assert len(var.all_input_nodes()) == 4


def test_all_input_nodes_weak_w_dist_2():
    dist = lnodes.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0)

    x = lnodes.Data(1)
    var = lnodes.Var(
        lnodes.Calc(
            lambda x, y: x + y,
            x,
            x,
        ),
        dist,
    )
    assert len(var.all_input_nodes()) == 4


def test_all_input_nodes_weak_w_dist_3():
    dist = lnodes.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0)
    x = lnodes.Data(1)
    y = lnodes.Data(2)
    var = lnodes.Var(
        lnodes.Calc(
            lambda x, y: x + y,
            x,
            y,
        ),
        dist,
    )
    assert len(var.all_input_nodes()) == 5


###


def test_all_input_vars_strong_no_dist():
    var = lnodes.Var(0)
    assert len(var.all_input_vars()) == 0


def test_all_input_vars_weak_no_dist():
    x = lnodes.Var(1)
    var = lnodes.Var(lnodes.Calc(lambda x: x + 1, x))
    assert len(var.all_input_vars()) == 1


def test_all_input_vars_weak_no_dist_2():
    x = lnodes.Var(1)
    var = lnodes.Var(
        lnodes.Calc(
            lambda x, y: x + y,
            x,
            x,
        )
    )
    assert len(var.all_input_vars()) == 1


def test_all_input_vars_weak_no_dist_3():
    x = lnodes.Var(1)
    y = lnodes.Var(2)
    var = lnodes.Var(
        lnodes.Calc(
            lambda x, y: x + y,
            x,
            y,
        )
    )
    assert len(var.all_input_vars()) == 2


def test_all_input_vars_strong_w_dist():
    dist = lnodes.Dist(
        tfp.distributions.Normal, loc=lnodes.Var(0.0), scale=lnodes.Var(1.0)
    )

    var = lnodes.Var(lnodes.Var(0.0), dist)
    assert len(var.all_input_vars()) == 3


def test_all_input_vars_weak_w_dist_1():
    def dist_mk():
        return lnodes.Dist(
            tfp.distributions.Normal, loc=lnodes.Var(0.0), scale=lnodes.Var(1.0)
        )

    x = lnodes.Var(1)
    y = lnodes.Var(1)
    z_node = lnodes.Data(1)
    var = lnodes.Var(lnodes.Calc(lambda x: x + 1, x), dist_mk())
    assert len(var.all_input_vars()) == 3

    var = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, x), dist_mk())
    assert len(var.all_input_vars()) == 3

    var = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, y), dist_mk())
    assert len(var.all_input_vars()) == 4

    var = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, z_node), dist_mk())
    assert len(var.all_input_vars()) == 3


def test_all_input_vars_weak_w_dist_2():
    def dist_mk():
        return lnodes.Dist(lnodes.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0))

    x = lnodes.Var(1)
    y = lnodes.Var(1)
    z_node = lnodes.Data(1)
    var = lnodes.Var(lnodes.Calc(lambda x: x + 1, x), dist_mk())
    assert len(var.all_input_vars()) == 1

    var = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, x), dist_mk())
    assert len(var.all_input_vars()) == 1

    var = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, y), dist_mk())
    assert len(var.all_input_vars()) == 2

    var = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, z_node), dist_mk())
    assert len(var.all_input_vars()) == 1


def test_all_output_vars():
    x = lnodes.Var(1, name="x")

    def dist_mk():
        return lnodes.Dist(tfp.distributions.Normal, loc=0.0, scale=x)

    y = lnodes.Var(1, name="y")
    var0 = lnodes.Var(lnodes.Calc(lambda x: x + 1, x), dist_mk(), name="var0")
    mod0 = lmodel.Model([var0] + [x, y], copy=True)
    assert len(mod0.vars["x"].all_output_vars()) == 1
    assert len(mod0.vars["y"].all_output_vars()) == 0
    assert len(mod0.vars["var0"].all_output_vars()) == 0

    var1 = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, x), dist_mk(), name="var1")
    mod1 = lmodel.Model([var0, var1] + [x, y], copy=True)
    assert len(mod1.vars["x"].all_output_vars()) == 2
    assert len(mod1.vars["y"].all_output_vars()) == 0
    assert len(mod1.vars["var1"].all_output_vars()) == 0

    var2 = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, y), dist_mk(), name="var2")
    mod2 = lmodel.Model([var0, var1, var2] + [x, y], copy=True)
    assert len(mod2.vars["x"].all_output_vars()) == 3
    assert len(mod2.vars["y"].all_output_vars()) == 1
    assert len(mod2.vars["var2"].all_output_vars()) == 0


def test_all_output_nodes():
    x = lnodes.Var(1, name="x")

    def dist_mk():
        return lnodes.Dist(tfp.distributions.Normal, loc=0.0, scale=x)

    y = lnodes.Var(1, name="y")
    var0 = lnodes.Var(lnodes.Calc(lambda x: x + 1, x), dist_mk(), name="var0")
    mod0 = lmodel.Model([var0] + [x, y], copy=True)
    assert len(mod0.vars["x"].all_output_nodes()) == 2
    assert len(mod0.vars["y"].all_output_nodes()) == 0
    assert (
        len(mod0.vars["var0"].all_output_nodes()) == 1 + 1
    )  # part of the _model_log_prob

    var1 = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, x), dist_mk(), name="var1")
    mod1 = lmodel.Model([var0, var1] + [x, y], copy=True)
    assert len(mod1.vars["x"].all_output_nodes()) == 4
    assert len(mod1.vars["y"].all_output_nodes()) == 0
    assert (
        len(mod1.vars["var1"].all_output_nodes()) == 1 + 1
    )  # part of the _model_log_prob

    var2 = lnodes.Var(lnodes.Calc(lambda x, y: x + y, x, y), dist_mk(), name="var2")
    mod2 = lmodel.Model([var0, var1, var2] + [x, y], copy=True)
    assert len(mod2.vars["x"].all_output_nodes()) == 6
    assert len(mod2.vars["y"].all_output_nodes()) == 1
    assert (
        len(mod2.vars["var2"].all_output_nodes()) == 1 + 1
    )  # part of the _model_log_prob


def test_indirect_connection() -> None:
    v0 = lnodes.Var(1.0, name="v0")
    n1 = lnodes.Calc(lambda x: 2.0 * x, v0, _name="n1")
    v2 = lnodes.Var(lnodes.Calc(lambda x: 2.0 * x, n1), name="v2")
    v3 = lnodes.Var(lnodes.Calc(lambda x: 2.0 * x, v2), name="v3")
    _ = lmodel.Model([v3])

    # test outputs
    outputs = v0.all_output_vars()
    assert len(outputs) == 1
    assert v2 in outputs
    assert v3 not in outputs

    # test inputs
    assert len(v3.all_input_vars()) == 1
    assert len(v2.all_input_vars()) == 1
    assert len(v0.all_input_vars()) == 0
