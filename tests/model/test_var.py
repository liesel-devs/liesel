import typing

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax as tfp

import liesel.model as lsl
from liesel.distributions.nodist import NoDistribution


def test_initialization() -> None:
    # simple data variable
    var0 = lsl.Var(0, None, "")
    assert var0.value == 0
    assert var0.dist_node is None
    assert var0.name == ""
    assert isinstance(var0.value_node, lsl.Value)

    # simple data variable with specified nodes
    dat = lsl.Value(1)
    dist = lsl.Dist(NoDistribution())
    var1 = lsl.Var(dat, dist, "foo")
    assert id(var1.value_node) == id(dat)
    assert id(var1.dist_node) == id(dist)
    assert var1.value == 1
    assert var1.name == "foo"


def test_default_values() -> None:
    # simple data variable
    var = lsl.Var(0, None)
    assert var.name == ""
    assert var.dist_node is None
    assert var.role == ""
    assert var.parameter is False
    assert var.observed is False
    assert not var.groups
    assert not var.info
    assert not var.model


def test_name() -> None:
    var = lsl.Var(0.0, None, "foo")

    assert var.name == "foo"

    var.name = "bar"
    assert var.name == "bar"

    with pytest.raises(RuntimeError):
        _ = lsl.Model([var])
        var.name = "foo"


def test_role() -> None:
    var = lsl.Var(0.0, None)

    assert var.role == ""

    var.role = "bar"
    assert var.role == "bar"

    _ = lsl.Model([var])
    var.role = "foo"


def test_observed() -> None:
    var = lsl.Var(0.0, None)

    assert not var.observed

    var.observed = True
    assert var.observed

    with pytest.raises(RuntimeError):
        _ = lsl.Model([var])
        var.observed = False


def test_parameter() -> None:
    var = lsl.Var(0.0, None)

    assert not var.parameter

    var.parameter = True
    assert var.parameter

    with pytest.raises(RuntimeError):
        _ = lsl.Model([var])
        var.parameter = False


def test_info() -> None:
    var = lsl.Var(0.0, None)

    info0 = {"foo": 1.0, "bar": "baz"}
    info1 = {"foo": 1.0}

    assert not var.info

    var.info = info0
    assert var.info is info0
    assert var.info is not info1

    # infos are always changable
    _ = lsl.Model([var])
    var.info = info1
    assert var.info is info1
    assert var.info is not info0


def test_value_node() -> None:
    node0 = lsl.Value(0.0)
    node1 = lsl.Value(0.0)
    var = lsl.Var(node0, None)

    assert var.value_node is node0
    assert var.value_node is not node1

    var.value_node = node1
    assert var.value_node is not node0
    assert var.value_node is node1

    with pytest.raises(RuntimeError):
        _ = lsl.Model([var])
        var.value_node = node0


def test_dist_node() -> None:
    node0 = lsl.Dist(NoDistribution)
    node1 = lsl.Dist(NoDistribution)
    var = lsl.Var(0.0, node0)

    assert var.dist_node is node0
    assert var.dist_node is not node1

    var.dist_node = node1
    assert var.dist_node is not node0
    assert var.dist_node is node1

    with pytest.raises(RuntimeError):
        _ = lsl.Model([var])
        var.dist_node = node0


def test_property_strong_node() -> None:
    var = lsl.Var(0.0, None)

    assert var.strong
    assert not var.weak


def test_property_weak_node() -> None:
    in0 = lsl.Value(0.0)
    calc = lsl.Calc(lambda x: x, in0)
    var = lsl.Var(calc, None)

    assert not var.strong
    assert var.weak


@typing.no_type_check
def test_writing_weak_strong_fails() -> None:
    var = lsl.Var(0.0, None)

    with pytest.raises(AttributeError):
        var.strong = False

    with pytest.raises(AttributeError):
        var.weak = False


def test_read_value() -> None:
    # strong node
    var0 = lsl.Var(0, None)
    assert var0.value == 0

    # weak node
    in0 = lsl.Value(0)
    calc = lsl.Calc(lambda x: x + 1, in0)
    var1 = lsl.Var(calc, None)

    # this value might be not 1 since
    # var.update() was not called
    assert var1.value is calc.value


def test_write_value() -> None:
    # strong node
    var0 = lsl.Var(0, None)
    var0.value = 1
    assert var0.value == 1

    # weak node
    in0 = lsl.Value(0)
    calc = lsl.Calc(lambda x: x + 1, in0)
    var1 = lsl.Var(calc, None)
    with pytest.raises(RuntimeError):
        var1.value = 2


def test_property_model() -> None:
    var = lsl.Var(0.0)

    assert not var.model

    model = lsl.Model([var])
    assert var.model

    model.pop_nodes_and_vars()
    assert not var.model


def test_auto_transform():
    var = lsl.Var(1, name="var")
    var.auto_transform = True

    assert var.auto_transform

    var.auto_transform = False

    assert not var.auto_transform


def test_method_nodes() -> None:
    in0 = lsl.Value(0)
    calc = lsl.Calc(lambda x: x + 1, in0)
    dist = lsl.Dist(NoDistribution())
    var0 = lsl.Var(calc, dist)

    assert len(var0.nodes) == 3
    assert calc in var0.nodes
    assert dist in var0.nodes

    var1 = lsl.Var(in0, None)
    assert len(var1.nodes) == 2
    assert in0 in var1.nodes
    assert calc not in var1.nodes
    assert dist not in var1.nodes


def test_update_value_strong():
    var = lsl.Var(1)
    var.update()
    assert var.value == 1


def test_update_value_unfrozen_strong():
    var = lsl.Var(1)
    var.update()


def test_update_value_weak():
    var0 = lsl.Var(1, name="in")
    var1 = lsl.Var(lsl.Calc(lambda x: x + 1, var0.value_node), name="out")

    var1.update()
    assert var1.value == 2


def test_update_value_unfrozen_weak():
    var0 = lsl.Var(1, name="in")
    var1 = lsl.Var(lsl.Calc(lambda x: x + 1, var0.value_node), name="out")
    var1.update()


# ------------- test all_inputs_* / all_outputs_* ------------------------


def test_all_input_nodes_strong_no_dist():
    var = lsl.Var(0)
    assert len(var.all_input_nodes()) == 0


def test_all_input_nodes_weak_no_dist():
    x = lsl.Value(1)
    var = lsl.Var(lsl.Calc(lambda x: x + 1, x))
    assert len(var.all_input_nodes()) == 1


def test_all_input_nodes_weak_no_dist_2():
    x = lsl.Value(1)
    var = lsl.Var(
        lsl.Calc(
            lambda x, y: x + y,
            x,
            x,
        )
    )
    assert len(var.all_input_nodes()) == 1


def test_all_input_nodes_weak_no_dist_3():
    x = lsl.Value(1)
    y = lsl.Value(2)
    var = lsl.Var(
        lsl.Calc(
            lambda x, y: x + y,
            x,
            y,
        )
    )
    assert len(var.all_input_nodes()) == 2


def test_all_input_nodes_strong_w_dist():
    dist = lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0)

    var = lsl.Var(0.0, dist)
    assert len(var.all_input_nodes()) == 3


def test_all_input_nodes_weak_w_dist():
    dist = lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0)
    x = lsl.Value(1)
    var = lsl.Var(lsl.Calc(lambda x: x + 1, x), dist)
    assert len(var.all_input_nodes()) == 4


def test_all_input_nodes_weak_w_dist_2():
    dist = lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0)

    x = lsl.Value(1)
    var = lsl.Var(
        lsl.Calc(
            lambda x, y: x + y,
            x,
            x,
        ),
        dist,
    )
    assert len(var.all_input_nodes()) == 4


def test_all_input_nodes_weak_w_dist_3():
    dist = lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0)
    x = lsl.Value(1)
    y = lsl.Value(2)
    var = lsl.Var(
        lsl.Calc(
            lambda x, y: x + y,
            x,
            y,
        ),
        dist,
    )
    assert len(var.all_input_nodes()) == 5


###


def test_all_input_vars_strong_no_dist():
    var = lsl.Var(0)
    assert len(var.all_input_vars()) == 0


def test_all_input_vars_weak_no_dist():
    x = lsl.Var(1)
    var = lsl.Var(lsl.Calc(lambda x: x + 1, x))
    assert len(var.all_input_vars()) == 1


def test_all_input_vars_weak_no_dist_2():
    x = lsl.Var(1)
    var = lsl.Var(
        lsl.Calc(
            lambda x, y: x + y,
            x,
            x,
        )
    )
    assert len(var.all_input_vars()) == 1


def test_all_input_vars_weak_no_dist_3():
    x = lsl.Var(1)
    y = lsl.Var(2)
    var = lsl.Var(
        lsl.Calc(
            lambda x, y: x + y,
            x,
            y,
        )
    )
    assert len(var.all_input_vars()) == 2


def test_all_input_vars_strong_w_dist():
    dist = lsl.Dist(tfp.distributions.Normal, loc=lsl.Var(0.0), scale=lsl.Var(1.0))

    var = lsl.Var(lsl.Var(0.0), dist)
    assert len(var.all_input_vars()) == 3


def test_all_input_vars_weak_w_dist_1():
    def dist_mk():
        return lsl.Dist(tfp.distributions.Normal, loc=lsl.Var(0.0), scale=lsl.Var(1.0))

    x = lsl.Var(1)
    y = lsl.Var(1)
    z_node = lsl.Value(1)
    var = lsl.Var(lsl.Calc(lambda x: x + 1, x), dist_mk())
    assert len(var.all_input_vars()) == 3

    var = lsl.Var(lsl.Calc(lambda x, y: x + y, x, x), dist_mk())
    assert len(var.all_input_vars()) == 3

    var = lsl.Var(lsl.Calc(lambda x, y: x + y, x, y), dist_mk())
    assert len(var.all_input_vars()) == 4

    var = lsl.Var(lsl.Calc(lambda x, y: x + y, x, z_node), dist_mk())
    assert len(var.all_input_vars()) == 3


def test_all_input_vars_weak_w_dist_2():
    def dist_mk():
        return lsl.Dist(lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0))

    x = lsl.Var(1)
    y = lsl.Var(1)
    z_node = lsl.Value(1)
    var = lsl.Var(lsl.Calc(lambda x: x + 1, x), dist_mk())
    assert len(var.all_input_vars()) == 1

    var = lsl.Var(lsl.Calc(lambda x, y: x + y, x, x), dist_mk())
    assert len(var.all_input_vars()) == 1

    var = lsl.Var(lsl.Calc(lambda x, y: x + y, x, y), dist_mk())
    assert len(var.all_input_vars()) == 2

    var = lsl.Var(lsl.Calc(lambda x, y: x + y, x, z_node), dist_mk())
    assert len(var.all_input_vars()) == 1


def test_all_output_vars():
    x = lsl.Var(1, name="x")

    def dist_mk():
        return lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=x)

    y = lsl.Var(1, name="y")
    var0 = lsl.Var(lsl.Calc(lambda x: x + 1, x), dist_mk(), name="var0")
    mod0 = lsl.Model([var0] + [x, y], copy=True)
    assert len(mod0.vars["x"].all_output_vars()) == 1
    assert len(mod0.vars["y"].all_output_vars()) == 0
    assert len(mod0.vars["var0"].all_output_vars()) == 0

    var1 = lsl.Var(lsl.Calc(lambda x, y: x + y, x, x), dist_mk(), name="var1")
    mod1 = lsl.Model([var0, var1] + [x, y], copy=True)
    assert len(mod1.vars["x"].all_output_vars()) == 2
    assert len(mod1.vars["y"].all_output_vars()) == 0
    assert len(mod1.vars["var1"].all_output_vars()) == 0

    var2 = lsl.Var(lsl.Calc(lambda x, y: x + y, x, y), dist_mk(), name="var2")
    mod2 = lsl.Model([var0, var1, var2] + [x, y], copy=True)
    assert len(mod2.vars["x"].all_output_vars()) == 3
    assert len(mod2.vars["y"].all_output_vars()) == 1
    assert len(mod2.vars["var2"].all_output_vars()) == 0


def test_all_output_nodes():
    x = lsl.Var(1, name="x")

    def dist_mk():
        return lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=x)

    y = lsl.Var(1, name="y")
    var0 = lsl.Var(lsl.Calc(lambda x: x + 1, x), dist_mk(), name="var0")
    mod0 = lsl.Model([var0] + [x, y], copy=True)
    assert len(mod0.vars["x"].all_output_nodes()) == 2
    assert len(mod0.vars["y"].all_output_nodes()) == 0
    assert (
        len(mod0.vars["var0"].all_output_nodes()) == 1 + 1
    )  # part of the _model_log_prob

    var1 = lsl.Var(lsl.Calc(lambda x, y: x + y, x, x), dist_mk(), name="var1")
    mod1 = lsl.Model([var0, var1] + [x, y], copy=True)
    assert len(mod1.vars["x"].all_output_nodes()) == 4
    assert len(mod1.vars["y"].all_output_nodes()) == 0
    assert (
        len(mod1.vars["var1"].all_output_nodes()) == 1 + 1
    )  # part of the _model_log_prob

    var2 = lsl.Var(lsl.Calc(lambda x, y: x + y, x, y), dist_mk(), name="var2")
    mod2 = lsl.Model([var0, var1, var2] + [x, y], copy=True)
    assert len(mod2.vars["x"].all_output_nodes()) == 6
    assert len(mod2.vars["y"].all_output_nodes()) == 1
    assert (
        len(mod2.vars["var2"].all_output_nodes()) == 1 + 1
    )  # part of the _model_log_prob


def test_indirect_connection() -> None:
    v0 = lsl.Var(1.0, name="v0")
    n1 = lsl.Calc(lambda x: 2.0 * x, v0, _name="n1")
    v2 = lsl.Var(lsl.Calc(lambda x: 2.0 * x, n1), name="v2")
    v3 = lsl.Var(lsl.Calc(lambda x: 2.0 * x, v2), name="v3")
    _ = lsl.Model([v3])

    # test outputs
    outputs = v0.all_output_vars()
    assert len(outputs) == 1
    assert v2 in outputs
    assert v3 not in outputs

    # test inputs
    assert len(v3.all_input_vars()) == 1
    assert len(v2.all_input_vars()) == 1
    assert len(v0.all_input_vars()) == 0


class TestVarConstructors:
    def test_new_param(self):
        loc = lsl.Var.new_param(1.0, name="loc")
        assert isinstance(loc, lsl.Var)
        assert loc.parameter
        assert loc.value_node.monitor
        assert loc.strong

        dist = lsl.Dist(tfp.distributions.Normal, 0.0, 1.0)
        loc = lsl.Var.new_param(1.0, dist, name="loc")
        assert isinstance(loc, lsl.Var)
        assert loc.parameter
        assert loc.value_node.monitor
        assert loc.strong

    def test_new_obs(self):
        loc = lsl.Var.new_obs(1.0, name="loc")
        assert isinstance(loc, lsl.Var)
        assert loc.observed
        assert loc.strong

        dist = lsl.Dist(tfp.distributions.Normal, 0.0, 1.0)
        loc = lsl.Var.new_obs(1.0, dist, name="loc")
        assert isinstance(loc, lsl.Var)
        assert loc.observed
        assert loc.strong

    def test_new_calc(self):
        loc = lsl.Var.new_calc(lambda x: x + 1.0, 1.0, name="loc")
        assert isinstance(loc, lsl.Var)
        assert loc.value == pytest.approx(2.0)
        assert loc.weak

    def test_new_calc_with_dist(self):
        loc = lsl.Var.new_calc(
            lambda x: x + 1.0,
            distribution=lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0),
            x=1.0,
            name="loc",
        )
        loc.update()
        assert isinstance(loc, lsl.Var)
        assert loc.value == pytest.approx(2.0)
        assert loc.weak
        assert loc.dist_node is not None
        assert loc.log_prob is not None

    def test_new_const(self):
        loc = lsl.Var.new_value(1.0, name="loc")
        assert isinstance(loc, lsl.Var)
        assert loc.strong


class TestVarPredictions:
    def test_predict(self) -> None:
        n = 10
        x = jax.random.uniform(jax.random.PRNGKey(1), (n,))
        b = 1.0
        e = jax.random.normal(jax.random.PRNGKey(2), (n,))
        y = x * b + e

        xvar = lsl.Var.new_obs(x, name="x")
        bvar = lsl.Var.new_param(jnp.array([b]), name="b")
        loc = lsl.Var.new_calc(lambda x, b: x * b, x=xvar, b=bvar, name="loc")
        scale = lsl.Var.new_param(jnp.array([1.0]), name="scale")
        scale.transform(tfp.bijectors.Exp())
        yvar = lsl.Var.new_obs(
            y, lsl.Dist(tfp.distributions.Normal, loc=loc, scale=scale), name="y"
        )

        _ = lsl.Model([yvar])

        samples = {"b": jax.random.uniform(jax.random.PRNGKey(3), (4, 7))}

        pred = loc.predict(samples)
        assert jnp.allclose(pred, x * jnp.expand_dims(samples["b"], -1))
        assert pred.shape[-1] == x.shape[-1]

        # predict at new observations with same shape
        xnew = jax.random.uniform(jax.random.PRNGKey(5), (n,))
        pred = loc.predict(samples, newdata={"x": xnew})

        assert jnp.allclose(pred, xnew * jnp.expand_dims(samples["b"], -1))
        assert pred.shape[-1] == x.shape[-1]

        # predict at new grid of observations
        xnew = jnp.linspace(0, 10)
        pred = loc.predict(samples, newdata={"x": xnew})

        assert jnp.allclose(pred, xnew * jnp.expand_dims(samples["b"], -1))
        assert pred.shape[-1] != x.shape[-1]
        assert pred.shape[-1] == xnew.shape[-1]
