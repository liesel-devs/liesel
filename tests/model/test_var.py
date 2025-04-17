import pickle
import typing
import warnings

import dill
import jax
import jax.numpy as jnp
import numpy as np
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


class TestVarTransform:
    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_weak_var_with_distribtution_inst(self, name) -> None:
        """
        Tests transformation of a weak var with distribution when the bijector is passed
        as an instance.
        """
        x = lsl.Var.new_value(jnp.linspace(0.1, 2, 5), name="all_x")
        batch_index = lsl.Var.new_value(1, name="index")

        x_batched = lsl.Var(
            value=lsl.Calc(lambda i, x: x[i], batch_index, x),
            distribution=lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0),
            name="x_batched",
        )

        x_batched_transformed = x_batched.transform(tfp.bijectors.Exp(), name=name)

        if name is None:
            assert x_batched_transformed.name == f"{x_batched.name}_transformed"
        else:
            assert x_batched_transformed.name == name

        assert x_batched_transformed.value == pytest.approx(jnp.log(x.value[1]))

        batch_index.value = 2
        x_batched_transformed.update()
        x_batched.update()
        assert x_batched_transformed.value == pytest.approx(jnp.log(x.value[2]))

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_weak_var_with_distribtution_class(self, name) -> None:
        """
        Tests transformation of a weak var with distribution when the bijector is passed
        as a class.
        """
        x = lsl.Var.new_value(jnp.linspace(0.1, 2, 5), name="all_x")
        batch_index = lsl.Var.new_value(1, name="index")

        x_batched = lsl.Var(
            value=lsl.Calc(lambda i, x: x[i], i=batch_index, x=x),
            distribution=lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0),
            name="x_batched",
        )

        x_batched_transformed = x_batched.transform(
            tfp.bijectors.Scale, scale=2.0, name=name
        )

        if name is None:
            assert x_batched_transformed.name == f"{x_batched.name}_transformed"
        else:
            assert x_batched_transformed.name == name

        assert x_batched_transformed.value == pytest.approx(x.value[1] / 2.0)

        batch_index.value = 2
        x_batched_transformed.update()
        x_batched.update()
        assert x_batched_transformed.value == pytest.approx(x.value[2] / 2.0)

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_weak_var_with_bijector_instance(self, name) -> None:
        tau = lsl.Var.new_param(10.0, name="tau")
        tau_sqrt = lsl.Var.new_calc(jnp.sqrt, tau)
        log_tau_sqrt = tau_sqrt.transform(tfp.bijectors.Exp(), name=name)

        if name is None:
            assert log_tau_sqrt.name == f"{tau_sqrt.name}_transformed"
        else:
            assert log_tau_sqrt.name == name

        assert tau.value == pytest.approx(10.0)
        assert tau_sqrt.value == pytest.approx(jnp.sqrt(10.0))
        assert log_tau_sqrt.value == pytest.approx(jnp.log(jnp.sqrt(10.0)))

        assert tau.strong
        assert tau_sqrt.weak
        assert log_tau_sqrt.weak
        assert tau.parameter
        assert not log_tau_sqrt.parameter
        assert not tau_sqrt.parameter

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_weak_var_with_bijector_class(self, name) -> None:
        tau = lsl.Var.new_param(10.0, name="tau")
        tau_sqrt = lsl.Var.new_calc(jnp.sqrt, tau)

        scale = lsl.Var.new_param(2.0, name="bijector_scale")
        scaled_tau_sqrt = tau_sqrt.transform(
            tfp.bijectors.Scale, scale=scale, name=name
        )

        if name is None:
            assert scaled_tau_sqrt.name == f"{tau_sqrt.name}_transformed"
        else:
            assert scaled_tau_sqrt.name == name

        assert tau.value == pytest.approx(10.0)
        assert tau_sqrt.value == pytest.approx(jnp.sqrt(10.0))
        assert scaled_tau_sqrt.value == pytest.approx(jnp.sqrt(10.0) / 2)

        assert tau.strong
        assert tau_sqrt.weak
        assert scaled_tau_sqrt.weak
        assert tau.parameter
        assert not scaled_tau_sqrt.parameter
        assert not tau_sqrt.parameter

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_without_dist_with_bijector_instance(self, name) -> None:
        tau = lsl.Var.new_param(10.0, name="tau")
        log_tau = tau.transform(tfp.bijectors.Exp(), name=name)

        if name is None:
            assert log_tau.name == f"{tau.name}_transformed"
        else:
            assert log_tau.name == name

        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(jnp.log(10.0))

        assert tau.weak
        assert log_tau.strong
        assert not tau.parameter
        assert log_tau.parameter

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_without_dist_with_bijector_class(self, name) -> None:
        tau = lsl.Var.new_param(10.0, name="tau")

        scale = lsl.Var.new_param(2.0, name="bijector_scale")
        log_tau = tau.transform(tfp.bijectors.Scale, scale=scale, name=name)

        if name is None:
            assert log_tau.name == f"{tau.name}_transformed"
        else:
            assert log_tau.name == name

        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(5.0)

        assert tau.weak
        assert log_tau.strong
        assert not tau.parameter
        assert log_tau.parameter

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_instance(self, name) -> None:
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        log_tau = tau.transform(tfp.bijectors.Exp(), name=name)
        tau.update()

        if name is None:
            assert log_tau.name == f"{tau.name}_transformed"
        else:
            assert log_tau.name == name

        assert tau.weak
        assert not log_tau.weak

        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(np.log(10.0))

        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (FutureWarning))
            log_tau_gb = lsl.GraphBuilder().transform(tau, tfp.bijectors.Exp())

        assert tau.weak
        assert not log_tau.weak

        tau.update()
        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau_gb.value == pytest.approx(np.log(10.0))

        log_tau.dist_node.update()  # type: ignore
        log_tau_gb.dist_node.update()  # type: ignore
        assert log_tau.log_prob == pytest.approx(log_tau_gb.log_prob)

    def test_transform_class_no_args(self) -> None:
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        with pytest.raises(ValueError):
            tau.transform(tfp.bijectors.Exp)

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_class_with_args(self, name) -> None:
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        transformed_tau = tau.transform(
            tfp.bijectors.Softplus, hinge_softness=lsl.Var(0.9), name=name
        )
        tau.update()

        if name is None:
            assert transformed_tau.name == f"{tau.name}_transformed"
        else:
            assert transformed_tau.name == name

        bijector = tfp.bijectors.Softplus(hinge_softness=0.9)

        assert tau.weak
        assert not transformed_tau.weak

        assert tau.value == pytest.approx(bijector.forward(transformed_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert transformed_tau.value == pytest.approx(bijector.inverse(10.0))

        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (FutureWarning))
            transformed_tau_gb = lsl.GraphBuilder().transform(
                tau, tfp.bijectors.Softplus, hinge_softness=lsl.Var(0.9)
            )

        assert tau.weak
        assert not transformed_tau.weak

        tau.update()
        assert tau.value == pytest.approx(bijector.forward(transformed_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert transformed_tau_gb.value == pytest.approx(bijector.inverse(10.0))

        transformed_tau.dist_node.update()  # type: ignore
        transformed_tau_gb.dist_node.update()  # type: ignore
        assert transformed_tau.log_prob == pytest.approx(transformed_tau_gb.log_prob)

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_default(self, name) -> None:
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        log_tau = tau.transform(name=name)
        tau.update()

        if name is None:
            assert log_tau.name == f"{tau.name}_transformed"
        else:
            assert log_tau.name == name

        assert tau.weak
        assert not log_tau.weak

        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(np.log(10.0))

        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (FutureWarning))
            log_tau_gb = lsl.GraphBuilder().transform(tau)

        assert tau.weak
        assert not log_tau.weak

        tau.update()
        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau_gb.value == pytest.approx(np.log(10.0))

        log_tau.dist_node.update()  # type: ignore
        log_tau_gb.dist_node.update()  # type: ignore
        assert log_tau.log_prob == pytest.approx(log_tau_gb.log_prob)

    def test_pickle_model_with_transformed_var(self, tmp_path):
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        _ = tau.transform()

        model = lsl.Model([tau])

        with pytest.raises(AttributeError):
            with open(tmp_path / "model.pkl", "wb") as f:
                pickle.dump(model, f)

            with open(tmp_path / "model.pkl", "rb") as f:
                model2 = pickle.load(f)

        with open(tmp_path / "model.pkl", "wb") as f:
            dill.dump(model, f)

        with open(tmp_path / "model.pkl", "rb") as f:
            model2 = dill.load(f)

        assert len(model2.vars) == len(model.vars)
        assert len(model2.nodes) == len(model.nodes)
        assert model2.log_prob == pytest.approx(model.log_prob)


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
