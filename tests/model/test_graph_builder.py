import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

import liesel.model.model as lmodel
import liesel.model.nodes as lnodes


def test_groups() -> None:
    v1 = lnodes.Var(0.0, name="v1")
    g1 = lnodes.Group("g1", var1=v1)
    gb = lmodel.GraphBuilder().add_groups(g1)
    assert v1 in gb.vars


def test_add_group_with_duplicate_name() -> None:
    v1 = lnodes.Var(0.0, name="v1")
    v2 = lnodes.Var(0.0, name="v2")
    g1 = lnodes.Group("g1", var1=v1)
    g2 = lnodes.Group("g1", var1=v2)
    with pytest.raises(RuntimeError):
        lmodel.GraphBuilder().add_groups(g1, g2)


def test_add_same_group_twice() -> None:
    v1 = lnodes.Var(0.0, name="v1")
    g1 = lnodes.Group("g1", var1=v1)
    gb = lmodel.GraphBuilder().add_groups(g1, g1)
    assert v1 in gb.vars


def test_manual_dtype_conversion(local_caplog) -> None:
    float_node = lnodes.Value(np.zeros(5, dtype="float64"), _name="float_node")
    int_node = lnodes.Value(np.zeros(5, dtype="int64"), _name="int_node")

    gb = lmodel.GraphBuilder()
    gb.add(float_node, int_node, to_float32=False)
    assert float_node.value.dtype == "float64"
    assert int_node.value.dtype == "int64"

    with local_caplog() as caplog:
        gb.convert_dtype("float64", "float32")
        assert float_node.value.dtype == "float32"
        assert int_node.value.dtype == "int64"

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert (
            caplog.records[0].msg == 'Converted dtype of Value(name="float_node").value'
        )

    gb.convert_dtype("int64", "int32")
    assert float_node.value.dtype == "float32"
    assert int_node.value.dtype == "int32"


def test_auto_dtype_conversion() -> None:
    float_node = lnodes.Value(np.zeros(5, dtype="float64"), _name="float_node")
    int_node = lnodes.Value(np.zeros(5, dtype="int64"), _name="int_node")

    gb = lmodel.GraphBuilder()
    gb.add(float_node, int_node)
    assert float_node.value.dtype == "float32"
    assert int_node.value.dtype == "int64"


def test_transform_raises_error_for_duplicate_nodes() -> None:
    lmbd = lnodes.Var(1.0, name="lambda")
    dist = lnodes.Dist(tfp.distributions.Exponential, lmbd)
    x = lnodes.Var(1.0, dist, name="x")

    gb = lmodel.GraphBuilder()

    model = gb.add(x).build_model()

    nodes, vars = model.copy_nodes_and_vars()

    with pytest.raises(RuntimeError):
        gb.add(*nodes.values(), *vars.values())
        x.transform()
