from pytest import raises

from liesel.liesel import ModelBuilder, Node, NodeGroup


def test_name():
    """Tests the name properties of a node group."""

    group = NodeGroup()

    assert not group.has_name

    with raises(RuntimeError):
        group.name

    group.name = "group"

    assert group.has_name
    assert group.name == "group"

    copy = group.copy()

    assert copy.name == "group"


def test_model():
    """Tests if a node group can be added to a model via a builder."""

    n0 = Node(0.0)
    n1 = Node(0.0)

    g = NodeGroup(n0=n0, n1=n1)

    mb = ModelBuilder()
    mb.add_groups(g)

    m = mb.build()

    assert n0 in m.nodes.values()
    assert n1 in m.nodes.values()
    assert g in m.groups.values()
