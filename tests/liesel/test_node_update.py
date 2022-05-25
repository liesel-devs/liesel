from pytest import approx

from liesel.liesel import Addition, Bijector, ModelBuilder, Node

# n0    n1    n2
#   \  /     /
#    n3    n4
#      \  /
#       n5

n0 = Node(1.0)
n1 = Node(2.0)
n2 = Node(3.0)
n3 = Addition(n0, n1)
n4 = Bijector("Identity", n2)
n5 = Addition(n3, n4)

builder = ModelBuilder()
builder.add_nodes(n0, n1, n2, n3, n4, n5)
model = builder.build()


def test_values():
    """Tests if the calculated values are correct."""

    assert n3.value == approx(3.0)
    assert n4.value == approx(3.0)
    assert n5.value == approx(6.0)


def test_outdated():
    """Tests if the outdated flags are set correctly."""

    n0.set_value(2.0, update=False)

    assert n0.outdated
    assert not n1.outdated
    assert not n2.outdated
    assert n3.outdated
    assert not n4.outdated
    assert n5.outdated


def test_update():
    """Tests if the model is updated correctly."""

    model.update()

    assert n0.value == approx(2.0)
    assert n3.value == approx(4.0)
    assert n5.value == approx(7.0)

    assert all(not n.outdated for n in model.sorted_nodes)
