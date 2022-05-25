from pytest import approx

from liesel.liesel import ModelBuilder, Node, NodeDistribution, transform_parameter

a = Node(0.01)
b = Node(0.01)
ig = NodeDistribution("InverseGamma", concentration=a, scale=b)
tau2 = Node(1.0, distribution=ig)

group = transform_parameter(tau2, "Log")

builder = ModelBuilder()
builder.add_groups(group)
model = builder.build()


def test_transform():
    """Tests if the graph, the values and the log-probabilities are correct."""

    assert group["original"].inputs == {group["transformed"]}

    assert group["transformed"].value == approx(0.0)
    assert group["original"].value == approx(1.0)

    assert group["transformed"].log_prob == approx(-4.6555314)
    assert group["original"].log_prob == approx(0.0)
