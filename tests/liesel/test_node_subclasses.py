import numpy as np
from pytest import approx, raises

from liesel.liesel import (
    PIT,
    Addition,
    Bijector,
    ColumnStack,
    Model,
    ModelBuilder,
    Node,
    NodeDistribution,
    Parameter,
    Smooth,
)


def test_parameter_initialization():
    """Tests the initialization of the `Parameter` node."""

    n_loc = Node(0.0)
    n_scale = Node(1.0)

    distribution = NodeDistribution("Normal", loc=n_loc, scale=n_scale)
    n_param = Parameter(1.0, distribution)

    assert n_param.value == 1.0
    n_param.initialize_with_mean()
    assert n_param.value == 0.0


def test_parameter_model():
    """Tests the model properties of the `Parameter` node."""

    n_loc = Node(0.0)
    n_scale = Node(1.0)

    distribution = NodeDistribution("Normal", loc=n_loc, scale=n_scale)
    n_param = Parameter(1.0, distribution)

    assert not n_param.has_model

    with raises(RuntimeError):
        n_param.model

    mb = ModelBuilder()
    mb.add_nodes(n_param)
    model = mb.build()

    assert n_param.value == 1.0

    assert n_param.has_model
    assert n_param.model is model

    with raises(RuntimeError):
        model = Model([n_param])

    assert n_loc.grad() == 1.0
    assert n_param.grad() == -1.0


def test_addition_without_distribution():
    """Tests the `Addition` node without a distribution."""

    n0 = Node(13.0)
    n1 = Node(73.0)
    n2 = Node(1.0)

    n_add = Addition(n0, n1, n2)

    assert n_add.value == 87.0

    with raises(RuntimeError):
        Addition()


def test_addition_with_distribution():
    """Tests the `Addition` node with a distribution."""

    n0 = Node(1.0)
    n1 = Node(0.0)
    n2 = Node(1.0)

    distribution = NodeDistribution("Normal", loc=Node(0.0), scale=Node(1.0))
    n_add = Addition(n0, n1, n2, distribution=distribution)

    assert n_add.log_prob == approx(-2.918939)


def test_bijector():
    """Tests the `Bijector` node."""

    n0 = Node(1.0)

    n_exp = Bijector("Exp", n0)
    assert n_exp.value == approx(2.718282)

    n_log = Bijector("Exp", n0, inverse=True)
    assert n_log.value == 0.0


def test_columnstack_without_distribution():
    """Tests the `ColumnStack` node without a distribution."""

    n0 = Node(13.0)
    n1 = Node(73.0)
    n2 = Node(1.0)

    n_cs = ColumnStack(n0, n1, n2)

    assert np.all(n_cs.value == np.array([13.0, 73.0, 1.0]))

    with raises(RuntimeError):
        ColumnStack()


def test_columnstack_with_distribution():
    """Tests the `ColumnStack` node with a distribution."""

    n0 = Node(13.0)
    n1 = Node(73.0)
    n2 = Node(1.0)

    n_loc = Node(np.zeros(3))
    n_cov = Node(np.identity(3))

    distribution = NodeDistribution(
        "MultivariateNormalFullCovariance", loc=n_loc, covariance_matrix=n_cov
    )

    n_cs = ColumnStack(n0, n1, n2, distribution=distribution)

    assert n_cs.log_prob == approx(-2752.256815599614)


def test_pit():
    """Tests the `PIT` node."""

    distribution = NodeDistribution("Normal", loc=Node(0.0), scale=Node(1.0))
    node = PIT(Node(1.0, distribution))

    assert node.value == approx(0.8413447)


def test_smooth():
    """Tests the `Smooth` node."""

    x = np.random.randn(3, 3)
    beta = np.random.randn(3)

    node = Smooth(x=Node(x), beta=Node(beta))

    assert np.all(node.value == x @ beta)
