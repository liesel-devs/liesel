from jaxlib.xla_extension import DeviceArray as JAXArray
from numpy import ndarray as NumPyArray
from pytest import approx, raises

from liesel.liesel import Bijector, Model, Node, NodeDistribution, NodeState


def test_calculator():
    """Tests the calculator properties of a node."""

    strong = Node(0.0)
    weak = Bijector("Exp", strong)
    weak.update()

    # test the strong node

    assert strong.strong
    assert not strong.weak

    assert strong.value == 0.0

    with raises(RuntimeError):
        strong.calculator

    # test the weak node

    assert weak.weak
    assert not weak.strong

    assert weak.value == approx(1.0)

    # test setting a value

    strong.value = 1.0
    weak.update()

    assert strong.value == 1.0
    assert weak.value == approx(2.718282)

    with raises(RuntimeError):
        weak.value = 0.0


def test_distribution():
    """Tests the distribution properties of a node."""

    loc = Node(0.0)
    scale = Node(1.0)
    distribution = NodeDistribution("Normal", loc=loc, scale=scale)
    random = Node(0.0, distribution=distribution)
    random.update()

    # test the parameter nodes

    assert not loc.has_distribution
    assert loc.log_prob == 0.0

    with raises(RuntimeError):
        loc.distribution

    assert not scale.has_distribution
    assert scale.log_prob == 0.0

    with raises(RuntimeError):
        scale.distribution

    # test the random node

    assert random.has_distribution
    assert random.log_prob == approx(-0.9189385)
    assert random.distribution == distribution


def test_jaxify():
    """Tests if a node can be jaxified and unjaxified."""

    strong = Node(0.0)
    weak = Bijector("Exp", strong)
    weak.update()

    # test jaxification

    strong.jaxify()
    weak.jaxified = True

    assert isinstance(strong.value, JAXArray)
    assert isinstance(strong.log_prob, JAXArray)

    assert isinstance(weak.value, JAXArray)
    assert isinstance(weak.log_prob, JAXArray)

    weak.update()

    assert isinstance(weak.value, JAXArray)
    assert isinstance(weak.log_prob, JAXArray)

    # test unjaxification

    strong.unjaxify()
    weak.jaxified = False

    assert isinstance(strong.value, NumPyArray)
    assert isinstance(strong.log_prob, NumPyArray)

    assert isinstance(weak.value, NumPyArray)
    assert isinstance(weak.log_prob, NumPyArray)

    weak.update()

    assert isinstance(weak.value, NumPyArray)
    assert isinstance(weak.log_prob, NumPyArray)


def test_model():
    """Tests the model properties of a node."""

    node = Node(0.0)

    assert not node.has_model

    with raises(RuntimeError):
        node.model

    model = Model([node])

    assert node.has_model
    assert node.model is model

    with raises(RuntimeError):
        model = Model([node])


def test_name():
    """Tests the name properties of a node."""

    named = Node(0.0, name="foo")
    unnamed = Node(0.0)

    # test the named node

    assert named.has_name
    assert named.name == "foo"

    named.name = "bar"

    assert named.name == "bar"

    model = Model([named])  # noqa: F841

    with raises(RuntimeError):
        named.name = "foobar"

    # test the unnamed node

    assert not unnamed.has_name

    with raises(RuntimeError):
        unnamed.name


def test_state():
    """Tests the state property of a node."""

    node = Node(0.0)

    assert node.state == NodeState(0.0, 0.0)

    node.state = NodeState(1.0, 2.0)

    assert node.state == NodeState(1.0, 2.0)

    assert node.value == 1.0
    assert node.log_prob == 2.0
