"""
# Model nodes
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from functools import reduce
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    TypeVar,
    cast,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np

from liesel.option import Option, WeakOption
from liesel.tfp.jax import bijectors as jb
from liesel.tfp.jax import distributions as jd
from liesel.tfp.numpy import bijectors as nb
from liesel.tfp.numpy import distributions as nd

from .goose import make_log_prob_fn
from .types import (
    Array,
    ModelState,
    NodeState,
    Position,
    TFPBijector,
    TFPBijectorClass,
    TFPDistribution,
    TFPDistributionClass,
)

if TYPE_CHECKING:
    from .model import Model


def _input_args(inputs: NodeInputs, num: int | None = None) -> str:
    items = inputs.items()
    args = [f"{node:s}" for key, node in items if isinstance(key, int)]
    kwargs = [f"{key}={node:s}" for key, node in items if isinstance(key, str)]
    return _join_args([_join_args(args, num), _join_args(kwargs, num)])


def _join_args(args: Iterable[str], num: int | None = None) -> str:
    args = [arg for arg in args if arg != ""]

    if num is not None and len(args) > num:
        args = args[0:num] + ["..."]

    return ", ".join(args)


def _num_arg(x: Any, key: str | None = None) -> str:
    if key is not None:
        key = f"{key}="
    else:
        key = ""

    arg = re.sub(r"\((?s:.*)\)", "(...)", repr(x))

    return f"{key}{arg}"


def _opt_arg(x: Any, key: str | None = None) -> str:
    if x is None:
        return ""

    if key is not None:
        key = f"{key}="
    else:
        key = ""

    return f"{key}{repr(x)}"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Node components ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class NodeInputs(tuple):
    """The inputs of a node component as a tuple with optional names."""

    def __new__(cls, *args, **kwargs):
        tup = args + tuple(kwargs.values())
        return super().__new__(cls, tup)

    def __init__(self, *args, **kwargs):
        self._dict = dict(enumerate(args))
        self._dict.update(kwargs)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)

        if isinstance(key, str):
            return self._dict[key]

        raise TypeError(f"Indices of {repr(self)} must be integers, slices or strings")

    def __getnewargs__(self):
        return self

    def __repr__(self):
        cls = type(self).__name__
        args = _input_args(self, 1)
        return f"{cls}({args})"

    def dict(self):
        """Returns the node inputs as a dictionary."""

        return self._dict.copy()

    def keys(self):
        """Returns the keys of the node inputs dictionary."""

        return self.dict().keys()

    def values(self):
        """Returns the values of the node inputs dictionary."""

        return self.dict().values()

    def items(self):
        """Returns the items of the node inputs dictionary."""

        return self.dict().items()

    def replace(self, old, new):
        """
        Returns a shallow copy of the node inputs with the `new` node instead of
        the `old` one.
        """

        def fn(node):
            return new if node is old else node

        args = [fn(node) for key, node in self.items() if isinstance(key, int)]
        kwargs = {key: fn(node) for key, node in self.items() if isinstance(key, str)}
        return NodeInputs(*args, **kwargs)


class NodeComponent(ABC):
    """
    An abstract node component with inputs and a NumPy module.

    Superclass of the `NodeDistribution` and the `NodeCalculator`.
    """

    def __init__(self, *args: Node, **kwargs: Node) -> None:
        self._inputs = NodeInputs(*args, **kwargs)
        self._jaxified = False

    @property
    def inputs(self) -> NodeInputs:
        """The inputs of the node component."""

        return self._inputs

    @property
    def jaxified(self) -> bool:
        """Whether JAX NumPy is enabled for the node component."""

        return self._jaxified

    @jaxified.setter
    def jaxified(self, jaxified: bool) -> None:
        if jaxified:
            self.jaxify()
        else:
            self.unjaxify()

    def jaxify(self) -> NodeComponent:
        """Enables JAX NumPy for the node component."""

        self._jaxified = True
        return self

    @property
    def _np(self) -> ModuleType:
        if self.jaxified:
            return jnp

        return np

    def unjaxify(self) -> NodeComponent:
        """Disables JAX NumPy for the node component."""

        self._jaxified = False
        return self

    def __repr__(self) -> str:
        cls = type(self).__name__
        args = _input_args(self.inputs, 1)
        return f"{cls}({args})"


class NodeCalculator(NodeComponent):
    """
    An abstract value calculator of a node.

    Computes the value of a node from its inputs.
    Superclass of the `AdditionCalculator`, the `BijectorCalculator`, etc.
    """

    @abstractmethod
    def value(self) -> Array:
        """Calculates the value of a node from its inputs and returns it."""


class NodeDistribution(NodeComponent):
    """
    A probability distribution of a node.

    Computes the log-probability of a node from its inputs.
    Implemented as a thin wrapper around a TFP distribution.

    ## Parameters

    - `distribution`:
      The name of a TFP distribution as a string, or alternatively, a user-defined
      TFP-compatible distribution class.

      If a class is provided instead of a string, the user needs to make sure it uses
      the right NumPy implementation.

    - `bijector`:
      The name of a TFP bijector as a string, or alternatively, a user-defined
      TFP-compatible bijector class.

      If a class is provided instead of a string, the user needs to make sure it uses
      the right NumPy implementation. Defaults to None.

    - `inputs`:
      The inputs of the distribution.

      The keywords must match the arguments of the TFP distribution.
    """

    def __init__(
        self,
        distribution: str | TFPDistributionClass,
        bijector: str | TFPBijectorClass | None = None,
        **inputs: Node,
    ) -> None:
        super().__init__(**inputs)
        self._distribution = distribution
        self._bijector = bijector

    def distribution(self) -> TFPDistribution:
        """The TFP distribution initialized with the values of the inputs."""

        distribution_module = jd if self.jaxified else nd

        if isinstance(self._distribution, str):
            distribution_cls = getattr(distribution_module, self._distribution)
        else:
            distribution_cls = self._distribution

        kwargs = {kw: node.value for kw, node in self.inputs.items()}
        distribution = distribution_cls(**kwargs)

        if self._bijector is not None:
            bijector_module = jb if self.jaxified else nb

            if isinstance(self._bijector, str):
                bijector_cls = getattr(bijector_module, self._bijector)
            else:
                bijector_cls = self._bijector

            transformed_cls = distribution_module.TransformedDistribution
            distribution = transformed_cls(distribution, bijector_cls())

        return distribution

    def transform(self, bijector: str | TFPBijectorClass) -> NodeDistribution:
        """Transforms the distribution with a TFP bijector."""

        if self._bijector is not None:
            msg = f"Cannot transform {repr(self)}, as it is already transformed"
            raise RuntimeError(msg)

        inputs = self.inputs.dict()

        return NodeDistribution(self._distribution, bijector, **inputs)

    def cdf(self, value: Array, **kwargs) -> Array:
        """
        The cumulative distribution function of the distribution.

        The arguments are passed on to the TFP distribution.
        """

        return self.distribution().cdf(value, **kwargs)

    def log_prob(self, value: Array, **kwargs) -> Array:
        """
        The log-probability (density) function of the distribution.

        The arguments are passed on to the TFP distribution.
        """

        log_probs = self.distribution().log_prob(value, **kwargs)
        return self._np.sum(log_probs)

    def mean(self, **kwargs) -> Array:
        """
        The mean function of the distribution.

        The arguments are passed on to the TFP distribution.
        """

        return self.distribution().mean(**kwargs)

    def sample(self, sample_shape=(), seed=None, **kwargs) -> Array:
        """
        The sampling function of the distribution.

        The arguments are passed on to the TFP distribution.
        """

        return self.distribution().sample(sample_shape, seed, **kwargs)

    def __repr__(self) -> str:
        cls = type(self).__name__

        distribution = repr(self._distribution)
        bijector = _opt_arg(self._bijector)
        inputs = _input_args(self.inputs)

        args = _join_args([distribution, bijector, inputs])

        return f"{cls}({args})"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Node ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


GradientFunction = Callable[[Position, ModelState], Position]
TNodeCalculator = TypeVar("TNodeCalculator", bound=NodeCalculator)


class Node(Generic[TNodeCalculator]):
    """
    A node, strong or weak, with or without a probability distribution,
    which can be used to build probabilistic graphical models.

    ## Attributes

    - `outdated`:
      Whether the node is outdated.

      A node is outdated if its value or the value of one of its inputs has changed.
      The value and the log-probability of an outdated node need to be recomputed.

    - `outputs`:
      An output of a node A is a node B which depends on the value of A.
    """

    @overload
    def __init__(
        self: Node[NodeCalculator],
        value: Array,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self: Node[TNodeCalculator],
        value: TNodeCalculator,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        ...

    def __init__(
        self,
        value: Array | TNodeCalculator,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        if isinstance(value, NodeCalculator):
            calculator = cast(TNodeCalculator, value)
            value = 0.0
        else:
            calculator = None

        self._calculator: Option[TNodeCalculator] = Option(calculator)
        self._distribution: Option[NodeDistribution] = Option(distribution)
        self._grad_fn: Option[GradientFunction] = Option(None)
        self._jaxified: bool = False
        self._log_prob: Array = 0.0
        self._model: WeakOption[Model] = WeakOption(None)
        self._name: Option[str] = Option(name)
        self.outdated: bool = True
        self.outputs: set[Node] = set()
        self._value: Array = value

        for input in self.inputs:
            input.outputs.add(self)

        self.update()

    @property
    def calculator(self) -> TNodeCalculator:
        """The calculator of the node."""

        return self._calculator.expect(f"{repr(self)} does not have a calculator")

    @property
    def distribution(self) -> NodeDistribution:
        """The distribution of the node."""

        return self._distribution.expect(f"{repr(self)} does not have a distribution")

    def grad(self) -> Array:
        """
        Returns the gradient of the model log-probability w.r.t. the node value.

        This method is **unlikely to be efficient**. Consider using JAX explicitly
        to JIT-compile more complex functions. Alternatively, you can use Goose.
        """

        if self._grad_fn.is_none():
            self._grad_fn = Option(jax.jit(jax.grad(make_log_prob_fn(self.model))))

        grad_fn = self._grad_fn.unwrap()
        position = Position({self.name: self.value})
        model_state = self.model.state

        grad_dict = grad_fn(position, model_state)
        grad_array = self._np.asarray(grad_dict[self.name])
        return grad_array

    @property
    def has_calculator(self) -> bool:
        """Whether the node has a calculator."""

        return self._calculator.is_some()

    @property
    def has_distribution(self) -> bool:
        """Whether the node has a distribution."""

        return self._distribution.is_some()

    @property
    def has_model(self) -> bool:
        """Whether the node is part of a model."""

        return self._model.is_some()

    @property
    def has_name(self) -> bool:
        """Whether the node has a name."""

        return self._name.is_some()

    def input_value_changed(self) -> Node:
        """
        Informs the node that the value of one of its inputs has changed.

        Flags the node as outdated, and if the node is weak, also flags its outputs
        as outdated.
        """

        self.outdated = True

        if self.weak:
            self.own_value_changed()

        return self

    @property
    def inputs(self) -> set[Node]:
        """All inputs of the node as a set."""

        inputs: set[Node] = set()
        self._calculator.map(lambda x: inputs.update(x.inputs))
        self._distribution.map(lambda x: inputs.update(x.inputs))
        return inputs

    @property
    def jaxified(self) -> bool:
        """Whether JAX NumPy is enabled for the node."""

        return self._jaxified

    @jaxified.setter
    def jaxified(self, jaxified: bool) -> None:
        if jaxified:
            self.jaxify()
        else:
            self.unjaxify()

    def jaxify(self) -> Node:
        """Enables JAX NumPy for the node."""

        self._value = jnp.asarray(self.value)
        self._log_prob = jnp.asarray(self.log_prob)

        self._calculator.map(lambda x: x.jaxify())
        self._distribution.map(lambda x: x.jaxify())

        self._jaxified = True
        return self

    @property
    def log_prob(self) -> Array:
        """The log-probability of the node."""

        return self._log_prob

    @property
    def model(self) -> Model:
        """The model the node is part of."""

        return self._model.expect(f"{repr(self)} is not part of a model")

    @model.setter
    def model(self, model: Model) -> None:
        if self.has_model:
            raise RuntimeError(f"{repr(self)} is already part of a model")

        self._grad_fn = Option(None)
        self._model = WeakOption(model)

    @property
    def name(self) -> str:
        """The name of the node."""

        return self._name.expect(f"{repr(self)} does not have a name")

    @name.setter
    def name(self, name: str) -> None:
        if self.has_model:
            msg = f"Cannot change the name of {repr(self)}, as it is part of a model"
            raise RuntimeError(msg)

        self._name = Option(name)

    @property
    def _np(self) -> ModuleType:
        if self.jaxified:
            return jnp

        return np

    def own_value_changed(self) -> Node:
        """
        Informs the node that its value has changed.

        Flags the node and its outputs as outdated.
        """

        self.outdated = True

        for node in self.outputs:
            node.input_value_changed()

        return self

    def set_value(self, value: Array, update: bool = True) -> Node:
        """
        Sets the value of the node.

        Flags the node and its outputs as outdated, and if requested,
        updates the model.
        """

        if self.weak:
            msg = f"Cannot set the value of {repr(self)}, as it is a weak node"
            raise RuntimeError(msg)

        self._value = value
        self.own_value_changed()

        if update:
            self._model.map(lambda x: x.update())

        return self

    @property
    def state(self) -> NodeState:
        """The value and the log-probability as a `NodeState`."""

        return NodeState(self.value, self.log_prob)

    @state.setter
    def state(self, state: NodeState) -> None:
        self._value, self._log_prob = state

    @property
    def strong(self) -> bool:
        """Whether the node is strong."""

        return self._calculator.is_none()

    def unjaxify(self) -> Node:
        """Disables JAX NumPy for the node."""

        self._value = np.asarray(self.value)
        self._log_prob = np.asarray(self.log_prob)

        self._calculator.map(lambda x: x.unjaxify())
        self._distribution.map(lambda x: x.unjaxify())

        self._jaxified = False
        return self

    def update(self) -> Node:
        """
        Updates the value and the log-probability of the node.

        Assumes that the inputs of the node are up-to-date.
        """

        if self.weak:
            self._value = self.calculator.value()

        if self.has_distribution:
            self._log_prob = self.distribution.log_prob(self.value)

        self.outdated = False
        return self

    def validate(self) -> Node:
        """Checks if the value and the log-probability of the node are finite."""

        assert self._np.all(self._np.isfinite(self.value))
        assert self._np.all(self._np.isfinite(self.log_prob))
        return self

    @property
    def value(self) -> Array:
        """The value of the node."""

        return self._value

    @value.setter
    def value(self, value: Array) -> None:
        self.set_value(value, update=True)

    @property
    def weak(self) -> bool:
        """Whether the node is weak."""

        return self._calculator.is_some()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_grad_fn"] = Option(None)
        state["_model"] = WeakOption(None)
        return state

    def __format__(self, format_spec: str) -> str:
        if format_spec == "s":
            return self.__short_repr__()

        return self.__repr__()

    def __repr__(self) -> str:
        cls = type(self).__name__

        value = _num_arg(self.value)
        calculator = _opt_arg(self._calculator.value)
        distribution = _opt_arg(self._distribution.value)
        name = _opt_arg(self._name.value, "name")

        first_arg = value if self.strong else calculator
        args = _join_args([first_arg, distribution, name])

        if cls != "Node":
            return f"{cls}(Node({args}))"

        return f"Node({args})"

    def __short_repr__(self) -> str:
        cls = type(self).__name__

        if self.has_name:
            return f"{cls}(..., name={repr(self.name)})"

        return f"{cls}(...)"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Node group ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class NodeGroup(dict):
    def __init__(self, name: str | None = None, **nodes: Node) -> None:
        super().__init__(**nodes)
        self._name = Option(name)

    def copy(self) -> NodeGroup:
        """Returns a shallow copy of the node group."""

        return NodeGroup(self._name.value, **self)

    @property
    def has_name(self) -> bool:
        """Whether the node group has a name."""

        return self._name.is_some()

    @property
    def name(self) -> str:
        """The name of the node group."""

        return self._name.expect(f"{repr(self)} does not have a name")

    @name.setter
    def name(self, name: str) -> None:
        self._name = Option(name)

    def __repr__(self) -> str:
        cls = type(self).__name__
        name = _opt_arg(self._name.value)
        nodes = [f"{key}={node:s}" for key, node in self.items()]
        args = _join_args([name, *nodes], 3)
        return f"{cls}({args})"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Strong nodes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DesignMatrix(Node):
    """A strong node representing a design matrix."""


class Hyperparameter(Node):
    """A strong node representing a hyperparameter."""


class Response(Node):
    """A strong node representing a response vector or matrix."""


class Parameter(Node):
    """A strong node representing a model parameter."""

    def __init__(
        self,
        value: Array,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(value, distribution, name)

    def initialize_with_mean(self) -> Parameter:
        """Initializes the value of the parameter with its prior mean."""

        def safe_mean() -> Array:
            assert self.has_distribution

            mean = self._np.squeeze(self.distribution.mean())
            mean = self._np.broadcast_to(mean, self._np.shape(self.value))
            assert self._np.all(self._np.isfinite(mean))

            return mean

        try:
            self.value = safe_mean()
        except (AssertionError, NotImplementedError):
            pass

        return self


class RegressionCoef(Parameter):
    """A parameter node representing a vector of regression coefficients."""


class SmoothingParam(Parameter):
    """A parameter node representing a smoothing parameter."""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weak addition nodes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class AdditionCalculator(NodeCalculator):
    """
    Calculates the element-wise sum of its inputs.

    Must have one or more inputs.
    """

    def __init__(self, *inputs: Node) -> None:
        super().__init__(*inputs)

        if len(self.inputs) == 0:
            raise RuntimeError(f"{repr(self)} must have one or more inputs")

    def value(self) -> Array:
        xs = [node.value for node in self.inputs]
        return reduce(self._np.add, xs)


class Addition(Node[AdditionCalculator]):
    """A weak node with an `AdditionCalculator`."""

    def __init__(
        self,
        *inputs: Node,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        calculator = AdditionCalculator(*inputs)
        super().__init__(calculator, distribution, name)


class Predictor(Addition):
    """An addition node representing a regression predictor."""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weak bijector nodes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class BijectorCalculator(NodeCalculator):
    """
    Evaluates the `forward` or the `inverse` method of a TFP bijector at its input.
    """

    def __init__(
        self, bijector: str | TFPBijectorClass, input: Node, inverse: bool = False
    ) -> None:
        super().__init__(input)
        self._bijector = bijector
        self._inverse = inverse

    @property
    def bijector(self) -> TFPBijector:
        module = jb if self.jaxified else nb

        if isinstance(self._bijector, str):
            cls = getattr(module, self._bijector)
        else:
            cls = self._bijector

        return cls()

    def value(self) -> Array:
        x = self.inputs[0].value

        if self._inverse:
            return self.bijector.inverse(x)

        return self.bijector.forward(x)


class Bijector(Node[BijectorCalculator]):
    """A weak node with a `BijectorCalculator`."""

    def __init__(
        self,
        bijector: str | TFPBijectorClass,
        input: Node,
        inverse: bool = False,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        calculator = BijectorCalculator(bijector, input, inverse)
        super().__init__(calculator, distribution, name)


class InverseLink(Bijector):
    """A bijector node representing an inverse link function."""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weak column-stack node ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ColumnStackCalculator(NodeCalculator):
    """
    Stacks its inputs column-wise.

    Must have one or more inputs.
    """

    def __init__(self, *inputs: Node) -> None:
        super().__init__(*inputs)

        if len(self.inputs) == 0:
            raise RuntimeError(f"{repr(self)} must have one or more inputs")

    def value(self) -> Array:
        xs = [node.value for node in self.inputs]
        return self._np.column_stack(xs)


class ColumnStack(Node[ColumnStackCalculator]):
    """A weak node with a `ColumnStackCalculator`."""

    def __init__(
        self,
        *inputs: Node,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        calculator = ColumnStackCalculator(*inputs)
        super().__init__(calculator, distribution, name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weak PIT node ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class PITCalculator(NodeCalculator):
    """Calculates the probability integral transform of its input."""

    def __init__(self, input, **kwargs):
        super().__init__(_pit_main_input=input, **kwargs)

    def value(self) -> Array:
        input = self.inputs["_pit_main_input"]
        return input.distribution.cdf(input.value)


class PIT(Node[PITCalculator]):
    """A weak node with a `PITCalculator`."""

    def __init__(
        self,
        input: Node,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        kwargs = input.distribution.inputs.dict()
        calculator = PITCalculator(input, **kwargs)
        super().__init__(calculator, distribution, name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weak smooth node ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class SmoothCalculator(NodeCalculator):
    """
    Calculates a smooth `x @ beta`.

    A smooth is the matrix-vector product of a design matrix `x` and a vector of
    regression coefficients `beta`.
    """

    def __init__(self, x, beta):
        super().__init__(x=x, beta=beta)

    def value(self) -> Array:
        x = self.inputs["x"].value
        beta = self.inputs["beta"].value
        return self._np.matmul(x, beta)


class Smooth(Node[SmoothCalculator]):
    """A weak node with a `SmoothCalculator`."""

    def __init__(
        self,
        x: Node,
        beta: Node,
        distribution: NodeDistribution | None = None,
        name: str | None = None,
    ) -> None:
        calculator = SmoothCalculator(x=x, beta=beta)
        super().__init__(calculator, distribution, name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameter transformation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def transform_parameter(node: Node, bijector: str | TFPBijectorClass) -> NodeGroup:
    """
    Transforms a parameter and its distribution with a TFP bijector.

    Returns a node group of the transformed and the original parameter.
    """

    if node.has_name:
        group_name = f"{node.name}_transformation"
        transformed_name = f"{node.name}_transformed"
        original_name = node.name
    else:
        group_name = "transformation"
        transformed_name = "transformed"
        original_name = "original"

    if isinstance(bijector, str):
        _bijector = getattr(nb, bijector)
    else:
        _bijector = bijector

    value = _bijector().forward(node.value)
    distribution = node.distribution.transform(bijector)
    transformed = Parameter(value, distribution, transformed_name)
    original = Bijector(bijector, transformed, inverse=True, name=original_name)
    transformed.jaxified = node.jaxified
    original.jaxified = node.jaxified

    return NodeGroup(group_name, transformed=transformed, original=original)
