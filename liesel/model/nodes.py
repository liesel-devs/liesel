"""
Nodes and variables.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, Union

import tensorflow_probability.substrates.jax.bijectors as jb
import tensorflow_probability.substrates.jax.distributions as jd
import tensorflow_probability.substrates.numpy.bijectors as nb
import tensorflow_probability.substrates.numpy.distributions as nd

from ..distributions.nodist import NoDistribution

if TYPE_CHECKING:
    from .model import Model

__all__ = [
    "Array",
    "Bijector",
    "Calc",
    "Data",
    "Dist",
    "Distribution",
    "InputGroup",
    "Node",
    "NodeState",
    "Obs",
    "Param",
    "TransientCalc",
    "TransientDist",
    "TransientIdentity",
    "TransientNode",
    "Var",
    "add_group",
]

Array = Any
Distribution = Union[jd.Distribution, nd.Distribution]
Bijector = Union[jb.Bijector, nb.Bijector]


def in_model_method(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not self.model:
            raise RuntimeError(
                f"{repr(self)} is not part of a model, cannot call {fn.__name__}()"
            )
        return fn(self, *args, **kwargs)

    return wrapped


def in_model_getter(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not self.model:
            raise RuntimeError(
                f"{repr(self)} is not part of a model, cannot call '{fn.__name__}'"
            )
        return fn(self, *args, **kwargs)

    return wrapped


def no_model_method(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if self.model:
            raise RuntimeError(
                f"{repr(self)} is part of a model, cannot call {fn.__name__}()"
            )
        return fn(self, *args, **kwargs)

    return wrapped


def no_model_setter(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if self.model:
            raise RuntimeError(
                f"{repr(self)} is part of a model, cannot set '{fn.__name__}'"
            )
        return fn(self, *args, **kwargs)

    return wrapped


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nodes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class NodeState(NamedTuple):
    """The state of a node."""

    value: Any
    """The value of the node."""

    outdated: bool
    """Whether the node is outdated."""

    extra: Any = None
    """Optional extra information."""


class Node(ABC):
    """A node of a computational graph."""

    def __init__(
        self,
        *inputs: Any,
        _name: str = "",
        _needs_seed: bool = False,
        **kwinputs: Any,
    ):
        self._groups: set[tuple[str, str]] = set()
        self._inputs = tuple(self._to_node(_input) for _input in inputs)
        self._kwinputs = {kw: self._to_node(_input) for kw, _input in kwinputs.items()}
        self._kwinputs_proxy = MappingProxyType(self._kwinputs)
        self._model: weakref.ref[Model] | Callable[[], None] = lambda: None
        self._name = _name
        self._needs_seed = _needs_seed
        self._outdated = True
        self._outputs: frozenset[Node] = frozenset()
        self._value: Any = None
        self._var: Var | None = None

        self.monitor = False
        """Whether the node should be monitored by an inference algorithm."""

    def _add_output(self, output: Node) -> Node:
        self._outputs = self._outputs | frozenset({output})
        return self

    def _clear_outputs(self) -> Node:
        self._outputs = frozenset()
        return self

    def _set_model(self, model: Model) -> Node:
        if self.model:
            raise RuntimeError(f"{repr(self)} can only be part of one model")

        self._model = weakref.ref(model)
        return self

    def _set_var(self, var: Var) -> Node:
        if self.var:
            raise RuntimeError(f"{repr(self)} can only be part of one var")

        self._var = var
        return self

    @staticmethod
    def _to_node(x: Any) -> Node:
        if isinstance(x, Var):
            return x.var_value_node

        if not isinstance(x, Node):
            return Data(x)

        return x

    def _unset_model(self) -> Node:
        self._model = lambda: None
        return self

    def _unset_var(self) -> Node:
        self._var = None
        return self

    def all_input_nodes(self) -> frozenset[Node]:
        """Returns all non-keyword and keyword input nodes as a frozen set."""
        return frozenset(self.inputs) | frozenset(self.kwinputs.values())

    @in_model_method
    def all_output_nodes(self) -> frozenset[Node]:
        """Returns all output nodes as a frozen set."""
        return self.outputs

    def clear_state(self) -> Node:
        """Clears the state of the node."""
        self.state = NodeState(None, True)
        return self

    @in_model_method
    def flag_outdated(self) -> Node:
        """Flags the node and its recursive outputs as outdated."""
        self._outdated = True

        for node in self._outputs:
            node.flag_outdated()

        return self

    @property
    def groups(self) -> set[tuple[str, str]]:
        """
        The groups the node is part of.

        The groups are defined as a set of tuples, each of which consists of
        the group name and the key of the node in the group, e.g.::

            {
                ("group1", "key_in_group1"),
                ("group2", "key_in_group2"),
            }
        """
        return self._groups

    @groups.setter
    def groups(self, groups: set[tuple[str, str]]):
        self._groups = groups

    @property
    def inputs(self) -> tuple[Node, ...]:
        """The non-keyword input nodes."""
        return self._inputs

    @property
    def kwinputs(self) -> MappingProxyType[str, Node]:
        """The keyword input nodes."""
        return self._kwinputs_proxy

    @property
    def model(self) -> Model | None:
        """The model the node is part of."""
        return self._model()

    @property
    def name(self) -> str:
        """The name of the node."""
        return self._name

    @name.setter
    @no_model_setter
    def name(self, name: str):
        self._name = name

    @property
    def needs_seed(self) -> bool:
        """Whether the node needs a seed / PRNG key."""
        return self._needs_seed

    @needs_seed.setter
    @no_model_setter
    def needs_seed(self, needs_seed: bool):
        self._needs_seed = needs_seed

    @property
    def outdated(self) -> bool:
        """Whether the node is outdated."""

        if not self.model:
            return True

        return self._outdated

    @property
    @in_model_getter
    def outputs(self) -> frozenset[Node]:
        """The output nodes."""
        return self._outputs

    @no_model_method
    def set_inputs(self, *inputs: Any, **kwinputs: Any) -> Node:
        """Sets the non-keyword and keyword input nodes."""
        self._inputs = tuple(self._to_node(_input) for _input in inputs)

        self._kwinputs.clear()
        kwinputs = {kw: self._to_node(_input) for kw, _input in kwinputs.items()}
        self._kwinputs.update(kwinputs)

        return self

    @property
    def state(self) -> NodeState:
        """
        The state of the node.

        For the default node, a :class:`.NodeState` with the value and the
        outdated flag, but subclasses can add extra information to the state.
        """
        return NodeState(self.value, self.outdated)

    @state.setter
    def state(self, state: NodeState):
        self._value = state.value
        self._outdated = state.outdated

    @abstractmethod
    def update(self) -> Node:
        """Updates the value of the node."""

    @property
    def value(self) -> Any:
        """
        The value of the node.

        Can only be set for a :class:`.Data` node, but not a :class:`.Calc` or
        :class:`.Dist` node. If the node is part of a :class:`.Model` ``m`` with
        ``m.auto_update == True``, setting the value of the node triggers an update
        of the model. The auto-update can be disabled to improve the performance if
        multiple model parameters are updated at once.
        """
        return self._value

    @property
    def var(self) -> Var | None:
        """The variable the node is part of."""
        return self._var

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_kwinputs_proxy"] = None
        state["_model"] = self._model()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._kwinputs_proxy = MappingProxyType(self._kwinputs)

        if self._model is not None:
            self._model = weakref.ref(self._model)
        else:
            self._model = lambda: None

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self.name}>"


class TransientNode(Node):
    """
    A node that does not cache its value.

    A transient node is outdated if and only if at least one of its input nodes
    is outdated. The :attr:`.outdated` property checks this condition on-the-fly.
    """

    @property
    def outdated(self) -> bool:
        """
        Whether the node is outdated.

        A transient node is outdated if and only if at least one of its input nodes
        is outdated. This condition is checked on-the-fly.
        """

        if not self.model:
            return True

        return any(_input.outdated for _input in self.all_input_nodes())

    @property
    def state(self) -> NodeState:
        """The state of the node with the value ``None``."""
        return NodeState(None, self.outdated)

    @state.setter
    def state(self, state: NodeState):
        self._value = state.value
        self._outdated = state.outdated

    def update(self):
        """Does nothing."""
        return self

    @property
    @abstractmethod
    def value(self) -> Any:
        """
        The value of the node.

        Computed on-the-fly.
        """


class ArgGroup(NamedTuple):
    """A group of arguments as a named tuple of ``args`` and ``kwargs``."""

    args: list[Any]
    """The non-keyword arguments."""

    kwargs: dict[str, Any]
    """The keyword arguments."""


class InputGroup(TransientNode):
    """
    A node that groups its inputs for another node.

    Essentially, this node "forwards" the values of its inputs to its outputs
    as an :class:`.ArgGroup`.
    """

    @property
    def value(self) -> ArgGroup:
        args = [_input.value for _input in self.inputs]
        kwargs = {kw: _input.value for kw, _input in self.kwinputs.items()}
        return ArgGroup(args, kwargs)


class Data(Node):
    """A data node. Always up-to-date."""

    def __init__(self, value: Any, _name: str = ""):
        super().__init__(_name=_name)
        self._value = value

    def flag_outdated(self) -> Data:
        """Stops the recursion setting outdated flags."""
        return self

    @property
    def outdated(self) -> bool:
        return False

    def update(self) -> Data:
        """Does nothing."""
        return self

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any):
        self._value = value

        if self.model:
            for node in self.outputs:
                node.flag_outdated()

            if self.model.auto_update:
                self.model.update()


class Calc(Node):
    """A calculator node."""

    def __init__(
        self,
        function: Callable[..., Any],
        *inputs: Any,
        _name: str = "",
        _needs_seed: bool = False,
        **kwinputs: Any,
    ):
        super().__init__(*inputs, **kwinputs, _name=_name, _needs_seed=_needs_seed)
        self._function = function

    @property
    def function(self) -> Callable[..., Any]:
        """The wrapped function."""
        return self._function

    @function.setter
    @no_model_setter
    def function(self, function: Callable[..., Any]):
        self._function = function

    def update(self) -> Calc:
        args = [_input.value for _input in self.inputs]
        kwargs = {kw: _input.value for kw, _input in self.kwinputs.items()}
        self._value = self.function(*args, **kwargs)
        self._outdated = False
        return self


class TransientCalc(TransientNode, Calc):
    """A transient calculator node that does not cache its value."""

    @property
    def value(self) -> Any:
        args = [_input.value for _input in self.inputs]
        kwargs = {kw: _input.value for kw, _input in self.kwinputs.items()}
        return self.function(*args, **kwargs)


class TransientIdentity(TransientCalc):
    """
    A transient identity node that does not cache its value.

    Essentially, this node "forwards" the value of its input to its outputs.
    """

    def __init__(self, _input: Any, _name: str = ""):
        super().__init__(lambda x: x, _input, _name=_name)


class VarValue(TransientIdentity):
    """
    A proxy node for the value of a :class:`.Var`.

    This node type is used to keep the references to a variable intact,
    even if the underlying value node is replaced.
    """


class Dist(Node):
    """A distribution node."""

    def __init__(
        self,
        distribution: Callable[..., Distribution],
        *inputs: Any,
        _name: str = "",
        _needs_seed: bool = False,
        **kwinputs: Any,
    ):
        super().__init__(*inputs, **kwinputs, _name=_name, _needs_seed=_needs_seed)

        self._at: Node | None = None
        self._distribution = distribution
        self._per_obs = True

    def all_input_nodes(self) -> frozenset[Node]:
        inputs = super().all_input_nodes()

        if self.at:
            inputs = inputs | frozenset({self.at})

        return inputs

    @property
    def at(self) -> Node | None:
        """Where to evaluate the distribution."""
        return self._at

    @at.setter
    @no_model_setter
    def at(self, at: Node | None):
        if self.var and at is not self.var.var_value_node:
            raise RuntimeError(
                f"{repr(self)} is part of a var, cannot set property `at`"
            )

        self._at = at

    @property
    def distribution(self) -> Callable[..., Distribution]:
        """The wrapped distribution."""
        return self._distribution

    @distribution.setter
    @no_model_setter
    def distribution(self, distribution: Callable[..., Distribution]):
        self._distribution = distribution

    def init_dist(self) -> Distribution:
        """Initializes the distribution."""
        args = [_input.value for _input in self.inputs]
        kwargs = {kw: _input.value for kw, _input in self.kwinputs.items()}
        dist = self.distribution(*args, **kwargs)
        return dist

    @property
    def log_prob(self) -> Array:
        """The log-probability of the distribution."""
        return self.value

    @property
    def per_obs(self) -> bool:
        """Whether the log-probability is stored per observation or summed up."""
        return self._per_obs

    @per_obs.setter
    @no_model_setter
    def per_obs(self, per_obs: bool):
        self._per_obs = per_obs

    def update(self) -> Dist:
        if not self.at:
            raise RuntimeError(
                f"{repr(self)} cannot evaluate log-prob, property `at` not set"
            )

        log_prob = self.init_dist().log_prob(self.at.value)

        if not self.per_obs and hasattr(log_prob, "sum"):
            log_prob = log_prob.sum()

        self._value = log_prob
        self._outdated = False
        return self


class TransientDist(TransientNode, Dist):
    """A transient distribution node that does not cache its value."""

    @property
    def value(self) -> Any:
        if not self.at:
            raise RuntimeError(
                f"{repr(self)} cannot evaluate log-prob, property `at` not set"
            )

        log_prob = self.init_dist().log_prob(self.at.value)

        if not self.per_obs and hasattr(log_prob, "sum"):
            log_prob = log_prob.sum()

        return log_prob


class NoDist(Dist):
    def __init__(self):
        super().__init__(NoDistribution)

    def all_input_nodes(self) -> frozenset[Node]:
        return frozenset()

    def all_output_nodes(self) -> frozenset[Node]:
        return frozenset()

    @property
    def outputs(self) -> frozenset[Node]:
        return frozenset()

    def update(self) -> NoDist:
        return self

    @property
    def value(self) -> float:
        return 0.0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Var:
    """A variable wrapping a value and a distribution node."""

    __slots__ = (
        "_dist_node",
        "_groups",
        "_name",
        "_observed",
        "_parameter",
        "_role",
        "_value_node",
        "_var_value_node",
        "info",
    )

    def __init__(
        self,
        value: Any,
        distribution: Dist | None = None,
        name: str = "",
    ):
        self._name = name
        self._value_node: Node = Data(None)
        self._dist_node: Dist = NoDist()

        self._var_value_node: VarValue = VarValue(
            self._value_node, _name=f"{self._name}_var_value"
        )

        self._var_value_node._set_var(self)

        # use setters
        self.value_node = value  # type: ignore  # unfrozen
        self.dist_node = distribution  # type: ignore  # unfrozen

        self._observed = False
        self._parameter = False
        self._role = ""

        self._groups: set[tuple[str, str]] = set()

        self.info: dict[str, Any] = {}
        """Additional meta-information about the variable as a dict."""

    def all_input_nodes(self) -> frozenset[Node]:
        """Returns all input *nodes* as a frozen set."""
        return self.value_node.all_input_nodes() | self._dist_node.all_input_nodes()

    def all_input_vars(self) -> frozenset[Var]:
        """
        Returns all input *variables* as a frozen set.

        The returned set also contains input variables that are indirect inputs
        of this variable through nodes without variables.
        """
        nodes = set(self.all_input_nodes())
        visited = []
        _vars = set()

        while nodes:
            node = nodes.pop()

            if node not in visited:
                if node.var and node.var is not self:
                    _vars.add(node.var)
                else:
                    nodes.update(node.all_input_nodes())

                visited.append(node)

        return frozenset(_vars)

    @in_model_method
    def all_output_nodes(self) -> frozenset[Node]:
        """Returns all output *nodes* as a frozen set."""
        nodes = set(self.value_node.all_output_nodes())
        nodes.update(self.var_value_node.all_output_nodes())
        nodes.update(self._dist_node.all_output_nodes())
        nodes.discard(self.var_value_node)
        return frozenset(nodes)

    @in_model_method
    def all_output_vars(self) -> frozenset[Var]:
        """
        Returns all output *variables* as a frozen set.

        The returned set also contains output variables that are indirect outputs
        of this variable through nodes without variables.
        """
        nodes = set(self.all_output_nodes())
        visited = []
        _vars = set()

        while nodes:
            node = nodes.pop()

            if node not in visited:
                if node.var and node.var is not self:
                    _vars.add(node.var)
                else:
                    nodes.update(node.all_output_nodes())

                visited.append(node)

        return frozenset(_vars)

    @property
    def dist_node(self) -> Dist | None:
        """The distribution node of the variable."""
        return self._dist_node if self.has_dist else None

    @dist_node.setter
    @no_model_setter
    def dist_node(self, dist_node: Dist | None):
        if not dist_node:
            dist_node = NoDist()

        if dist_node.model:
            raise RuntimeError(
                f"{repr(dist_node)} is part of a model, cannot be set as dist node"
            )

        if self.name and not dist_node.name:
            dist_node.name = f"{self.name}_log_prob"  # type: ignore  # unfrozen

        self._dist_node._unset_var()

        dist_node._set_var(self)
        dist_node.at = self.var_value_node  # type: ignore  # unfrozen
        self._dist_node = dist_node

    @property
    def groups(self) -> set[tuple[str, str]]:
        """
        The groups the variable belongs to.

        The groups are defined as a set of tuples, each of which consists of
        the group name and the key of the variable in the group, e.g.::

            {
                ("group1", "key_in_group1"),
                ("group2", "key_in_group2"),
            }
        """
        return self._groups

    @groups.setter
    def groups(self, groups: set[tuple[str, str]]):
        self._groups = groups

    @property
    def has_dist(self) -> bool:
        """Whether the variable has a probability distribution."""
        return not isinstance(self._dist_node, NoDist)

    @property
    def log_prob(self) -> Array:
        """
        The log-probability of the variable.

        A variable without a probability distribution has a log-probability of 0.0.
        """
        return self._dist_node.value

    @property
    def model(self) -> Model | None:
        """The model the variable is part of."""
        return self.value_node.model

    @property
    def name(self) -> str:
        """The name of the variable."""
        return self._name

    @name.setter
    @no_model_setter
    def name(self, name: str):
        if name and self.value_node.name in ("", f"{self.name}_value"):
            self.value_node.name = f"{name}_value"  # type: ignore  # unfrozen
            self.var_value_node.name = f"{name}_var_value"  # type: ignore  # unfrozen

        if name and self._dist_node.name in ("", f"{self.name}_log_prob"):
            self._dist_node.name = f"{name}_log_prob"  # type: ignore  # unfrozen

        self._name = name

    @property
    def nodes(self) -> list[Node]:
        """The nodes of the variable as a list."""
        nodes = [self.value_node, self.var_value_node]

        if self.dist_node:
            nodes.append(self.dist_node)

        return nodes

    @property
    def observed(self) -> bool:
        """Whether the variable is observed."""
        return self._observed

    @observed.setter
    @no_model_setter
    def observed(self, observed: bool):
        self._observed = observed

    @property
    def parameter(self) -> bool:
        """Whether the variable is a parameter."""
        return self._parameter

    @parameter.setter
    @no_model_setter
    def parameter(self, parameter: bool):
        self._parameter = parameter

    @property
    def role(self) -> str:
        """The role of the variable."""
        return self._role

    @role.setter
    def role(self, role: str):
        self._role = role

    @property
    def strong(self) -> bool:
        """Whether the variable is strong."""
        return isinstance(self.value_node, Data)

    def update(self) -> Var:
        """Updates the variable."""
        self.value_node.update()
        self._dist_node.update()
        return self

    @property
    def value(self) -> Any:
        """
        The value of the variable.

        Can only be set if the variable is strong. If the variable is part of
        a :class:`.Model` ``m`` with ``m.auto_update == True``, setting the value of
        the variable triggers an update of the model. The auto-update can be disabled
        to improve the performance if multiple model parameters are updated at once.
        """
        return self.value_node.value

    @value.setter
    def value(self, value: Any):
        if self.weak:
            raise RuntimeError(f"{repr(self)} is weak, cannot set value")

        self.value_node.value = value  # type: ignore  # data node

    @property
    def value_node(self) -> Node:
        """The value node of the variable."""
        return self._value_node

    @value_node.setter
    @no_model_setter
    def value_node(self, value_node: Any):
        if isinstance(value_node, Var):
            value_node = Calc(lambda x: x, value_node)

        if not isinstance(value_node, Node):
            value_node = Data(value_node)

        if value_node.model:
            raise RuntimeError(
                f"{repr(value_node)} is part of a model, cannot be set as value node"
            )

        if self.name and not value_node.name:
            value_node.name = f"{self.name}_value"

        self.value_node._unset_var()

        value_node._set_var(self)
        self._value_node = value_node
        self._var_value_node.set_inputs(self._value_node)

    @property
    def var_value_node(self) -> VarValue:
        """The proxy node for the value of the variable."""
        return self._var_value_node

    @property
    def weak(self) -> bool:
        """Whether the variable is weak."""
        return not self.strong

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{self.name}>"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variable helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def Obs(value: Any | Calc, distribution: Dist | None = None, name: str = "") -> Var:
    """Defines an observed variable."""
    var = Var(value, distribution, name)
    var.observed = True
    return var


def Param(value: Any | Calc, distribution: Dist | None = None, name: str = "") -> Var:
    """Defines a parameter variable."""
    var = Var(value, distribution, name)
    var.value_node.monitor = True
    var.parameter = True
    return var


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Other helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def add_group(name: str, **kwargs: Node | Var) -> None:
    """
    Assigns the nodes and variables to a group.

    See :attr:`liesel.model.nodes.Node.groups`.

    Parameters
    ----------
    name
        The name of the group.
    kwargs
        The nodes and variables in the group with their keys in the group
        as keywords.
    """

    for key, arg in kwargs.items():
        arg.groups.add((name, key))
