"""
Nodes and variables.
"""

from __future__ import annotations

import logging
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable
from functools import wraps
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, Union

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
    "Group",
    "InputGroup",
    "Node",
    "NodeState",
    "obs",
    "param",
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

T = TypeVar("T", bound=Hashable)

logger = logging.getLogger(__name__)


def _unique_tuple(*args: Iterable[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(chain(*args)))


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
    """
    A node of a computational graph that can cache its value.

    Liesel represents statistical models as directed acyclic graphs (DAGs) of
    random variables (see :class:`.Var`) and computational nodes. The graph of
    random variables is built on top of the computational graph. The nodes of the
    computational graph will typically express computations in JAX returning arrays
    or pytrees_, but in general, they can represent arbitrary operations in Python.

    Nodes can cache the result of the operations they represent, improving
    the efficiency of the graph. The cached values are part of the model state
    (see :attr:`.Model.state`), and can be stored in a chain by Liesel's MCMC engine
    Goose.

    .. note::
        This class is an abstract class that cannot be initialized without defining the
        :meth:`.update` method. See below for the most important concrete node classes.

    Parameters
    ----------
    inputs
        Non-keyword inputs. Any inputs that are not already nodes or :class:`.Var`
        will be converted to :class:`.Data` nodes.
    _name
        The name of the node. If you do not specify a name, a unique name will be \
        automatically generated upon initialization of a :class:`.Model`.
    _needs_seed
        Whether the node needs a seed / PRNG key.

    See Also
    --------
    .Calc :
        A node representing a general calculation/operation
        in JAX or Python.
    .Data :
        A node representing some static data.
    .Dist :
        A node representing a ``tensorflow_probability``
        :class:`~tfp.distributions.Distribution`.
    .Var : A variable in a statistical model, typically with a probability
        distribution.
    .param :
        A helper function to initialize a :class:`.Var` as a model parameter.
    .obs :
        A helper function to initialize a :class:`.Var` as an observed variable.


    .. _pytrees: https://jax.readthedocs.io/en/latest/pytrees.html
    .. _TensorFlow Probability: https://www.tensorflow.org/probability
    """

    def __init__(
        self,
        *inputs: Any,
        _name: str = "",
        _needs_seed: bool = False,
        **kwinputs: Any,
    ):
        self._groups: dict[str, Group] = {}
        self._inputs = tuple(self._to_node(_input) for _input in inputs)
        self._kwinputs = {kw: self._to_node(_input) for kw, _input in kwinputs.items()}
        self._model: weakref.ref[Model] | Callable[[], None] = lambda: None
        self._name = _name
        self._needs_seed = _needs_seed
        self._outdated = True
        self._outputs: tuple[Node, ...] = ()
        self._value: Any = None
        self._var: Var | None = None

        self.monitor = False
        """Whether the node should be monitored by an inference algorithm."""

    def _add_output(self, output: Node) -> Node:
        self._outputs = _unique_tuple(self._outputs, [output])
        return self

    def _clear_outputs(self) -> Node:
        self._outputs = ()
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

    @no_model_method
    def add_inputs(self, *inputs: Any, **kwinputs: Any) -> Node:
        """Adds non-keyword and keyword input nodes to the existing ones."""
        inputs = self.inputs + inputs
        kwinputs = self.kwinputs | kwinputs
        self.set_inputs(*inputs, **kwinputs)
        return self

    def all_input_nodes(self) -> tuple[Node, ...]:
        """Returns all non-keyword and keyword input nodes as a unique tuple."""
        return _unique_tuple(self.inputs, self.kwinputs.values())

    @in_model_method
    def all_output_nodes(self) -> tuple[Node, ...]:
        """Returns all output nodes as a unique tuple."""
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
    def groups(self) -> MappingProxyType[str, Group]:
        """The groups that this node is a part of."""
        return MappingProxyType(self._groups)

    @property
    def inputs(self) -> tuple[Node, ...]:
        """The non-keyword input nodes."""
        return self._inputs

    @property
    def kwinputs(self) -> MappingProxyType[str, Node]:
        """The keyword input nodes."""
        return MappingProxyType(self._kwinputs)

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
    def outputs(self) -> tuple[Node, ...]:
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

    @property
    def iloc(self) -> tuple[Node | Var, ...]:
        input_list: list[Node | Var] = []
        for input_ in self.all_input_nodes():
            if isinstance(input_, VarValue):
                if input_.var is None:
                    raise RuntimeError(f"{input_}.var is None.")
                input_list.append(input_.var)
            else:
                input_list.append(input_)

        return tuple(input_list)

    @property
    def loc(self) -> dict[str, Node | Var]:
        input_dict: dict[str, Node | Var] = {}
        for key, input_ in self.kwinputs.items():
            if isinstance(input_, VarValue):
                if input_.var is None:
                    raise RuntimeError(f"{input_}.var is None.")
                input_dict[key] = input_.var
            else:
                input_dict[key] = input_

        return input_dict

    def __getitem__(self, key: int | str) -> Node | Var:
        if isinstance(key, int):
            return self.iloc[key]
        elif isinstance(key, str):
            return self.loc[key]
        else:
            raise ValueError(f"Key must be str or int, not {type(key)}.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = self._model()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self._model is not None:
            self._model = weakref.ref(self._model)
        else:
            self._model = lambda: None

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name="{self.name}")'


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
    """
    A :class:`.Node` subclass that holds constant values.

    Since the value represented by a data node does not change, it is always up-to-date.
    A common usecase for data nodes is to cache computed values.

    - By default, data nodes *will* appear in the node graph created by
      :func:`.viz.plot_nodes`, but they will *not* appear in the model graph created by
      :func:`.viz.plot_vars`.
    - You can wrap a data node in a :class:`.Var` to make it appear in the model graph.

    Parameters
    ----------
    value
        The value of the data node.
    _name
        The name of the data node. If you do not specify a name, a unique name will be \
        automatically generated upon initialization of a :class:`.Model`.

    See Also
    --------
    .Calc :
        A node representing a general calculation/operation
        in JAX or Python.
    .Dist :
        A node representing a ``tensorflow_probability``
        :class:`~tfp.distributions.Distribution`.
    .Var : A variable in a statistical model, typically with a probability
        distribution.
    .param :
        A helper function to initialize a :class:`.Var` as a model parameter.
    .obs :
        A helper function to initialize a :class:`.Var` as an observed variable.


    Examples
    --------

    A simple data node representing a constant value without a name:

    >>> nameless_node = lsl.Data(1.0)
    >>> nameless_node
    Data(name="")

    Adding this node to a model leads to an automatically generated name:

    >>> model = lsl.GraphBuilder().add(nameless_node).build_model()
    >>> nameless_node
    Data(name="n0")

    A data node with a name:

    >>> node = lsl.Data(1.0, _name="my_name")
    >>> node
    Data(name="my_name")
    """

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
    """
    A :class:`.Node` subclass that calculates its value based on its inputs nodes.

    Calculator nodes are a central element block of the Liesel graph building toolkit.
    They wrap arbitrary calculations in pure JAX functions.

    - By default, calculator nodes *will* appear in the node graph created by
      :func:`.viz.plot_nodes`, but they will *not* appear in the model graph created by
      :func:`.viz.plot_vars`.
    - You can wrap a calculator node in a :class:`.Var` to make it appear in the model
      graph.

    .. tip::
        The wrapped function must be jit-compilable by JAX. This mainly means that
        it must be a pure function, i.e. it must not have any side effects and, given
        the same input, it must always return the same output. Some special
        consideration is also required for loops and conditionals.

        Please consult the JAX docs_ for details.

    Parameters
    ----------
    function
        The function to be wrapped. Must be jit-compilable by JAX.
    *inputs
        Non-keyword inputs. Any inputs that are not already nodes or :class:`.Var`
        will be converted to :class:`.Data` nodes. The values of these inputs will be
        passed to the wrapped function in the same order they are entered here.
    _name
        The name of the node. If you do not specify a name, a unique name will be \
        automatically generated upon initialization of a :class:`.Model`.
    _needs_seed
        Whether the node needs a seed / PRNG key.
    update_on_init
        If ``True``, the calculator will try to evaluate its function upon \
        initialization.
    **kwinputs
        Keyword inputs. Any inputs that are not already nodes or :class:`.Var`s
        will be converted to :class:`.Data` nodes. The values of these inputs will be
        passed to the wrapped function as keyword arguments.

    See Also
    --------
    .Data :
        A node representing some static data.
    .Dist :
        A node representing a ``tensorflow_probability``
        :class:`~tfp.distributions.Distribution`.
    .Var : A variable in a statistical model, typically with a probability
        distribution.
    .param :
        A helper function to initialize a :class:`.Var` as a model parameter.
    .obs :
        A helper function to initialize a :class:`.Var` as an observed variable.

    Notes
    -----

    A calculator node will compute its value only when :meth:`.Calc.update` is called.
    This does not happen automatically upon initialization. Commonly, the first time
    this method is called is during the initialization of a :class:`.Model`, which might
    make it hard to spot errors in the wrapped computations. To update the value
    immediately, you can call :meth:`.Calc.update` manually.

    Examples
    --------

    A simple calculator node, taking the exponential value of an input parameter.

    >>> log_scale = lsl.param(0.0, name="log_scale")
    >>> scale = lsl.Calc(jnp.exp, log_scale)
    >>> print(scale.value)
    1.0

    The value of the calculator node is updated when :meth:`.Calc.update` is called.

    >>> scale.update()
    Calc(name="")
    >>> print(scale.value)
    1.0

    You can also update the value of the calculator node in one step upon initilization.

    >>> log_scale = lsl.param(0.0, name="log_scale")
    >>> scale = lsl.Calc(jnp.exp, log_scale).update()
    >>> print(scale.value)
    1.0

    You can also use your own functions as long as they are jit-compilable by JAX.

    >>> def compute_variance(x):
    ...     return jnp.exp(x)**2
    >>> log_scale = lsl.param(0.0, name="log_scale")
    >>> variance = lsl.Calc(compute_variance, log_scale).update()
    >>> print(variance.value)
    1.0

    You can wrap a calculator node in a :class:`.Var` to declare its role as a
    statistical model variable and make it appear in the variable graph.

    >>> log_scale = lsl.param(0.0, name="log_scale")
    >>> scale = lsl.Var(lsl.Calc(jnp.exp, log_scale).update())
    >>> print(scale.value)
    1.0


    .. _docs: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

    """

    def __init__(
        self,
        function: Callable[..., Any],
        *inputs: Any,
        _name: str = "",
        _needs_seed: bool = False,
        update_on_init: bool = True,
        **kwinputs: Any,
    ):
        super().__init__(*inputs, **kwinputs, _name=_name, _needs_seed=_needs_seed)
        self._function = function

        if update_on_init:
            try:
                self.update()
            except Exception as e:
                logger.warning(
                    f"{self} was not updated during initialization, because the"
                    f" following exception occured: {repr(e)}. See debug log for the"
                    " full traceback."
                )
                logger.debug(
                    f"{self} was not updated during initialization, because the"
                    " following exception occured:",
                    exc_info=e,
                )

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
        try:
            self._value = self.function(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error while updating {self}.") from e
        self._outdated = False
        return self


class TransientCalc(TransientNode, Calc):
    """A transient calculator node that does not cache its value."""

    @property
    def value(self) -> Any:
        args = [_input.value for _input in self.inputs]
        kwargs = {kw: _input.value for kw, _input in self.kwinputs.items()}
        try:
            value = self.function(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error while updating {self}.") from e

        return value


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
    """
    A :class:`.Node` subclass that wraps a probability distribution.

    Distribution nodes wrap distribution classes that follow the
    ``tensorflow_probability`` :class:`~tfp.distributions.Distribution` interface.
    They can be used to represent observation models and priors.

    Distribution nodes *will* appear in the node graph created by
    :func:`.viz.plot_nodes`, but they will *not* appear in the model graph created by
    :func:`.viz.plot_vars`.

    Parameters
    ----------
    distribution
        The wrapped distribution class that follows the ``tensorflow_probability``
        :class:`~tfp.distributions.Distribution` interface.
    *inputs
        Non-keyword inputs. Any inputs that are not already nodes or :class:`.Var`
        will be converted to :class:`.Data` nodes. The values of these inputs will be
        passed to the wrapped distribution in the same order they are entered here.
    _name
        The name of the node. If you do not specify a name, a unique name will be \
        automatically generated upon initialization of a :class:`.Model`.
    _needs_seed
        Whether the node needs a seed / PRNG key.
    **kwinputs
        Keyword inputs. Any inputs that are not already nodes or :class:`.Var`s
        will be converted to :class:`.Data` nodes. The values of these inputs will be
        passed to the wrapped distribution as keyword arguments.

    See Also
    --------
    .Var : A variable in a statistical model, typically with a probability
        distribution.
    .param :
        A helper function to initialize a :class:`.Var` as a model parameter.
    .obs :
        A helper function to initialize a :class:`.Var` as an observed variable.
    .MultivariateNormalDegenerate : A custom distribution class that implements
        a degenerate multivariate normal distribution in the ``tensorflow_probability``
        :class:`~tfp.distributions.Distribution` interface.


    Examples
    --------

    For the examples below, we import ``tensorflow_probability``:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    Creating an observation model for a normally distributed variable with fixed
    mean and scale. The log probability of the node ``y`` in the example below is
    ``None``, until the variable is updated.


    >>> dist = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
    >>> y = lsl.obs(jnp.array([-0.5, 0.0, 0.5]), dist, name="y")
    >>> print(y.log_prob)
    None

    >>> y.update()
    Var(name="y")
    >>> y.log_prob
    Array([-1.0439385, -0.9189385, -1.0439385], dtype=float32)

    Now we define the same observation model, but include the location and scale
    as parameters:

    >>> loc = lsl.param(0.0, name="loc")
    >>> scale = lsl.param(1.0, name="scale")
    >>> dist = lsl.Dist(tfd.Normal, loc=loc, scale=scale)
    >>> y = lsl.obs(jnp.array([-0.5, 0.0, 0.5]), dist, name="y").update()
    >>> y.log_prob
    Array([-1.0439385, -0.9189385, -1.0439385], dtype=float32)

    .. rubric:: Summed-up log-probability

    You can set the ``per_obs`` attribute of a distribution node to ``False`` to
    sum up the log-probability of the distribution over all observations.

    >>> dist.per_obs = False
    >>> y.update().log_prob
    Array(-3.0068154, dtype=float32)

    """

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

    def all_input_nodes(self) -> tuple[Node, ...]:
        inputs = super().all_input_nodes()

        if self.at:
            inputs = _unique_tuple(inputs, [self.at])

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

    def all_input_nodes(self) -> tuple[Node, ...]:
        return ()

    def all_output_nodes(self) -> tuple[Node, ...]:
        return ()

    @property
    def outputs(self) -> tuple[Node, ...]:
        return ()

    def update(self) -> NoDist:
        return self

    @property
    def value(self) -> float:
        return 0.0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Var:
    """
    A variable in a statistical model, typically with a probability distribution.

    A variable in Liesel is typically a random variable, e.g. an observed or
    latent variable with a probability distribution, or a model parameter with
    a prior distribution. Note that observed variables and model parameters should
    typically be declared with the :func:`.obs` and :func:`.param` helper functions.
    Other quantities can also be declared as variables, e.g. fixed data like
    hyperparameters or design matrices, or quantities that are computed from
    other nodes, e.g. structured additive predictors in semi-parametric
    regression models.

    If a :class:`.Data` or :class:`.Calc` node does not have an associated probability
    distribution, it is possible but not necessary to declare it as a variable. There
    is no hard and fast rule when a node without a probability distribution should be
    declared as a variable and when not. The advantages of a variable in this case are:
    (1) easier access via the :attr:`.Model.vars` attribute, and (2) more explicit
    visualization with the :func:`.plot_vars` function. This might be particularly
    desirable for the hyperparameters of a prior distribution.

    Parameters
    ----------
    value
        The value of the variable.
    distribution
        The probability distribution of the variable.
    name
        The name of the variable. If you do not specify a name, a unique name will be \
        automatically generated upon initialization of a :class:`.Model`.

    See Also
    --------
    .obs : Helper function to declare a variable as an observed quantity.
    .param : Helper function to declare a variable as a model parameter.
    .Calc :
        A node representing a general calculation/operation
        in JAX or Python.
    .Data :
        A node representing some static data.
    .Dist :
        A node representing a ``tensorflow_probability``
        :class:`~tfp.distributions.Distribution`.

    Examples
    --------

    A simple variable without a distribution and without a name:

    >>> x = lsl.Var(1.0)
    >>> x
    Var(name="")

    Adding this variable to a model leads to an automatically generated name:

    >>> model = lsl.GraphBuilder().add(x).build_model()
    >>> x
    Var(name="v0")

    A simple variable with a name:

    >>> x = lsl.Var(1.0, name="x")
    >>> x
    Var(name="x")

    """

    __slots__ = (
        "info",
        "_auto_transform",
        "_dist_node",
        "_groups",
        "_name",
        "_observed",
        "_parameter",
        "_role",
        "_value_node",
        "_var_value_node",
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

        self._auto_transform = False
        self._groups: dict[str, Group] = {}
        self._observed = False
        self._parameter = False
        self._role = ""

        self.info: dict[str, Any] = {}
        """Additional meta-information about the variable as a dict."""

    def all_input_nodes(self) -> tuple[Node, ...]:
        """Returns all input *nodes* as a unique tuple."""
        inputs1 = self.value_node.all_input_nodes()
        inputs2 = self._dist_node.all_input_nodes()
        return _unique_tuple(inputs1, inputs2)

    def all_input_vars(self) -> tuple[Var, ...]:
        """
        Returns all input *variables* as a unique tuple.

        The returned tuple also contains input variables that are indirect inputs
        of this variable through nodes without variables.
        """
        nodes = list(self.all_input_nodes())
        visited = []
        _vars = []

        while nodes:
            node = nodes.pop()

            if node not in visited:
                if node.var and node.var is not self:
                    _vars.append(node.var)
                else:
                    nodes.extend(node.all_input_nodes())

                visited.append(node)

        return _unique_tuple(_vars)

    @in_model_method
    def all_output_nodes(self) -> tuple[Node, ...]:
        """Returns all output *nodes* as a unique tuple."""
        nodes = list(self.value_node.all_output_nodes())
        nodes.extend(self.var_value_node.all_output_nodes())
        nodes.extend(self._dist_node.all_output_nodes())
        nodes = [node for node in nodes if node is not self.var_value_node]
        return _unique_tuple(nodes)

    @in_model_method
    def all_output_vars(self) -> tuple[Var, ...]:
        """
        Returns all output *variables* as a unique tuple.

        The returned tuple also contains output variables that are indirect outputs
        of this variable through nodes without variables.
        """
        nodes = list(self.all_output_nodes())
        visited = []
        _vars = []

        while nodes:
            node = nodes.pop()

            if node not in visited:
                if node.var and node.var is not self:
                    _vars.append(node.var)
                else:
                    nodes.extend(node.all_output_nodes())

                visited.append(node)

        return _unique_tuple(_vars)

    @property
    def auto_transform(self) -> bool:
        """
        Whether the variable should automatically be transformed to the unconstrained
        space ``R**n`` upon model initialization.
        """
        return self._auto_transform

    @auto_transform.setter
    def auto_transform(self, auto_transform: bool):
        self._auto_transform = auto_transform

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
    def groups(self) -> MappingProxyType[str, Group]:
        """The groups that this variable is a part of."""
        return MappingProxyType(self._groups)

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
        """
        Whether the variable is observed.

        If a variable is observed and has an associated
        probability distribution, its log-probability is automatically added to the
        model log-likelihood (see :attr:`.Model.log_lik`).

        See Also
        --------
        .obs : Helper function to declare a variable as a parameter.
        .Model.log_prior : The log-prior of a Liesel model.
        .Var.parameter : Whether the variable is a parameter. If a variable is \
            a parameter, it is not observed.

        Notes
        -----

        We recommend to use the :func:`.obs` helper function to declare an observed
        variable.
        """
        return self._observed

    @observed.setter
    @no_model_setter
    def observed(self, observed: bool):
        self._observed = observed

    @property
    def parameter(self) -> bool:
        """
        Whether the variable is a parameter.

        If the variable is a parameter and has an associated
        probability distribution, its log-probability is added to the
        model's :attr:`~.Model.log_prior`.

        See Also
        --------
        .param : Helper function to declare a variable as a parameter.
        .Model.log_prior : The log-prior of a Liesel model.
        .Var.observed : Whether the variable is observed. If a variable is \
            a parameter, it is not observed.

        Notes
        -----

        We recommend to use the :func:`.param` helper function to declare a variable
        as a parameter.

        """
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
        """
        Whether the variable is strong.

        A strong node is a node whose value is defined outside of the model, for
        example, if the node represents some observed data or a parameter (parameters
        are usually set by an inference algorithm such as an optimizer or sampler). In
        contrast, a weak node is a node whose value is defined within the model, that
        is, it is a deterministic function of some other nodes. An exp-transformation
        mapping a real-valued parameter to a positive number, for example, would be a
        weak node.

        See Also
        --------
        .weak : Whether the variable is weak. In general, ``strong = not weak``.

        """
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
        """
        Whether the variable is weak.

        A weak variable is a variable whose value is defined within the model, that
        is, it is a deterministic function of some other nodes.

        See Also
        --------
        .strong : Whether the variable is strong. In general, ``weak = not strong``.
        """
        return not self.strong

    @property
    def iloc(self) -> tuple[Var | Node, ...]:
        return self.value_node.iloc

    @property
    def loc(self) -> dict[str, Var | Node]:
        return self.value_node.loc

    def __getitem__(self, key: int | str) -> Var | Node:
        return self.value_node[key]

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name="{self.name}")'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variable helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def obs(value: Any | Calc, distribution: Dist | None = None, name: str = "") -> Var:
    """
    Helper function that returns an observed :class:`.Var`.

    Sets the :attr:`.Var.observed` flag. If the observed variable is a random variable,
    i.e. if it has an associated probability distribution, its log-probability is
    automatically added to the model log-likelihood (see :attr:`.Model.log_lik`).

    Parameters
    ----------
    value
        The value of the variable.
    distribution
        The probability distribution of the variable.
    name
        The name of the variable. If you do not specify a name, a unique name will be \
        automatically generated upon initialization of a :class:`.Model`.


    Returns
    -------
    An observed variable.

    See Also
    --------
    .Calc :
        A node representing a general calculation/operation in JAX or Python.
    .Data :
        A node representing some static data.
    .Dist :
        A node representing a ``tensorflow_probability``
        :class:`~tfp.distributions.Distribution`.
    .Var : A variable in a statistical model, typically with a probability
        distribution.
    .param :
        A helper function to initialize a :class:`.Var` as a model parameter.

    Notes
    -----

    A variable will compute its log probability only when :meth:`.Calc.update` is
    called. This does not happen automatically upon initialization. Commonly, the first
    time this method is called is during the initialization of a :class:`.Model`. To
    update the value immediately, you can call :meth:`.Var.update` manually.


    Examples
    --------

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    We can declare an observed variable with a normal distribution as the observation
    model:

    >>> dist = lsl.Dist(tfd.Normal, loc=0.0, scale=1.0)
    >>> y = lsl.obs(jnp.array([-0.5, 0.0, 0.5]), dist, name="y")
    >>> y
    Var(name="y")

    Now we build the model graph:

    >>> model = lsl.GraphBuilder().add(y).build_model()

    The log-likelihood of the model is the sum of the log-probabilities of all observed
    variables. In this case this is only our ``y`` variable:

    >>> model.log_lik
    Array(-3.0068154, dtype=float32)

    >>> jnp.sum(y.log_prob)
    Array(-3.0068154, dtype=float32)

    """
    var = Var(value, distribution, name)
    var.observed = True
    return var


def param(value: Any | Calc, distribution: Dist | None = None, name: str = "") -> Var:
    """
    Helper function that returns a parameter :class:`.Var`.

    Sets the :attr:`.Var.parameter` flag. If the parameter variable is a
    random variable, i.e. if it has an associated probability distribution,
    its log-probability is automatically added to the model log-prior
    (see :attr:`.Model.log_prior`).

    Parameters
    ----------
    value
        The value of the parameter.
    distribution
        The probability distribution of the parameter.
    name
        The name of the parameter. If you do not specify a name, a unique name will be \
        automatically generated upon initialization of a :class:`.Model`.

    Returns
    -------
    A parameter variable.

    See Also
    --------
    .Calc :
        A node representing a general calculation/operation
        in JAX or Python.
    .Data :
        A node representing some static data.
    .Dist :
        A node representing a ``tensorflow_probability``
        :class:`~tfp.distributions.Distribution`.
    .Var : A variable in a statistical model, typically with a probability
        distribution.
    .obs :
        A helper function to initialize a :class:`.Var` as an observed variable.

    Notes
    -----

    A variable will compute its log probability only when :meth:`.Calc.update` is
    called. This does not happen automatically upon initialization. Commonly, the first
    time this method is called is during the initialization of a :class:`.Model`. To
    update the value immediately, you can call :meth:`.Var.update` manually.

    Examples
    --------

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    A variance parameter with an inverse-gamma prior:

    >>> prior = lsl.Dist(tfd.InverseGamma, concentration=0.1, scale=0.1)
    >>> variance = lsl.param(1.0, prior, name="variance")
    >>> variance
    Var(name="variance")

    We can use this parameter variable in the distribution of an observed variable:

    >>> scale = lsl.Calc(jnp.sqrt, variance)
    >>> dist = lsl.Dist(tfd.Normal, loc=0.0, scale=scale)
    >>> y = lsl.obs(jnp.array([-0.5, 0.0, 0.5]), dist, name="y")
    >>> y
    Var(name="y")

    Now we can build the model graph:

    >>> model = lsl.GraphBuilder().add(y).build_model()

    The log_prior of the model is the sum of the log-priors of all parameters. In this
    case this is only our ``variance`` parameter:

    >>> model.log_prior
    Array(-2.582971, dtype=float32)

    >>> variance.log_prob
    Array(-2.582971, dtype=float32)

    The log-likelihood of the model is the sum of the log-probabilities of all observed
    variables. In this case this is only our ``y`` variable:

    >>> model.log_lik
    Array(-3.0068154, dtype=float32)

    >>> jnp.sum(y.log_prob)
    Array(-3.0068154, dtype=float32)

    """
    var = Var(value, distribution, name)
    var.value_node.monitor = True
    var.parameter = True
    return var


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Other helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def add_group(name: str, **kwargs: Node | Var) -> Group:
    """
    Assigns the nodes and variables to a group.

    .. deprecated:: 0.2.2
        Use the :class:`.Group` object directly. This function will be removed in
        v0.4.0.

    Parameters
    ----------
    name
        The name of the group.
    kwargs
        The nodes and variables in the group with their keys in the group as keywords.

    See Also
    --------
    .Node.groups : The groups that a node is a part of.
    .Group : A group of nodes and variables.
    """
    warnings.warn(
        "The `add_group` function is deprecated and will be removed in v0.4.0. "
        "Use the `Group` object directly.",
        FutureWarning,
    )
    return Group(name, **kwargs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Group ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Group:
    """
    A group holds a collection of related :class:`.Var` and/or :class:`.Node` objects.

    They allow you to do three basic things:

    1. Store related nodes together for easier access.
    2. Access their member nodes and variables via ``group["name"]``, where ``"name"``
       is the group-specific name, which can be different from the :attr:`.Var.name` /
       :attr:`.Node.name`.
    3. Easily retrieve a variable's or a node's value from a :attr:`.Model.state` based
       on their group-specific name via :meth:`.value_from`.

    Parameters
    ----------
    name
        The group's name. Must be unique among the groups of its members, and within
        a model.
    **nodes_and_vars
        An arbitrary number of nodes or variables. The keywords will be used as the
        group-specific names of the respective objects.

    See Also
    --------
    * :attr:`.Node.groups` and :attr:`.Var.groups` are :obj:`MappingProxyType` objects
      (basically read-only dictionaries) of the groups whose member the respective
      object is.
    * :meth:`.GraphBuilder.groups` and :meth:`.Model.groups` are methods that collect
      and return all groups within the graph/model.

    Notes
    -----
    Note the following:

    - Groups can only be filled upon initialization.
    - After initialization, variables and nodes cannot be removed from a group.

    Examples
    --------
    Add a variable to a group:

    >>> my_var = lsl.Var(0.0, name="long_unique_variable_name")
    >>> grp = lsl.Group(name="demo_group", short_name=my_var)
    >>> grp
    Group(name="demo_group")

    Access the variable by its group-specific name:

    >>> grp["short_name"]
    Var(name="long_unique_variable_name")

    Retrieve the value of a variable from a model state:

    >>> model_state = {my_var.value_node.name: lsl.NodeState(10., False)}
    >>> grp.value_from(model_state, "short_name")
    10.0
    """

    def __init__(self, name: str, **nodes_and_vars: Node | Var) -> None:
        self._name = name
        self._nodes_and_vars = nodes_and_vars

        for member in self._nodes_and_vars.values():
            if name in member.groups:
                raise RuntimeError(
                    f"{repr(member)} is already a member of a group "
                    f"with the name {repr(name)}"
                )
            member._groups[name] = self

        self._nodes = {
            name: obj
            for name, obj in self._nodes_and_vars.items()
            if isinstance(obj, Node)
        }

        self._vars = {
            name: obj
            for name, obj in self._nodes_and_vars.items()
            if isinstance(obj, Var)
        }

    @property
    def name(self) -> str:
        """The group's name."""
        return self._name

    def value_from(self, model_state: dict[str, NodeState], name: str) -> Array:
        """
        Retrieves the value of a node or variable that is a member of the group from
        a model state.

        Parameters
        ----------
        model_state
            The state of a Liesel model, i.e. a :class:`~.Model.state`.
        name
            The name of the node or variable within this group.

        Returns
        -------
        The value of the node or variable.
        """
        member = self[name]

        if isinstance(member, Var):
            value_name = member.value_node.name
        else:
            value_name = member.name

        return model_state[value_name].value

    @property
    def vars(self) -> MappingProxyType[str, Var]:
        """A mapping of the variables in the group with their names as keys."""
        return MappingProxyType(self._vars)

    @property
    def nodes(self) -> MappingProxyType[str, Node]:
        """A mapping of the nodes in the group with their names as keys."""
        return MappingProxyType(self._nodes)

    @property
    def nodes_and_vars(self) -> MappingProxyType[str, Node | Var]:
        """A mapping of all group members with their names as keys."""
        return MappingProxyType(self._nodes_and_vars)

    def __contains__(self, key) -> bool:
        return key in self._nodes_and_vars

    def __getitem__(self, key) -> Var | Node:
        return self._nodes_and_vars[key]

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name="{self.name}")'
