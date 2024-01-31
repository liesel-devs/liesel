"""
Model interfaces.
"""

import copy
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..docs import usedocs
from .types import ModelInterface, ModelState, Position

if TYPE_CHECKING:
    from ..model.model import Model

LogProbFunction = Callable[[ModelState], float]


@usedocs(ModelInterface)
class DictInterface:
    """
    A model interface for a model state represented by a ``dict[str, Array]`` and a
    corresponding log-probability function.

    Parameters
    ----------
    log_prob_fn
        A function that takes a model state and returns the log-probability. The
        model state is expected to be a ``dict[str, Array]``.

    See Also
    --------
    .DataclassInterface : A model interface for a model state represented by a
        :obj:`~dataclasses.dataclass` and a corresponding log-probability function.
    .LieselInterface : A model interface for a Liesel :class:`.Model`.

    Examples
    --------

    For this example, we import ``tensorflow_probability`` as follows:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    Now we define a very simple log_prob_fn for the sake of demonstration:

    >>> def log_prob_fn(model_state):
    ...     loc = model_state["loc"]
    ...     scale = model_state["scale"]
    ...     x = model_state["x"]
    ...     return tfd.Normal(loc, scale).log_prob(x)

    We initialize the interface by passing the log_prob_fn to the constructor:

    >>> interface = gs.DictInterface(log_prob_fn)

    We evaluate the log-probability of a model state by calling the log_prob method:

    >>> state = {"x": jnp.array(0.0), "loc": jnp.array(0.0), "scale": jnp.array(1.0)}
    >>> interface.log_prob(state)
    Array(-0.9189385, dtype=float32)

    We update the model state by passing a position to the update_state method:

    >>> position = {"x": jnp.array(1.0)}
    >>> state = interface.update_state(position, state)

    We can now evaluate the log-probability of the updated model state:

    >>> interface.log_prob(state)
    Array(-1.4189385, dtype=float32)
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        self._log_prob_fn = log_prob_fn

    def extract_position(
        self, position_keys: Sequence[str], model_state: ModelState
    ) -> Position:
        """
        Extracts a position from a model state.

        Parameters
        ----------
        position_keys
            An iterable of variable or node names.
        model_state
            A dictionary of variable or node names and values.
        """
        return Position({key: model_state[key] for key in position_keys})

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        """
        Updates and returns a model state given a position.

        Parameters
        ----------
        position
            A dictionary of variable or node names and values.
        model_state
            An dictionary of variable or node names and values.
        """
        return model_state | position

    def log_prob(self, model_state: ModelState) -> float:
        """
        Returns the log-probability from a model state.

        Parameters
        ----------
        model_state
            A dictionary of variable or node names and values.
        """
        return self._log_prob_fn(model_state)


@usedocs(ModelInterface)
class DataclassInterface:
    """
    A model interface for a model state represented by a :obj:`~dataclasses.dataclass`
    and a corresponding log-probability function.

    Parameters
    ----------
    log_prob_fn
        A function that takes a model state and returns the log-probability. The
        model state is expected to be a :obj:`~dataclasses.dataclass`.

    See Also
    --------
    .DictInterface : A model interface for a model state represented by a
        ``dict[str, Array]`` and a corresponding log-probability function.
    .LieselInterface : A model interface for a Liesel :class:`.Model`.

    Examples
    --------

    For this example, we import ``tensorflow_probability`` as follows:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    We define a dataclass representing the model state:

    >>> from dataclasses import dataclass
    ...
    >>> @dataclass
    ... class State:
    ...     x: jnp.ndarray
    ...     loc: jnp.ndarray
    ...     scale: jnp.ndarray

    Now we define a very simple log_prob_fn for the sake of demonstration:

    >>> def log_prob_fn(model_state):
    ...     loc = model_state.loc
    ...     scale = model_state.scale
    ...     x = model_state.x
    ...     return tfd.Normal(loc, scale).log_prob(x)

    We initialize the interface by passing the log_prob_fn to the constructor:

    >>> interface = gs.DataclassInterface(log_prob_fn)

    We evaluate the log-probability of a model state by calling the log_prob method:

    >>> state = State(x=jnp.array(0.0), loc=jnp.array(0.0), scale=jnp.array(1.0))
    >>> interface.log_prob(state)
    Array(-0.9189385, dtype=float32)

    We update the model state by passing a position to the update_state method:

    >>> position = {"x": jnp.array(1.0)}
    >>> state = interface.update_state(position, state)

    We can now evaluate the log-probability of the updated model state:

    >>> interface.log_prob(state)
    Array(-1.4189385, dtype=float32)
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        self._log_prob_fn = log_prob_fn

    def extract_position(
        self, position_keys: Sequence[str], model_state: ModelState
    ) -> Position:
        """
        Extracts a position from a model state.

        Parameters
        ----------
        position_keys
            An iterable of variable or node names.
        model_state
            An instance of the dataclass representing the model state.
        """
        return Position({key: getattr(model_state, key) for key in position_keys})

    def log_prob(self, model_state: ModelState) -> float:
        """
        Returns the log-probability from a model state.

        Parameters
        ----------
        model_state
            An instance of the dataclass representing the model state.
        """
        return self._log_prob_fn(model_state)

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        """
        Updates and returns a model state given a position.

        Parameters
        ----------
        position
            A dictionary of variable or node names and values.
        model_state
            An instance of the dataclass representing the model state.
        """
        new_state = copy.copy(model_state)  # don't change the input
        for key, value in position.items():
            if hasattr(new_state, key):
                setattr(new_state, key, value)
            else:
                raise RuntimeError(
                    f"ModelState {model_state!r} does not have field with name {key}"
                )
        return new_state


class LieselInterface:
    """
    A :class:`.ModelInterface` for a Liesel :class:`.Model`.

    Parameters
    ----------
    model
        A Liesel :class:`.Model`.

    See Also
    --------
    .GraphBuilder : The graph builder class, used to set up a :class:`.Model`.

    See Also
    --------
    .DictInterface : A model interface for a model state represented by a
        ``dict[str, Array]`` and a corresponding log-probability function.
    .DataclassInterface : A model interface for a model state represented by a
        :obj:`~dataclasses.dataclass` and a corresponding log-probability function.
    .LieselInterface : A model interface for a Liesel :class:`.Model`.

    Examples
    --------

    First, we initialize a Liesel model. This is a minimal example only for the purpose
    of demonstrating how to use the interface.

    >>> y = lsl.obs(jnp.array([1.0, 2.0, 3.0]), name="y")
    >>> model = lsl.GraphBuilder().add(y).build_model()

    The interface is initialized by passing the model to the constructor.

    >>> interface = gs.LieselInterface(model)

    The interface instance can now be used in :meth:`~.goose.EngineBuilder.set_model`.
    """

    def __init__(self, model: "Model"):
        self._model = model._copy_computational_model()

    def extract_position(
        self, position_keys: Sequence[str], model_state: ModelState
    ) -> Position:
        """
        Extracts a position from a model state.

        Parameters
        ----------
        position_keys
            An iterable of variable or node names.
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.
        """
        position = {}

        for key in position_keys:
            try:
                position[key] = model_state[key].value
            except KeyError:
                node_key = self._model.vars[key].value_node.name
                position[key] = model_state[node_key].value

        return Position(position)

    def update_state(self, position: Position, model_state: ModelState) -> ModelState:
        """
        Updates and returns a model state given a position.

        Parameters
        ----------
        position
            A dictionary of variable or node names and values.
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.

        Warnings
        --------
        The ``model_state`` must be up-to-date, i.e. it must *not* contain any outdated
        nodes. Updates can only be triggered through new variable or node values in the
        ``position``. If you supply a ``model_state`` with outdated nodes, these nodes
        and their outputs will not be updated.
        """

        # sets all outdated flags in the model state to false
        # this is required to make the function jittable

        self._model.state = model_state

        for node in self._model.nodes.values():
            node._outdated = False

        for key, value in position.items():
            try:
                self._model.nodes[key].value = value  # type: ignore  # data node
            except KeyError:
                self._model.vars[key].value = value

        self._model.update()
        return self._model.state

    def log_prob(self, model_state: ModelState) -> float:
        """
        Returns the log-probability from a model state.

        Parameters
        ----------
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.
        """
        return model_state["_model_log_prob"].value


class NamedTupleInterface:
    """
    A model interface for a model state represented by a :obj:`~typing.NamedTuple`
    and a corresponding log-probability function.

    Parameters
    ----------
    log_prob_fn
        A function that takes a model state and returns the log-probability. The
        model state is expected to be a :obj:`~typing.NamedTuple`.

    See Also
    --------
    .DictInterface : A model interface for a model state represented by a
        ``dict[str, Array]`` and a corresponding log-probability function.
    .DataclassInterface : A model interface for a model state represented by a
        :obj:`~dataclasses.dataclass` and a corresponding log-probability function.
    .LieselInterface : A model interface for a Liesel :class:`.Model`.

    Examples
    --------

    For this example, we import ``tensorflow_probability`` as follows:

    >>> import tensorflow_probability.substrates.jax.distributions as tfd

    We define a subclass of NamedTuple representing the model state:

    >>> from typing import NamedTuple
    ...
    >>> class State(NamedTuple):
    ...     x: jnp.ndarray
    ...     loc: jnp.ndarray
    ...     scale: jnp.ndarray

    Now we define a very simple log_prob_fn for the sake of demonstration:

    >>> def log_prob_fn(model_state):
    ...     loc = model_state.loc
    ...     scale = model_state.scale
    ...     x = model_state.x
    ...     return tfd.Normal(loc, scale).log_prob(x)

    We initialize the interface by passing the log_prob_fn to the constructor:

    >>> interface = gs.NamedTupleInterface(log_prob_fn)

    We evaluate the log-probability of a model state by calling the log_prob method:

    >>> state = State(x=jnp.array(0.0), loc=jnp.array(0.0), scale=jnp.array(1.0))
    >>> interface.log_prob(state)
    Array(-0.9189385, dtype=float32)

    We update the model state by passing a position to the update_state method:

    >>> position = {"x": jnp.array(1.0)}
    >>> state = interface.update_state(position, state)

    We can now evaluate the log-probability of the updated model state:

    >>> interface.log_prob(state)
    Array(-1.4189385, dtype=float32)
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        self._log_prob_fn = log_prob_fn

    def extract_position(self, position_keys: Sequence[str], model_state: ModelState):
        """
        Extracts a position from a model state.

        Parameters
        ----------
        position_keys
            An iterable of variable or node names.
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.
        """
        return {key: getattr(model_state, key) for key in position_keys}

    def update_state(self, position, model_state: ModelState) -> ModelState:
        """
        Updates and returns a model state given a position.

        Parameters
        ----------
        position
            A dictionary of variable or node names and values.
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.

        Warnings
        --------
        The ``model_state`` must be up-to-date, i.e. it must *not* contain any outdated
        nodes. Updates can only be triggered through new variable or node values in the
        ``position``. If you supply a ``model_state`` with outdated nodes, these nodes
        and their outputs will not be updated.
        """
        new_state = model_state._replace(**position)
        return new_state

    def log_prob(self, model_state: ModelState) -> float:
        """
        Returns the log-probability from a model state.

        Parameters
        ----------
        model_state
            A dictionary of node names and their corresponding :class:`.NodeState`.
        """
        return self._log_prob_fn(model_state)
