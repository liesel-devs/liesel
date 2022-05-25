"""
# Probabilistic graphical models (PGMs)
"""

from __future__ import annotations

import logging
import pickle
import re
from copy import deepcopy
from functools import reduce
from typing import Iterable, TypeVar

import networkx as nx

from .nodes import Node, NodeGroup, transform_parameter
from .types import ModelState, NodeState, TFPBijectorClass

logger = logging.getLogger(__name__)


def _join_args(args: Iterable[str]) -> str:
    args = [arg for arg in args if arg != ""]
    return ", ".join(args)


def _nodes_to_args(nodes: Iterable[Node], short: bool = False) -> str:
    args = [f"{node:s}" for node in nodes]

    if short and len(args) > 3:
        args = args[0:3] + ["..."]

    return _join_args(args)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


T = TypeVar("T", Node, NodeGroup)


class Model:
    """
    A probabilistic graphical model.

    A model is defined by the input-output relations between a number of
    `liesel.liesel.nodes.Node` objects. This class provides methods to compute
    the log-probability of a model and to update its nodes efficiently.

    **Caveat emptor:** This class does not check for missing inputs or groups
    that are not fully contained in a model. Use the `ModelBuilder` to set up
    a model in a safe way.

    ## Attributes

    - `graph`: The graph of the input-output relations between the nodes in the model.
    - `groups`: A dictionary of the nodes groups in the model with their names as keys.
    - `nodes`: A dictionary of the nodes in the model with their names as keys.
    - `sorted_nodes`: A list of the nodes in the model in a topological order.
    """

    def __init__(
        self,
        nodes: Iterable[Node] | None = None,
        groups: Iterable[NodeGroup] | None = None,
    ) -> None:
        nodes = self._update_names(nodes or [], "n")
        self.nodes = {node.name: node for node in nodes}

        groups = self._update_names(groups or [], "g")
        self.groups = {group.name: group for group in groups}

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)

        for node in nodes:
            node.outputs = set()

        for node in nodes:
            for input in node.inputs:
                self.graph.add_edge(input, node)
                input.outputs.add(node)

        self.sorted_nodes = list(nx.topological_sort(self.graph))

        for node in self.sorted_nodes:
            node.model = self
            node.update()

    @staticmethod
    def _update_names(xs: Iterable[T], prefix: str) -> Iterable[T]:
        """
        Sets any missing node or group names during initialization and checks for
        duplicate names.
        """

        names = [x.name for x in xs if x.has_name]
        counter = 0

        for x in xs:
            if not x.has_name:
                while prefix + str(counter) in names:
                    counter += 1

                x.name = prefix + str(counter)
                names.append(x.name)

        if len(names) != len(set(names)):
            _type = "node" if isinstance(next(iter(xs)), Node) else "group"
            raise RuntimeError(f"Duplicate {_type} names supplied to Model()")

        return xs

    def empty_copy(self) -> Model:
        """Returns a deep copy of the model with an empty state."""

        state = self.state.copy()

        zeros = {name: NodeState(0.0, 0.0) for name in self.nodes}

        self.state = zeros
        copy = deepcopy(self)
        self.state = state

        return copy

    def get_nodes_by_class(self, cls: type[Node]) -> dict[str, Node]:
        """Returns the nodes of the provided class from the model."""

        return {
            name: node  # black: break line
            for name, node in self.nodes.items()
            if isinstance(node, cls)
        }

    def get_nodes_by_regex(self, regex: str) -> dict[str, Node]:
        """Returns the nodes with matching names from the model."""

        _regex = re.compile(regex)

        return {
            name: node  # black: break line
            for name, node in self.nodes.items()
            if _regex.search(name)
        }

    @property
    def jaxified(self) -> bool:
        """Whether JAX NumPy is enabled for all nodes in the model."""

        return all(node.jaxified for node in self.sorted_nodes)

    @jaxified.setter
    def jaxified(self, jaxified: bool) -> None:
        if jaxified:
            self.jaxify()
        else:
            self.unjaxify()

    def jaxify(self) -> Model:
        """Enables JAX NumPy for all nodes in the model."""

        for node in self.sorted_nodes:
            node.jaxify()

        return self

    @property
    def log_prob(self) -> float:
        """
        Returns the log-probability of the model.

        The log-probability is computed as the sum of the log-probabilities
        of all nodes in the model. In a Bayesian context, it can be understood
        as the unnormalized log-posterior.
        """

        log_probs = [node.log_prob for node in self.sorted_nodes]
        return reduce(lambda x, y: x + y, log_probs)

    @property
    def state(self) -> ModelState:
        """
        Returns the state of the model.

        A model state is a dictionary of node names and states.
        """

        return {node.name: node.state for node in self.sorted_nodes}

    @state.setter
    def state(self, state: ModelState) -> None:
        for name, node_state in state.items():
            self.nodes[name].state = node_state

    def transform_parameter(self, name: str, bijector: str | TFPBijectorClass) -> Model:
        """
        Returns a deep copy of the model with the transformed parameter.

        The node groups are currently not copied over to the new model.
        """

        nodes = deepcopy(self.nodes)

        group = transform_parameter(nodes[name], bijector)

        old = nodes[name]
        new = group["original"]
        nodes[name] = new

        for output in old.outputs:
            if output.has_calculator:
                inputs = output.calculator.inputs.replace(old, new)
                output.calculator._inputs = inputs

            if output.has_distribution:
                inputs = output.distribution.inputs.replace(old, new)
                output.distribution._inputs = inputs

        mb = ModelBuilder()
        mb.add_nodes(nodes.values())
        return mb.build()

    def unjaxify(self) -> Model:
        """Disables JAX NumPy for all nodes in the model."""

        for node in self.sorted_nodes:
            node.unjaxify()

        return self

    def update(self) -> Model:
        """
        Updates all outdated nodes in the model.

        The update is performed in a topological order, restoring a consistent state
        of the model. This method is called automatically by the nodes if their value
        has changed (unless requested otherwise by the user).
        """

        for node in self.sorted_nodes:
            if node.outdated:
                node.update()

        return self

    def __setstate__(self, state):
        self.__dict__.update(state)

        for node in self.sorted_nodes:
            node.model = self

        self.jaxified = self.jaxified

    def __repr__(self) -> str:
        cls = type(self).__name__
        nodes = self.sorted_nodes

        if nodes:
            args = _nodes_to_args(nodes, short=True)
            return f"{cls}([{args}])"

        return f"{cls}()"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model builder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ModelBuilder:
    """
    A builder class for the `Model`.

    ## Attributes

    - `nodes`: A list of nodes to be added to the model.
    - `groups`: A list of node groups to be added to the model.
    """

    def __init__(
        self,
        nodes: Iterable[Node] | None = None,
        groups: Iterable[NodeGroup] | None = None,
    ) -> None:
        self.nodes = list(nodes or [])
        self.groups = list(groups or [])

    def add_nodes(self, *nodes: Node | Iterable[Node]) -> ModelBuilder:
        """Adds nodes to the model builder."""

        for arg in nodes:
            if isinstance(arg, Node):
                self.nodes.append(arg)
            else:
                self.add_nodes(*arg)

        return self

    def add_groups(self, *groups: NodeGroup | Iterable[NodeGroup]) -> ModelBuilder:
        """Adds node groups to the model builder."""

        for arg in groups:
            if isinstance(arg, NodeGroup):
                self.groups.append(arg)
            else:
                self.add_groups(*arg)

        return self

    @staticmethod
    def _add_inputs(nodes: Iterable[Node]) -> list[Node]:
        """Adds the inputs to an iterable of nodes."""

        nodes = list(nodes)
        visited = []

        while nodes:
            node = nodes.pop(0)  # pop from the left, maintain order

            if node not in visited:
                nodes.extend(node.inputs)
                visited.append(node)

        return visited

    def all_nodes(self) -> list[Node]:
        """
        Returns a list of all *unique* nodes that will be part of the model,
        including the nodes from the groups and the inputs.
        """

        nodes = self.nodes.copy()

        for group in self.groups:
            nodes.extend(group.values())

        nodes = self._add_inputs(nodes)
        nodes = list(dict.fromkeys(nodes))  # remove duplicates, maintain order
        return nodes

    def build(self) -> Model:
        """Builds the model, including the inputs."""

        nodes = self.all_nodes()

        if not nodes:
            logger.warning(f"No nodes in {repr(self)}, building an empty model")

        return Model(nodes, self.groups)

    def __repr__(self) -> str:
        cls = type(self).__name__
        nodes = self.all_nodes()

        if nodes:
            args = _nodes_to_args(nodes, short=True)
            return f"{cls}([{args}])"

        return f"{cls}()"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save and load models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_model(model: Model, path: str) -> None:
    """Saves a model to a pickle file."""

    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str) -> Model:
    """Loads a model from a pickle file."""

    with open(path, "rb") as handle:
        model = pickle.load(handle)

    return model
