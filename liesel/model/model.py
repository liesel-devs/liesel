"""
The model and the graph builder.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from types import MappingProxyType
from typing import IO, Any, TypeVar

import dill
import jax
import jax.random
import networkx as nx
import tensorflow_probability.substrates.jax.bijectors as jb
import tensorflow_probability.substrates.jax.distributions as jd
import tensorflow_probability.substrates.numpy.bijectors as nb
import tensorflow_probability.substrates.numpy.distributions as nd

from .nodes import (
    ArgGroup,
    Array,
    Bijector,
    Calc,
    Data,
    Dist,
    InputGroup,
    Node,
    NodeState,
    TransientIdentity,
    Var,
)
from .viz import plot_nodes, plot_vars

__all__ = ["GraphBuilder", "Model", "load_model", "save_model"]

logger = logging.getLogger(__name__)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Graph builder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


NV = TypeVar("NV", Node, Var)


def _reduced_sum(*args: Array) -> Array:
    """Computes the sum after reducing arrays to scalars."""
    reduced = (arg.sum() if hasattr(arg, "sum") else arg for arg in args)
    return sum(reduced)


def _transform_back(var_transformed: Var) -> Calc:
    """
    Creates a :class:`.Calc` mapping a transformed parameter back to
    the original domain.
    """

    if var_transformed.dist_node is None:
        raise RuntimeError(
            f"{repr(var_transformed)} must have a transformed distribution"
        )

    transformed_distribution = var_transformed.dist_node.distribution

    def fn(at, *args, **kwargs):
        bijector = transformed_distribution(*args, **kwargs).bijector
        return bijector.inverse(at)

    inputs = var_transformed.dist_node.inputs
    kwinputs = var_transformed.dist_node.kwinputs

    return Calc(fn, var_transformed.value_node, *inputs, **kwinputs)  # type: ignore


class GraphBuilder:
    """
    A graph builder to prepare a :class:`.Model`.

    Constructs a model containing all nodes and variables that were added to the graph
    builder and their recursive inputs. The outputs of the nodes are not added to the
    model automatically, so the root nodes always need to be added explicitly.

    The standard workflow is to create the nodes and variables, add them to a graph
    builder, and construct a model from the graph builder. After the model has been
    constructed, some methods of the graph builder are not available anymore.

    Example
    -------
    >>> a = lsl.Var(1.0, name="a")
    >>> b = lsl.Var(2.0, name="b")
    >>> c = lsl.Var(lsl.Calc(lambda x, y: x + y, a, b), name="c")
    >>> gb = lsl.GraphBuilder()
    >>> gb.add(c)
    GraphBuilder<0 nodes, 1 vars>
    >>> model = gb.build_model()
    >>> model
    Model<9 nodes, 3 vars>
    >>> c.value
    3.0
    >>> gb.vars
    set()
    """

    def __init__(self):
        self.nodes: set[Node] = set()
        """The nodes that were explicitly added to the graph."""

        self.vars: set[Var] = set()
        """The variables that were explicitly added to the graph."""

        self._log_lik_node: Node | None = None
        self._log_prior_node: Node | None = None
        self._log_prob_node: Node | None = None

    def _add_model_log_lik_node(self) -> GraphBuilder:
        """Adds the model log-likelihood node with the name ``_model_log_lik``."""

        if self.log_lik_node:
            self.add(TransientIdentity(self.log_lik_node, _name="_model_log_lik"))
            return self

        _, _vars = self._all_nodes_and_vars()
        inputs = (v.dist_node for v in _vars if v.has_dist and v.observed)
        node = Calc(_reduced_sum, *inputs, _name="_model_log_lik")
        self.add(node)
        return self

    def _add_model_log_prior_node(self) -> GraphBuilder:
        """Adds the model log-prior node with the name ``_model_log_prior``."""

        if self.log_prior_node:
            self.add(TransientIdentity(self.log_prior_node, _name="_model_log_prior"))
            return self

        _, _vars = self._all_nodes_and_vars()
        inputs = (v.dist_node for v in _vars if v.has_dist and v.parameter)
        node = Calc(_reduced_sum, *inputs, _name="_model_log_prior")
        self.add(node)
        return self

    def _add_model_log_prob_node(self) -> GraphBuilder:
        """Adds the model log-probability node with the name ``_model_log_prob``."""

        if self.log_prob_node:
            self.add(TransientIdentity(self.log_prob_node, _name="_model_log_prob"))
            return self

        nodes, _ = self._all_nodes_and_vars()
        inputs = (n for n in nodes if isinstance(n, Dist))
        node = Calc(_reduced_sum, *inputs, _name="_model_log_prob")
        self.add(node)
        return self

    def _add_model_seed_nodes(self) -> GraphBuilder:
        """Adds the model seed nodes with the names ``_model_*_seed``."""
        nodes, _ = self._all_nodes_and_vars()

        for node in nodes:
            if node.needs_seed:
                seed = Data(jax.random.PRNGKey(0), _name=f"_model_{node.name}_seed")
                node.set_inputs(*node.inputs, **{"seed": seed} | node.kwinputs)

        return self

    def _all_nodes_and_vars(self) -> tuple[list[Node], list[Var]]:
        """
        Returns all nodes and variables that were explicitly or implicitly
        (as recursive inputs) added to the graph.
        """
        nodes = list(self.nodes)

        nodes.extend(node for var in self.vars for node in var.nodes)

        if self.log_lik_node:
            nodes.append(self.log_lik_node)

        if self.log_prior_node:
            nodes.append(self.log_prior_node)

        if self.log_prob_node:
            nodes.append(self.log_prob_node)

        all_nodes: list[Node] = []
        all_vars: list[Var] = []

        while nodes:
            node = nodes.pop()

            if node in all_nodes:
                continue

            nodes.extend(node.all_input_nodes())
            all_nodes.append(node)

            if node.var:
                if node.var in all_vars:
                    continue

                nodes.extend(node.var.nodes)
                all_vars.append(node.var)

        return all_nodes, all_vars

    @staticmethod
    def _do_set_missing_names(nodes_or_vars: Iterable[NV], prefix: str) -> None:
        """Sets the missing names for the given nodes or variables."""
        other = [nv.name for nv in nodes_or_vars if nv.name]
        counter = -1

        for nv in nodes_or_vars:
            if not nv.name:
                name = f"{prefix}{(counter := counter + 1)}"

                while name in other:
                    name = f"{prefix}{(counter := counter + 1)}"

                nv.name = name
                other.append(name)

    def _set_missing_names(self) -> GraphBuilder:
        """Sets the missing node and variable names."""
        nodes, _vars = self._all_nodes_and_vars()
        self._do_set_missing_names(_vars, prefix="v")
        self._do_set_missing_names(nodes, prefix="n")
        return self

    def add(self, *args: Node | Var | GraphBuilder) -> GraphBuilder:
        """Adds nodes, variables or other graph builders to the graph."""

        for arg in args:
            if isinstance(arg, Node):
                self.nodes.add(arg)
            elif isinstance(arg, Var):
                self.vars.add(arg)
            elif isinstance(arg, GraphBuilder):
                self.nodes.update(arg.nodes)
                self.vars.update(arg.vars)
            else:
                raise RuntimeError(f"Cannot add {type(arg).__name__} to graph builder")

        return self

    def add_group(self, name: str, **kwargs: Node | Var) -> GraphBuilder:
        """
        Adds a group to the graph.

        Also assigns the nodes and variables to the group,
        see :attr:`liesel.model.nodes.Node.groups`.

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

        self.add(*kwargs.values())
        return self

    def build_model(self, copy: bool = False) -> Model:
        """
        Builds a model from the graph.

        Constructs a model containing all nodes and variables that were added to the
        graph builder and their recursive inputs. The outputs of the nodes are not
        added to the model automatically, so the root nodes always need to be added
        explicitly.

        The standard workflow is to create the nodes and variables, add them to a graph
        builder, and construct a model from the graph builder. After the model has been
        constructed, some methods of the graph builder are not available anymore.

        Notes
        -----
        If this method is called with the argument ``copy=False``, all nodes and
        variables are removed from the graph builder, because most methods of the
        graph builder do not work with nodes that are part of a model.

        Example
        -------
        >>> a = lsl.Var(1.0, name="a")
        >>> b = lsl.Var(2.0, name="b")
        >>> c = lsl.Var(lsl.Calc(lambda x, y: x + y, a, b), name="c")
        >>> gb = lsl.GraphBuilder()
        >>> gb.add(c)
        GraphBuilder<0 nodes, 1 vars>
        >>> model = gb.build_model()
        >>> model
        Model<9 nodes, 3 vars>
        >>> c.value
        3.0
        >>> gb.vars
        set()

        Parameters
        ----------
        copy
            Whether the nodes and variables should be copied when building the model.
        """
        nodes, _vars = self._all_nodes_and_vars()

        if not nodes:
            logger.warning("No nodes in graph builder, building an empty model")

        for node in nodes:
            if node.name.startswith("_model"):
                raise RuntimeError(f"{repr(node)} has reserved name '_model*'")

        gb = self.copy()
        gb._set_missing_names()
        gb._add_model_log_lik_node()
        gb._add_model_log_prior_node()
        gb._add_model_log_prob_node()
        gb._add_model_seed_nodes()

        nodes, _vars = gb._all_nodes_and_vars()
        nodes_and_vars = nodes + _vars

        model = Model(nodes_and_vars, grow=False, copy=copy)

        if not copy:
            self.nodes.clear()
            self.vars.clear()

            self._log_lik_node = None
            self._log_prior_node = None
            self._log_prob_node = None

        return model

    def copy(self) -> GraphBuilder:
        """Returns a shallow copy of the graph builder."""
        gb = GraphBuilder()
        gb.nodes = self.nodes.copy()
        gb.vars = self.vars.copy()

        gb.log_lik_node = self.log_lik_node
        gb.log_prior_node = self.log_prior_node
        gb.log_prob_node = self.log_prob_node

        return gb

    def count_node_names(self) -> dict[str, int]:
        """Counts the number of times each node name occurs in the graph."""
        nodes, _ = self._all_nodes_and_vars()
        counter = Counter(node.name for node in nodes if node.name)
        return dict(counter.most_common())

    def count_var_names(self) -> dict[str, int]:
        """Counts the number of times each variable name occurs in the graph."""
        _, _vars = self._all_nodes_and_vars()
        counter = Counter(var.name for var in _vars if var.name)
        return dict(counter.most_common())

    @property
    def log_lik_node(self) -> Node | None:
        """The user-defined log-likelihood node."""
        return self._log_lik_node

    @log_lik_node.setter
    def log_lik_node(self, log_lik_node: Node | None):
        if log_lik_node and not isinstance(log_lik_node, Node):
            raise RuntimeError("The log-likelihood node must be a node, not var")

        self._log_lik_node = log_lik_node

    @property
    def log_prior_node(self) -> Node | None:
        """The user-defined log-prior node."""
        return self._log_prior_node

    @log_prior_node.setter
    def log_prior_node(self, log_prior_node: Node | None):
        if log_prior_node and not isinstance(log_prior_node, Node):
            raise RuntimeError("The log-prior node must be a node, not var")

        self._log_prior_node = log_prior_node

    @property
    def log_prob_node(self) -> Node | None:
        """The user-defined log-probability node."""
        return self._log_prob_node

    @log_prob_node.setter
    def log_prob_node(self, log_prob_node: Node | None):
        if log_prob_node and not isinstance(log_prob_node, Node):
            raise RuntimeError("The log-probability node must be a node, not var")

        self._log_prob_node = log_prob_node

    def plot_nodes(self) -> GraphBuilder:
        """Plots all nodes in the graph."""
        nodes, _vars = self._all_nodes_and_vars()
        nodes_and_vars = nodes + _vars

        self._set_missing_names()
        model = Model(nodes_and_vars, grow=False)
        plot_nodes(model)

        model.pop_nodes_and_vars()

        return self

    def plot_vars(self) -> GraphBuilder:
        """Plots all variables in the graph."""
        nodes, _vars = self._all_nodes_and_vars()
        nodes_and_vars = nodes + _vars

        self._set_missing_names()
        model = Model(nodes_and_vars, grow=False)
        plot_vars(model)

        model.pop_nodes_and_vars()

        return self

    def rename(self, pattern: str, replacement: str) -> GraphBuilder:
        """Renames all nodes and variables in the graph."""
        self.rename_nodes(pattern, replacement)
        self.rename_vars(pattern, replacement)
        return self

    def rename_nodes(self, pattern: str, replacement: str) -> GraphBuilder:
        """Renames all nodes in the graph."""
        nodes, _ = self._all_nodes_and_vars()

        for node in nodes:
            if node.name:
                node.name = re.sub(pattern, replacement, node.name)

        return self

    def rename_vars(self, pattern: str, replacement: str) -> GraphBuilder:
        """Renames all variables in the graph."""
        _, _vars = self._all_nodes_and_vars()

        for var in _vars:
            if var.name:
                var.name = re.sub(pattern, replacement, var.name)

        return self

    def replace_node(self, old: Node, new: Node) -> GraphBuilder:
        """Replaces the ``old`` with the ``new`` node."""
        self.nodes = {new if x is old else x for x in self.nodes}
        nodes, _ = self._all_nodes_and_vars()

        for node in nodes:
            inputs = [new if x is old else x for x in node.inputs]
            kwinputs = {k: new if v is old else v for k, v in node.kwinputs.items()}
            node.set_inputs(*inputs, **kwinputs)

        return self

    def replace_var(self, old: Var, new: Var) -> GraphBuilder:
        """Replaces the ``old`` with the ``new`` variable."""
        self.vars = {new if x is old else x for x in self.vars}
        self.replace_node(old.var_value_node, new.var_value_node)
        self.replace_node(old.value_node, new.value_node)

        if old.dist_node:
            if not new.dist_node:
                raise RuntimeError(
                    f"Cannot replace {repr(old)} with distribution "
                    f"with {repr(new)} without distribution"
                )

            self.replace_node(old.dist_node, new.dist_node)

        return self

    def transform(
        self, var: Var, bijector: type[Bijector] | None = None, *args, **kwargs
    ) -> Var:
        """
        Transforms a variable by adding a new transformed variable as an input.

        Creates a new variable on the unconstrained space ``R**n`` with the appropriate
        transformed distribution, turning the original variable into a weak variable
        without an associated distribution. The transformation is performed using
        TFP's bijector classes.

        The value of the attribute :attr:`~liesel.model.nodes.Var.parameter` is
        transferred to the transformed variable and set to ``False`` on the original
        variable. The attributes :attr:`~liesel.model.nodes.Var.observed` and
        :attr:`~liesel.model.nodes.Var.role` are set to the default values for
        the transformed variable and remain unchanged on the original variable.

        Parameters
        ----------
        var
            The variable to transform (and add to the graph).
        bijector
            The bijector used to map the new transformed variable to this variable \
            (forward transformation). If ``None``, the experimental default event \
            space bijector (see TFP documentation) is used.
        args
            The arguments passed on to the init function of the bijector.
        kwargs
            The keyword arguments passed on to the init function of the bijector.

        Returns
        -------
        The new transformed variable which acts as an input to this variable.

        Raises
        ------
        RuntimeError
            If the variable is weak, has no TFP distribution, the distribution has
            no default event space bijector and the argument ``bijector`` is ``None``,
            or the local model for the variable cannot be built.
        """

        if var.weak:
            raise RuntimeError(f"{repr(var)} is weak")

        if var.dist_node is None:
            raise RuntimeError(f"{repr(var)} has no distribution")

        # avoid name clashes
        self._set_missing_names()

        try:
            Model([var])
        except Exception:
            raise RuntimeError(f"Cannot build local model for {repr(var)}")

        self.add(var)

        # if we got this far, we can assume:
        # - the var and its inputs have numeric values
        # - the var and its inputs are up-to-date

        tfp_dist = var.dist_node.init_dist()
        default_bijector = tfp_dist.experimental_default_event_space_bijector()
        has_default_bijector = default_bijector is not None
        use_default_bijector = bijector is None

        if use_default_bijector and not has_default_bijector:
            raise RuntimeError(
                f"{repr(var)} has distribution without default event space bijector "
                "and no bijector was given"
            )

        if isinstance(tfp_dist, jd.Distribution):
            tfd = jd
            tfb = jb
        elif isinstance(tfp_dist, nd.Distribution):
            tfd = nd
            tfb = nb
        else:
            raise RuntimeError(f"{repr(var)} has no TFP distribution")

        # no copy necessary:
        # >>> from copy import copy
        # >>> import tensorflow_probability.substrates.numpy.distributions as tfd
        # >>> CopiedNormal = copy(tfd.Normal)
        # >>> CopiedNormal is tfd.Normal
        # True

        tfp_dist_cls = var.dist_node.distribution

        dist_inputs = InputGroup(
            *var.dist_node.inputs,
            **var.dist_node.kwinputs,  # type: ignore
        )

        bijector_inputs = InputGroup(*args, **kwargs)

        # define distribution "class" for the transformed var
        def make_transformed_distribution(dist_args: ArgGroup, bijector_args: ArgGroup):
            tfp_dist = tfp_dist_cls(*dist_args.args, **dist_args.kwargs)

            if bijector is None:
                bijector_obj = tfp_dist.experimental_default_event_space_bijector(
                    *bijector_args.args, **bijector_args.kwargs
                )

                bijector_inv = tfb.Invert(bijector_obj)
            else:
                bijector_obj = bijector(*bijector_args.args, **bijector_args.kwargs)
                bijector_inv = tfb.Invert(bijector_obj)

            return tfd.TransformedDistribution(
                tfp_dist, bijector_inv, validate_args=tfp_dist.validate_args
            )

        # build transformed var
        dist_node_transformed = Dist(
            make_transformed_distribution, dist_inputs, bijector_inputs
        )

        # transfer flags
        dist_node_transformed.needs_seed = var.dist_node.needs_seed
        dist_node_transformed.per_obs = var.dist_node.per_obs

        # transform value
        bijector_obj = dist_node_transformed.init_dist().bijector
        value_transformed = bijector_obj.forward(var.value)

        name_transformed = f"{var.name}_transformed" if var.name else ""

        var_transformed = Var(
            value_transformed, dist_node_transformed, name_transformed
        )

        var_transformed.parameter = var.parameter

        # var is now the forward transformation (a weak node without distribution)
        var.value_node = _transform_back(var_transformed)
        var.dist_node = None
        var.parameter = False

        return var_transformed

    def update(self) -> GraphBuilder:
        """Updates all nodes in the graph."""
        nodes, _vars = self._all_nodes_and_vars()
        nodes_and_vars = nodes + _vars

        self._set_missing_names()
        model = Model(nodes_and_vars, grow=False)
        model.pop_nodes_and_vars()

        return self

    def __repr__(self) -> str:
        brackets = f"<{len(self.nodes)} nodes, {len(self.vars)} vars>"
        return type(self).__name__ + brackets


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Model:
    """
    A model with a static graph.

    Parameters
    ----------
    nodes_and_vars
        The nodes and variables to include in the model.
    grow
        Whether a :class:`.GraphBuilder` should be used to grow the model (finding \
        the recursive inputs of the nodes and variables), and to add the model nodes.
    copy
        Whether the nodes and variables should be copied upon initialization.
    """

    def __init__(
        self,
        nodes_and_vars: Iterable[Node | Var],
        grow: bool = True,
        copy: bool = False,
    ):
        if grow:
            model = GraphBuilder().add(*nodes_and_vars).build_model()
            nodes_and_vars = [*model.nodes.values(), *model.vars.values()]
            model.pop_nodes_and_vars()

        self._nodes = {nv.name: nv for nv in nodes_and_vars if isinstance(nv, Node)}
        self._vars = {nv.name: nv for nv in nodes_and_vars if isinstance(nv, Var)}

        if len(self._nodes) < sum(isinstance(nv, Node) for nv in nodes_and_vars):
            raise RuntimeError("Model received nodes with duplicate names")

        if len(self._vars) < sum(isinstance(nv, Var) for nv in nodes_and_vars):
            raise RuntimeError("Model received vars with duplicate names")

        if copy:
            self._nodes, self._vars = deepcopy((self._nodes, self._vars))

        for node in self._nodes.values():
            node._clear_outputs()
            node._set_model(self)

        for node in self._nodes.values():
            for _input in node.all_input_nodes():
                _input._add_output(node)

        self._node_graph = self._build_node_graph(self._nodes.values())
        self._var_graph = self._build_var_graph(self._vars.values())

        self._sorted_nodes = list(nx.topological_sort(self._node_graph))

        self._auto_update = True
        self._seed_nodes = []

        for node in self._sorted_nodes:
            if node.name.startswith("_model_") and node.name.endswith("_seed"):
                self._seed_nodes.append(node)

            node.update()

    @staticmethod
    def _build_node_graph(nodes: Iterable[Node]) -> nx.DiGraph:
        """Builds the directed graph of the model nodes."""
        edges: list[tuple[Node, Node]] = []

        for node in nodes:
            edges.extend((_input, node) for _input in node.all_input_nodes())

        graph = nx.DiGraph(edges)
        graph.add_nodes_from(nodes)
        return graph

    @staticmethod
    def _build_var_graph(_vars: Iterable[Var]) -> nx.DiGraph:
        """Builds the directed graph of the model variables."""
        edges: list[tuple[Var, Var]] = []

        for var in _vars:
            edges.extend((_input, var) for _input in var.all_input_vars())

        graph = nx.DiGraph(edges)
        graph.add_nodes_from(_vars)
        return graph

    def _copy_computational_model(self) -> Model:
        """Returns a deep copy of the model with all node states cleared."""
        backup = self.state

        for node in self._nodes.values():
            node.clear_state()

        empty = deepcopy(self)
        self.state = backup

        return empty

    def _recursive_inputs(self, name: str) -> list[Node]:
        """Returns the recursive inputs of a model node."""
        nodes = [self._nodes[name]]
        visited = []

        while nodes:
            node = nodes.pop()

            if node in visited:
                continue

            nodes.extend(node.all_input_nodes())
            visited.append(node)

        return visited

    @property
    def auto_update(self) -> bool:
        """
        Whether to update the model automatically if the value of a node is modified.

        The auto-update can be disabled to improve the performance if multiple model
        parameters are updated at once.
        """
        return self._auto_update

    @auto_update.setter
    def auto_update(self, auto_update: bool):
        self._auto_update = auto_update

    def groups(self) -> dict[str, dict[str, Node | Var]]:
        """Composes the groups defined in the model nodes and variables."""
        result: dict[str, dict[str, Node | Var]] = {}

        for node in self._nodes.values():
            for group, key in node.groups:
                if group not in result:
                    result[group] = {}

                result[group][key] = node

        for var in self._vars.values():
            for group, key in var.groups:
                if group not in result:
                    result[group] = {}

                result[group][key] = var

        return result

    def copy_nodes_and_vars(self) -> tuple[dict[str, Node], dict[str, Var]]:
        """Returns an unfrozen deep copy of the model nodes and variables."""
        nodes, _vars = deepcopy((self._nodes, self._vars))

        for node in nodes.values():
            node._unset_model()

        nodes = {nm: nd for nm, nd in nodes.items() if not nm.startswith("_model")}

        return nodes, _vars

    @property
    def log_lik(self) -> Array:
        """
        The log-likelihood of the model.

        Defined as the sum of the log-probabilities of all observed variables
        with a probability distribution.
        """
        return self._nodes["_model_log_lik"].value

    @property
    def log_prior(self) -> Array:
        """
        The log-prior of the model.

        Defined as the sum of the log-probabilities of all parameter variables
        with a probability distribution.
        """
        return self._nodes["_model_log_prior"].value

    @property
    def log_prob(self) -> Array:
        """
        The (unnormalized) log-probability / log-posterior of the model.

        Defined as the sum of all distribution nodes.
        """
        return self._nodes["_model_log_prob"].value

    @property
    def node_graph(self) -> nx.DiGraph:
        """The directed graph of the model nodes."""
        return self._node_graph

    @property
    def nodes(self) -> MappingProxyType[str, Node]:
        """A mapping of the model nodes with their names as keys."""
        return MappingProxyType(self._nodes)

    def pop_nodes_and_vars(self) -> tuple[dict[str, Node], dict[str, Var]]:
        """
        Pops the nodes and variables out of this model.

        All nodes and variables are unfrozen and their reference to this model
        is removed. This model becomes invalid and cannot be used anymore.
        """
        nodes = self._nodes.copy()
        _vars = self._vars.copy()

        for node in nodes.values():
            node._unset_model()

        nodes = {nm: nd for nm, nd in nodes.items() if not nm.startswith("_model")}

        # clear the model
        self._nodes.clear()
        self._vars.clear()
        self._node_graph.clear()
        self._var_graph.clear()
        self._sorted_nodes.clear()
        self._seed_nodes.clear()

        return nodes, _vars

    def set_seed(self, seed: jax.random.KeyArray) -> Model:
        """
        Splits and sets the seed / PRNG key.

        Parameters
        ----------
        seed
            The seed is split and distributed to the seed nodes of the model.
            Must be a ``KeyArray``, i.e. an array of shape (2,) and dtype ``uint32``.
            See :mod:`jax.random` for more details.
        """
        seeds = jax.random.split(seed, len(self._seed_nodes))

        for node, seed in zip(self._seed_nodes, seeds):
            node.value = seed  # type: ignore  # data node

        return self

    @property
    def state(self) -> dict[str, NodeState]:
        """The state of the model as a dict of node names and states."""
        return {name: node.state for name, node in self._nodes.items()}

    @state.setter
    def state(self, state: dict[str, NodeState]):
        for name, node_state in state.items():
            self._nodes[name].state = node_state

    def update(self, *names: str) -> Model:
        """
        Updates the target nodes and their recursive inputs if they are outdated.

        The update is performed in a topological order, restoring a consistent state
        of the model. This method is called automatically by the nodes if their value
        is modified (unless :attr:`.auto_update` is ``False``).

        Parameters
        ----------
        names
            The names of the target nodes to be updated.
        """

        if not names:
            for node in self._sorted_nodes:
                if node.outdated:
                    node.update()
        else:
            inputs = set().union(*(self._recursive_inputs(name) for name in names))

            for node in self._sorted_nodes:
                if node in inputs and node.outdated:
                    node.update()

        return self

    @property
    def var_graph(self) -> nx.DiGraph:
        """The directed graph of the model variables."""
        return self._var_graph

    @property
    def vars(self) -> MappingProxyType[str, Var]:
        """A mapping of the model variables with their names as keys."""
        return MappingProxyType(self._vars)

    def __repr__(self) -> str:
        brackets = f"<{len(self._nodes)} nodes, {len(self._vars)} vars>"
        return type(self).__name__ + brackets


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save and load models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_model(model: Any, file: str | IO[bytes]) -> None:
    """
    Saves a model to a `dill <https://github.com/uqfoundation/dill>`_ file.

    Parameters
    ----------
    model
        The model to be saved.
    file
        The file handler or path to save the model to.
    """

    if isinstance(file, str):
        with open(file, "wb") as handle:
            dill.dump(model, handle)
    else:
        dill.dump(model, file)


def load_model(file: str | IO[bytes]) -> Any:
    """
    Loads a model from a `dill <https://github.com/uqfoundation/dill>`_ file.

    Parameters
    ----------
    file
        The file handler or path to load the model from.
    """

    if isinstance(file, str):
        with open(file, "rb") as handle:
            model = dill.load(handle)
    else:
        model = dill.load(file)

    return model
