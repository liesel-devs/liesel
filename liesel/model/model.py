"""
The model and the graph builder.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from types import MappingProxyType
from typing import IO, Any, TypeVar

import dill
import jax
import jax.numpy as jnp
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
    Dist,
    Group,
    InputGroup,
    Node,
    NodeState,
    Value,
    Var,
    VarValue,
    is_bijector_class,
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
    A graph builder, used to set up a :class:`.Model`.

    Constructs a model containing all nodes and variables that were added to the graph
    builder and their recursive inputs.

    .. important::
        - In :meth:`.build_model` , the graph builder will automatically find all
          **inputs** to its nodes - and the inputs to these inputs
          (i.e. it finds inputs recursively).
        - The **outputs** of the nodes, however, are not added to the model
          automatically, so all **root nodes** need to be added explicitly.
        - Root nodes are nodes that are not inputs to any other node in the graph.
          The response in a regression model is an example of a root node.

    The standard workflow is to create the nodes and variables, add the root var to a
    graph builder, and construct a model from the graph builder. After the model has
    been constructed, some methods of the graph builder are not available anymore,
    because the graph is considered static.

    Parameters
    ----------
    to_float32
        Whether to convert the dtype of the values of the added nodes \
        from float64 to float32.

    See Also
    --------

    :class:`.Model` : The liesel model class, representing a static graph.
    :meth:`.GraphBuilder.add` : Method for adding variables and nodes to the
        GraphBuilder.
    :meth:`.GraphBuilder.build_model` : Method for building a model from the
        GraphBuilder.
    :meth:`.Var.transform` : Transforms a variable by adding a new transformed
        variable as an input. This is useful for variables that are constrained to a
        certain domain, e.g. positive values.

    Examples
    --------

    We start by creating some variables:

    >>> a = lsl.Var(1.0, name="a")
    >>> b = lsl.Var(2.0, name="b")
    >>> c = lsl.Var(lsl.Calc(lambda x, y: x + y, a, b), name="c")

    We now initialize a GraphBuilder and add the root node ``c`` to it:

    >>> gb = lsl.GraphBuilder()
    >>> gb.add(c)
    GraphBuilder(0 nodes, 1 vars)

    We are now ready to build the model:

    >>> model = gb.build_model()
    >>> model
    Model(9 nodes, 3 vars)

    Note that when :meth:`.build_model` is called, all :attr:`~.Var.weak` variables in
    the graph will be updated. So the value of ``c`` is now available:

    >>> c.value
    3.0

    The graph builder is now empty:

    >>> gb.vars
    []
    """

    def __init__(self, to_float32: bool = True):
        self.nodes: list[Node] = []
        """The nodes that were explicitly added to the graph."""

        self.vars: list[Var] = []
        """The variables that were explicitly added to the graph."""

        self._log_lik_node: Node | None = None
        self._log_prior_node: Node | None = None
        self._log_prob_node: Node | None = None
        self.to_float32 = to_float32

    def _add_model_log_lik_node(self) -> GraphBuilder:
        """Adds the model log-likelihood node with the name ``_model_log_lik``."""

        if self.log_lik_node:
            self.add(
                Calc(
                    lambda x: x,
                    self.log_lik_node,
                    _name="_model_log_lik",
                    update_on_init=False,
                )
            )
            return self

        _, _vars = self._all_nodes_and_vars()
        inputs = (v.dist_node for v in _vars if v.has_dist and v.observed)
        node = Calc(_reduced_sum, *inputs, _name="_model_log_lik", update_on_init=False)
        self.add(node)
        return self

    def _add_model_log_prior_node(self) -> GraphBuilder:
        """Adds the model log-prior node with the name ``_model_log_prior``."""

        if self.log_prior_node:
            self.add(
                Calc(
                    lambda x: x,
                    self.log_prior_node,
                    _name="_model_log_prior",
                    update_on_init=False,
                )
            )
            return self

        _, _vars = self._all_nodes_and_vars()
        inputs = (v.dist_node for v in _vars if v.has_dist and v.parameter)
        node = Calc(
            _reduced_sum, *inputs, _name="_model_log_prior", update_on_init=False
        )
        self.add(node)
        return self

    def _add_model_log_prob_node(self) -> GraphBuilder:
        """Adds the model log-probability node with the name ``_model_log_prob``."""

        if self.log_prob_node:
            self.add(
                Calc(
                    lambda x: x,
                    self.log_prob_node,
                    _name="_model_log_prob",
                    update_on_init=False,
                )
            )
            return self

        nodes, _ = self._all_nodes_and_vars()
        inputs = (n for n in nodes if isinstance(n, Dist))
        node = Calc(
            _reduced_sum, *inputs, _name="_model_log_prob", update_on_init=False
        )
        self.add(node)
        return self

    def _add_model_seed_nodes(self) -> GraphBuilder:
        """Adds the model seed nodes with the names ``_model_*_seed``."""
        nodes, _ = self._all_nodes_and_vars()

        for node in nodes:
            if node.needs_seed:
                seed = Value(jax.random.PRNGKey(0), _name=f"_model_{node.name}_seed")
                node.set_inputs(*node.inputs, **{"seed": seed} | node.kwinputs)

        return self

    def _all_nodes_and_vars(self) -> tuple[list[Node], list[Var]]:
        """
        Returns all nodes and variables that were explicitly or implicitly
        (as recursive inputs) added to the graph.
        """
        nodes = self.nodes.copy()
        nodes.extend(node for var in self.vars for node in var.nodes)

        nodes = list(dict.fromkeys(nodes))

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

    def add(
        self, *args: Node | Var | GraphBuilder, to_float32: bool | None = None
    ) -> GraphBuilder:
        """
        Adds nodes, variables or other graph builders to the graph.

        Parameters
        ----------
        *args
            The nodes, variables or graph builders to add to the graph. Note that \
            the GraphBuilder will find input nodes recursively for all nodes and \
            variables that are added to it, so you only need to add root nodes.
        to_float32
            Whether to convert the dtype of the values of the added nodes \
            from float64 to float32. If ``None`` (default), the GraphBuilder's \
            attribute ``GraphBuilder.to_float32``, which is set during initialization \
            will be used instead.

        See Also
        --------
        :meth:`.GraphBuilder.build_model` : Method for building a model from the \
            GraphBuilder.
        :meth:`.Var.transform` : Transforms a variable by adding a new
            transformed variable as an input.

        Examples
        --------

        We start by creating some variables:

        >>> a = lsl.Var(1.0, name="a")
        >>> b = lsl.Var(2.0, name="b")
        >>> c = lsl.Var(lsl.Calc(lambda x, y: x + y, a, b), name="c")

        We now initialize a GraphBuilder and add the root node ``c`` to it:

        >>> gb = lsl.GraphBuilder()
        >>> gb.add(c)
        GraphBuilder(0 nodes, 1 vars)

        We are now ready to build the model:

        >>> model = gb.build_model()
        >>> model
        Model(9 nodes, 3 vars)
        """

        if to_float32 is None:
            to_float32 = self.to_float32

        for arg in args:
            if isinstance(arg, Node):
                self.nodes.append(arg)
            elif isinstance(arg, Var):
                self.vars.append(arg)
            elif isinstance(arg, GraphBuilder):
                self.nodes.extend(arg.nodes)
                self.vars.extend(arg.vars)
            else:
                raise RuntimeError(f"Cannot add {type(arg).__name__} to graph builder")

        if to_float32:
            self.convert_dtype("float64", "float32")

        return self

    def add_groups(
        self, *groups: Group, to_float32: bool | None = None
    ) -> GraphBuilder:
        """
        Adds groups to the graph.

        Parameters
        ----------
        *groups
            The groups to add to the graph.
        to_float32
            Whether to convert the dtype of the values of the added nodes \
            from float64 to float32. If ``None`` (default), the GraphBuilder's \
            attribute ``GraphBuilder.to_float32``, which is set during initialization \
            will be used instead.

        Returns
        -------
        The graph builder.
        """

        if to_float32 is None:
            to_float32 = self.to_float32

        for group in groups:
            old = self.groups()

            if group.name in old and group is not old[group.name]:
                raise RuntimeError(
                    f"Group with name {repr(group.name)} already exists "
                    "in graph builder"
                )

            self.add(*group.nodes_and_vars.values())

        if to_float32:
            self.convert_dtype("float64", "float32")

        return self

    def build_model(self, copy: bool = False) -> Model:
        """
        Builds a model from the graph.

        Constructs a model containing all nodes and variables that were added to the
        graph builder and their recursive inputs. The outputs of the nodes are not added
        to the model automatically, so the root nodes always need to be added
        explicitly.

        The standard workflow is to create the nodes and variables, add them to a graph
        builder, and construct a model from the graph builder. After the model has been
        constructed, some methods of the graph builder are not available anymore.

        Parameters
        ----------
        copy
            Whether the nodes and variables should be copied when building the model.

        Returns
        -------
        The liesel model, which is a static graph built from the GraphBuilder.

        Notes
        -----
        If this method is called with the argument ``copy=False``, all nodes and
        variables are removed from the graph builder, because most methods of the graph
        builder do not work with nodes that are part of a model.

        Examples
        --------

        We start by creating some variables:

        >>> a = lsl.Var(1.0, name="a")
        >>> b = lsl.Var(2.0, name="b")
        >>> c = lsl.Var(lsl.Calc(lambda x, y: x + y, a, b), name="c")

        We now initialize a GraphBuilder and add the root node ``c`` to it:

        >>> gb = lsl.GraphBuilder()
        >>> gb.add(c)
        GraphBuilder(0 nodes, 1 vars)

        We are now ready to build the model:

        >>> model = gb.build_model()
        >>> model
        Model(9 nodes, 3 vars)

        Note that when :meth:`.build_model` is called, all :attr:`~.Var.weak` variables
        in the graph will be updated. So the value of ``c`` is now available:

        >>> c.value
        3.0

        The graph builder is now empty:

        >>> gb.vars
        []
        """
        nodes, _vars = self._all_nodes_and_vars()

        if not nodes:
            logger.warning("No nodes in graph builder, building an empty model")

        for node in nodes:
            if node.name.startswith("_model"):
                raise RuntimeError(f"{repr(node)} has reserved name '_model*'")

        gb = self.copy()

        nodes, _vars = gb._all_nodes_and_vars()

        for var in _vars:
            if var.auto_transform:
                if var.dist_node is None:
                    raise RuntimeError(
                        f"Auto-transform of {var} failed, because it has no"
                        " distribution, which means no default bijector can be found."
                    )
                tname = f"{var.name}_transformed"
                if tname in nodes or tname in _vars:
                    raise RuntimeError(
                        f"Auto-transform of {var} failed, because a variable of the "
                        f"name {tname} is already present in {gb}."
                    )
                var.transform(bijector=None)

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

    def convert_dtype(
        self, from_dtype: str | jax.numpy.dtype, to_dtype: str | jax.numpy.dtype
    ) -> GraphBuilder:
        """
        Tries to convert the node values in the graph to the specified data type.

        Works for nodes whose value is an array or pytree_. Nodes whose value is of
        another type are silently ignored.

        .. _pytree: https://jax.readthedocs.io/en/latest/pytrees.html

        Parameters
        ----------
        from_dtype
            The data type to convert from.
        to_dtype
            The data type to convert to.

        Returns
        -------
        The graph builder.


        """
        nodes, _ = self._all_nodes_and_vars()

        class ConversionWrapper:
            def __init__(self, value):
                self.value = value
                self.converted = False

                try:
                    if value.dtype == from_dtype:
                        self.value = value.astype(to_dtype)
                        self.converted = True
                except AttributeError:
                    pass

        for node in nodes:
            try:
                wrappers = jax.tree.map(ConversionWrapper, node.value)

                value = jax.tree.map(lambda x: x.value, wrappers)
                node.value = value  # type: ignore # data node

                converted = jax.tree.map(lambda x: x.converted, wrappers)

                if any(jax.tree_util.tree_flatten(converted)[0]):
                    logger.info(f"Converted dtype of {repr(node)}.value")
            except AttributeError:
                pass

        return self

    def copy(self) -> GraphBuilder:
        """Returns a shallow copy of the graph builder."""
        gb = GraphBuilder(to_float32=self.to_float32)
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

    def groups(self) -> dict[str, Group]:
        """Collects the groups from all nodes and variables."""
        nodes, _vars = self._all_nodes_and_vars()
        g1 = {g.name: g for n in nodes for g in n.groups.values()}
        g2 = {g.name: g for v in _vars for g in v.groups.values()}
        return g1 | g2

    @property
    def log_lik_node(self) -> Node | None:
        """User-defined log-likelihood node, if there is one."""
        return self._log_lik_node

    @log_lik_node.setter
    def log_lik_node(self, log_lik_node: Node | None):
        if log_lik_node and not isinstance(log_lik_node, Node):
            raise RuntimeError("The log-likelihood node must be a node, not var")

        self._log_lik_node = log_lik_node

    @property
    def log_prior_node(self) -> Node | None:
        """User-defined log-prior node, if there is one."""
        return self._log_prior_node

    @log_prior_node.setter
    def log_prior_node(self, log_prior_node: Node | None):
        if log_prior_node and not isinstance(log_prior_node, Node):
            raise RuntimeError("The log-prior node must be a node, not var")

        self._log_prior_node = log_prior_node

    @property
    def log_prob_node(self) -> Node | None:
        """User-defined log-probability node, if there is one."""
        return self._log_prob_node

    @log_prob_node.setter
    def log_prob_node(self, log_prob_node: Node | None):
        if log_prob_node and not isinstance(log_prob_node, Node):
            raise RuntimeError("The log-probability node must be a node, not var")

        self._log_prob_node = log_prob_node

    def plot_nodes(self) -> GraphBuilder:
        """
        Plots all nodes in the graph.

        See Also
        --------
        :meth:`.viz.plot_nodes` : The function used to plot the nodes.

        """
        nodes, _vars = self._all_nodes_and_vars()
        nodes_and_vars = nodes + _vars

        self._set_missing_names()
        model = Model(nodes_and_vars, grow=False)
        plot_nodes(model)

        model.pop_nodes_and_vars()

        return self

    def plot_vars(self) -> GraphBuilder:
        """
        Plots all variables in the graph.

        Returns
        -------
        The graph builder.

        See Also
        --------
        :meth:`.viz.plot_vars` : The function used to plot the variables.
        """
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
        self.nodes = [new if x is old else x for x in self.nodes]
        nodes, _ = self._all_nodes_and_vars()

        for node in nodes:
            inputs = [new if x is old else x for x in node.inputs]
            kwinputs = {k: new if v is old else v for k, v in node.kwinputs.items()}
            node.set_inputs(*inputs, **kwinputs)

        return self

    def replace_var(self, old: Var, new: Var) -> GraphBuilder:
        """Replaces the ``old`` with the ``new`` variable."""
        self.vars = [new if x is old else x for x in self.vars]
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
        self,
        var: Var,
        bijector: type[Bijector] | Bijector | None = None,
        *args,
        **kwargs,
    ) -> Var:
        """
        Transforms a variable by adding a new transformed variable as an input.

        .. deprecated:: 0.2.9
            ``GraphBuilder.transform`` is deprecated.
            Use :meth:`.Var.transform` instead. This method will be removed in v0.4.0.

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

        Examples
        --------

        >>> import tensorflow_probability.substrates.jax.distributions as tfd
        >>> import tensorflow_probability.substrates.jax.bijectors as tfb

        Assume we have a variable ``scale`` that is constrained to be positive, and
        we want to include the log-transformation of this variable in the model.
        We first set up the parameter var with its distribution:

        >>> prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        >>> scale = lsl.param(1.0, prior, name="scale")

        Then we create a GraphBuilder and use the ``transform`` method to transform
        the ``scale`` variable.

        >>> gb = lsl.GraphBuilder()
        >>> log_scale = gb.transform(scale, bijector=tfb.Exp)
        >>> log_scale
        Var(name="scale_transformed")

        Now the ``log_scale`` has a log probability, and the ``scale`` variable is
        has not:

        >>> log_scale.update().log_prob
        Array(-3.6720574, dtype=float32)

        >>> scale.update().log_prob
        0.0
        """

        warnings.warn(
            "GraphBuilder.transform is deprecated. Use Var.transform instead. "
            "This method will be removed in v0.4.0",
            FutureWarning,
        )

        if var.weak:
            raise RuntimeError(f"{repr(var)} is weak")

        if var.dist_node is None:
            raise RuntimeError(f"{repr(var)} has no distribution")

        if not is_bijector_class(bijector) and (args or kwargs):
            raise RuntimeError(
                "You passed a bijector instance and "
                "nonempty bijector arguments. You should either initialise your "
                "bijector directly with the arguments, or pass a bijector class "
                "instead."
            )

        # avoid name clashes
        self._set_missing_names()

        if var.value_node in self.nodes:
            raise RuntimeError(
                f"{var.value_node.name} is already present in the GraphBuilder. "
                + "If you have added some Node objects to the GraphBuilder, "
                "try to add only the Var objects instead."
            )

        # avoid infinite recursion
        var.auto_transform = False

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

        using_bijector_instance = not use_default_bijector and not isinstance(
            bijector, type
        )

        if isinstance(bijector, type) and not (args or kwargs):
            warnings.warn(
                "You passed a bijector class instead of an instance, but did not "
                "provide any arguments. You should either provide arguments "
                "or pass an instance of the bijector class.",
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

            if use_default_bijector:
                bijector_cls = tfp_dist.experimental_default_event_space_bijector
                bijector_obj = bijector_cls(*bijector_args.args, **bijector_args.kwargs)
            elif using_bijector_instance:
                bijector_obj = bijector
            else:
                bijector_cls = bijector
                bijector_obj = bijector_cls(*bijector_args.args, **bijector_args.kwargs)

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
        """
        Updates all nodes in the graph.

        Returns
        -------
        The graph builder.
        """
        nodes, _vars = self._all_nodes_and_vars()
        nodes_and_vars = nodes + _vars

        self._set_missing_names()
        model = Model(nodes_and_vars, grow=False)
        model.pop_nodes_and_vars()

        return self

    def __repr__(self) -> str:
        brackets = f"({len(self.nodes)} nodes, {len(self.vars)} vars)"
        return type(self).__name__ + brackets


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Model:
    """
    A model with a static graph.

    .. tip::
        If you have an existing model and want to make changes to it, you can use the
        :meth:`.Model.pop_nodes_and_vars` method to release the nodes and variables
        of the model. You can then make changes to them, for example i.e. changing the
        distribution of a variable or the inputs of a calculation. Afterwards, you
        initialize a *new* model with your changed variables.
        If you simply want to change the value of a variable, it is not necessary to
        call :meth:`~.Model.pop_nodes_and_vars`, you can simply override the
        :attr:`.Var.value` attribute. Remeber to call :meth:`.Model.update` afterwards.

    Parameters
    ----------
    nodes_and_vars
        The nodes and variables to include in the model.
    grow
        Whether a :class:`.GraphBuilder` should be used to grow the model (finding \
        the recursive inputs of the nodes and variables), and to add the model nodes.
    copy
        Whether the nodes and variables should be copied upon initialization.
    to_float32
        Whether to convert the dtype of the values of the added nodes \
        from float64 to float32. Only takes effect if ``grow=True``.

    See Also
    --------
    .Var.new_obs : Initializes a strong variable that holds observed data.
    .Var.new_param : Initializes a strong variable that acts as a model parameter.
    .Var.new_calc :
        Initializes a weak variable that is a function of other variables.
    .Var.new_value : Initializes a strong variable without a distribution.
    :class:`.GraphBuilder` :
        A graph builder, which can be used to set up and manipulate a model if you need
        more control.

    Examples
    --------

    Here, we set up a basic model based on three variables:

    >>> a = lsl.Var.new_value(1.0, name="a")
    >>> b = lsl.Var.new_value(2.0, name="b")
    >>> c = lsl.Var.new_calc(lambda x, y: x + y, a, b, name="c")

    We now build a model:

    >>> model = lsl.Model([c])
    >>> model
    Model(9 nodes, 3 vars)

    """

    def __init__(
        self,
        nodes_and_vars: Iterable[Node | Var],
        grow: bool = True,
        copy: bool = False,
        to_float32: bool = True,
    ):
        if grow:
            model = (
                GraphBuilder(to_float32=to_float32).add(*nodes_and_vars).build_model()
            )
            nodes_and_vars = [*model.nodes.values(), *model.vars.values()]
            model.pop_nodes_and_vars()

        nodes = [nv for nv in nodes_and_vars if isinstance(nv, Node)]
        nodes = list(dict.fromkeys(nodes).keys())
        counts = Counter(n.name for n in nodes)
        dups = [k for k, v in counts.items() if v > 1]

        if dups:
            raise RuntimeError(f"Duplicate node names: {', '.join(dups)}")

        _vars = [nv for nv in nodes_and_vars if isinstance(nv, Var)]
        _vars = list(dict.fromkeys(_vars).keys())
        counts = Counter(v.name for v in _vars)
        dups = [k for k, v in counts.items() if v > 1]

        if dups:
            raise RuntimeError(f"Duplicate variable names: {', '.join(dups)}")

        groups = [g for nv in nodes_and_vars for g in nv.groups.values()]
        groups = list(dict.fromkeys(groups).keys())
        counts = Counter(g.name for g in groups)
        dups = [k for k, v in counts.items() if v > 1]

        if dups:
            raise RuntimeError(f"Duplicate group names: {', '.join(dups)}")

        self._nodes = {n.name: n for n in nodes}
        self._vars = {v.name: v for v in _vars}

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

        self._simulation_graph = self._build_simulation_graph(self._nodes.values())
        self._simulation_nodes = list(nx.topological_sort(self._simulation_graph))

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
    def _build_simulation_graph(nodes: Iterable[Node]) -> nx.DiGraph:
        """Builds the simulation graph of the model nodes."""
        edges: list[tuple[Node, Node]] = []

        for node in nodes:
            for _input in node.all_input_nodes():
                if isinstance(node, Dist) and _input is node.at:
                    edges.append((node, _input))
                else:
                    edges.append((_input, node))

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

    def groups(self) -> dict[str, Group]:
        """Collects the groups from all nodes and variables."""
        g1 = {g.name: g for n in self._nodes.values() for g in n.groups.values()}
        g2 = {g.name: g for v in self._vars.values() for g in v.groups.values()}
        return g1 | g2

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

    def simulate(self, seed: jax.random.KeyArray, skip: Iterable[str] = ()) -> Model:
        """
        Updates the model state simulating from the probability distributions in the
        model using a provided random seed, optionally skipping specified nodes.

        Parameters
        ----------
        seed
            The seed is split and distributed to the distribution nodes in the model.
            Must be a ``KeyArray``, i.e. an array of shape (2,) and dtype ``uint32``.
            See :mod:`jax.random` for more details.
        skip
            The names of the nodes or variables to be excluded from the simulation. \
            By default, no nodes or variables are skipped.

        Returns
        -------
        The model instance itself after updating its state with the simulated values.

        Raises
        ------
        AttributeError
            If the value of the :attr:`.Dist.at` node of a distribution node cannot be
            set.

        Notes
        -----
        The simulation is based on the shapes of the current values of the
        :attr:`.Dist.at` nodes of the distribution nodes. If the :attr:`.Dist.at` node
        of a distribution node is a :Class:`.VarValue` node, the value of its input is
        updated.
        """
        dists = [
            node
            for node in self._simulation_nodes
            if isinstance(node, Dist)
            and node.at is not None
            and node.name not in skip
            and node.at.name not in skip
            and (node.var is not None and node.var.name not in skip)
        ]

        seeds = jax.random.split(seed, len(dists))

        for dist, seed in zip(dists, seeds):
            tfp_dist = dist.init_dist()

            event_shape = tfp_dist.event_shape
            batch_shape = tfp_dist.batch_shape
            value_shape = jnp.asarray(dist.at.value).shape  # type: ignore
            sample_index = len(value_shape) - len(batch_shape) - len(event_shape)
            sample_shape = value_shape[:sample_index]

            value = tfp_dist.sample(sample_shape, seed)

            if isinstance(dist.at, VarValue):
                try:
                    dist.at.inputs[0].value = value  # type: ignore
                except AttributeError:
                    raise AttributeError(f"Cannot set value of {dist.at.inputs[0]}")
            else:
                try:
                    dist.at.value = value  # type: ignore
                except AttributeError:
                    raise AttributeError(f"Cannot set value of {dist.at}")

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
        brackets = f"({len(self._nodes)} nodes, {len(self._vars)} vars)"
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
