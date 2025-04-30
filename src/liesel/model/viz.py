"""
Model visualization.
"""

import logging
from typing import IO, Literal

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def plot_nodes(
    model,
    show: bool = True,
    save_path: str | None | IO = None,
    width: int = 14,
    height: int = 10,
    prog: Literal[
        "dot", "circo", "fdp", "neato", "osage", "patchwork", "sfdp", "twopi"
    ] = "dot",
):
    """
    Plots the nodes of a Liesel model.

    Parameters
    ----------
    model
        The model to be plotted.
    show
        Whether to show the plot in a new window.
    save_path
        Path to save the plot. If not provided, the plot will not be saved.
    width
        Width of the plot in inches.
    height
        Height of the plot in inches.
    prog
        Layout parameter. Available layouts: circo, dot (the default), fdp, neato, \
        osage, patchwork, sfdp, twopi.

    See Also
    --------
    .Var.plot_vars : Plots the variables of the Liesel sub-model that terminates in
        this variable.
    .Var.plot_nodes : Plots the nodes of the Liesel sub-model that terminates in
        this variable.
    .Model.plot_vars : Plots the variables of a Liesel model.
    .Model.plot_nodes : Plots the nodes of a Liesel model.
    .viz.plot_vars : Plots the variables of a Liesel model.
    .viz.plot_nodes : Plots the nodes of a Liesel model.
    """

    try:
        graph = model.node_graph
    except AttributeError:
        graph = model

    colors = ["#fc8d62" if node.outdated else "#8da0cb" for node in graph.nodes]

    _, axis, pos = _prepare_figure(graph, width, height, prog)
    nx.draw_networkx_nodes(graph, pos, node_color=colors, ax=axis)
    _add_labels(graph, axis, pos)
    _draw_edges(graph, axis, pos, False)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_vars(
    model,
    show: bool = True,
    save_path: str | None | IO = None,
    width: int = 14,
    height: int = 10,
    prog: Literal[
        "dot", "circo", "fdp", "neato", "osage", "patchwork", "sfdp", "twopi"
    ] = "dot",
):
    """
    Plots the variables of a Liesel model.

    Parameters
    ----------
    model
        The model to be plotted.
    show
        Whether to show the plot in a new window.
    save_path
        Path to save the plot. If not provided, the plot will not be saved.
    width
        Width of the plot in inches.
    height
        Height of the plot in inches.
    prog
        Layout parameter. Available layouts: circo, dot (the default), fdp, neato,
        osage, patchwork, sfdp, twopi.

    See Also
    --------
    .Var.plot_vars : Plots the variables of the Liesel sub-model that terminates in
        this variable.
    .Var.plot_nodes : Plots the nodes of the Liesel sub-model that terminates in
        this variable.
    .Model.plot_vars : Plots the variables of a Liesel model.
    .Model.plot_nodes : Plots the nodes of a Liesel model.
    .viz.plot_vars : Plots the variables of a Liesel model.
    .viz.plot_nodes : Plots the nodes of a Liesel model.
    """
    try:
        graph = model.var_graph
    except AttributeError:
        graph = model

    _, axis, pos = _prepare_figure(graph, width, height, prog)
    _add_nodes_with_distribution_to_plot(graph, axis, pos)
    _add_nodes_without_distribution_to_plot(graph, axis, pos)
    _add_labels(graph, axis, pos)
    _draw_edges(graph, axis, pos, True)
    _add_legend(axis)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def _prepare_figure(graph, width, height, prog):
    """Prepares the figure for plotting."""

    fig, axis = plt.subplots()
    fig.set_size_inches(width, height)

    try:
        pos = nx.nx_pydot.pydot_layout(graph, prog=prog)
    except FileNotFoundError:
        logger.warning(
            "Graphviz not found in PATH. Using fallback graph layout. "
            "Consider installing Graphviz: https://graphviz.org/download"
        )

        pos = nx.kamada_kawai_layout(graph.to_undirected())
    except Exception as e:
        logger.warning(
            "Graphviz via pydot failed. Using fallback graph layout. "
            f"Raised exception: {e}"
        )

        pos = nx.kamada_kawai_layout(graph.to_undirected())

    return fig, axis, pos


def _add_nodes_with_distribution_to_plot(graph, axis, pos):
    """Adds nodes with distribution to the figure."""

    nodes_with_distribution = {
        node: "#fc8d62" if node.weak else "#8da0cb"
        for node in graph.nodes
        if node.has_dist
    }

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1200,
        node_color=nodes_with_distribution.values(),
        nodelist=nodes_with_distribution,
        node_shape="*",
        ax=axis,
    )


def _add_nodes_without_distribution_to_plot(graph, axis, pos):
    """Adds nodes without distribution to the figure."""

    nodes_without_distribution = {
        node: "#fc8d62" if node.weak else "#8da0cb"
        for node in graph.nodes
        if not node.has_dist
    }

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=500,
        node_color=nodes_without_distribution.values(),
        nodelist=nodes_without_distribution,
        node_shape="o",
        ax=axis,
    )


def _add_labels(graph, axis, pos):
    """Adds labels to the figure."""

    labels = {
        node: (
            f"{type(node).__name__}\n{node.name}"
            if node.name is not None
            else node.role.name
        )
        for node in pos
    }

    nx.draw_networkx_labels(graph, pos, labels=labels, ax=axis, font_size=10)


def _draw_edges(graph, axis, pos, is_var):
    """Adds edges to the figure."""

    edges = list(graph.edges)

    if is_var:
        dist_edges = []
        value_edges = []

        for edge in edges:
            # find distribution edges
            if edge[1].has_dist:
                edge_0_output_nodes = set(edge[0].all_output_nodes())
                edge_0_nodes = edge[0].nodes
                edge_1_input_nodes = set(edge[1].dist_node.all_input_nodes())

                if bool(edge_0_output_nodes.union(edge_0_nodes) & edge_1_input_nodes):
                    dist_edges.append(edge)

            # find value edges
            edge_0_output_nodes = set(edge[0].all_output_nodes())
            edge_0_nodes = edge[0].nodes
            edge_1_input_nodes = set(edge[1].value_node.all_input_nodes())

            if bool(edge_0_output_nodes.union(edge_0_nodes) & edge_1_input_nodes):
                value_edges.append(edge)

        edges_in_both = set(dist_edges) & set(value_edges)
        dist_edges = set(dist_edges) - edges_in_both
        value_edges = set(value_edges) - edges_in_both

        # assigns value_edges to edges to make it comparible with is_var=False
        edges = value_edges

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edges_in_both,
            edge_color="#FF0000",
            arrows=True,
            ax=axis,
            node_size=500,
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=dist_edges,
            edge_color="#aaaaaa",
            arrows=True,
            ax=axis,
            node_size=500,
        )

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=edges,
        edge_color="#111111",
        arrows=True,
        ax=axis,
        node_size=500,
    )


def _add_legend(axis):
    """Adds a legend to the figure."""

    legend_elements = [
        Line2D([0], [0], color="#8da0cb", lw=4, label="Strong"),
        Line2D([0], [0], color="#fc8d62", lw=4, label="Weak"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="With distribution",
            markerfacecolor="k",
            markersize=18,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Without distribution",
            markerfacecolor="k",
            markersize=12,
        ),
        Line2D(
            [0],
            [0],
            marker=r"$\rightarrow$",
            color="#111111",
            label="Used in value",
            markerfacecolor="k",
            markersize=12,
            lw=0,
        ),
        Line2D(
            [0],
            [0],
            marker=r"$\rightarrow$",
            color="#AAAAAA",
            label="Used in distribution",
            markerfacecolor="k",
            markersize=12,
            lw=0,
        ),
        Line2D(
            [0],
            [0],
            marker=r"$\rightarrow$",
            color="#FF0000",
            label="Used in value and distribution",
            markerfacecolor="k",
            markersize=12,
            lw=0,
        ),
    ]

    axis.legend(handles=legend_elements, loc="best")
