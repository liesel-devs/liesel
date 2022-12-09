"""
Model visualization.
"""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def plot_nodes(model, show=True, save_path=None, width=14, height=10, prog="dot"):
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
        Layout parameter. Available layouts: circo, dot (the default), fdp, neato,
        osage, patchwork, sfdp, twopi.
    """

    colors = [
        "#fc8d62" if node.outdated else "#8da0cb" for node in model.node_graph.nodes
    ]

    _, axis, pos = _prepare_figure(model.node_graph, width, height, prog)
    nx.draw_networkx_nodes(model.node_graph, pos, node_color=colors, ax=axis)
    _add_labels(model.node_graph, axis, pos)
    _draw_edges(model.node_graph, axis, pos)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_vars(model, show=True, save_path=None, width=14, height=10, prog="dot"):
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
    """

    _, axis, pos = _prepare_figure(model.var_graph, width, height, prog)
    _add_nodes_with_distribution_to_plot(model.var_graph, axis, pos)
    _add_nodes_without_distribution_to_plot(model.var_graph, axis, pos)
    _add_labels(model.var_graph, axis, pos)
    _draw_edges(model.var_graph, axis, pos)
    _add_legend(axis)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def _prepare_figure(graph, width, height, prog):
    """Prepares the figure for plotting."""

    fig, axis = plt.subplots()
    fig.set_size_inches(width, height)

    if _is_pygraphviz_installed():
        pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog=prog)
    else:
        pos = nx.fruchterman_reingold_layout(graph)

    return fig, axis, pos


def _is_pygraphviz_installed():
    """Checks if pygraphviz is installed."""

    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        return False
    else:
        return True


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
        node: f"{type(node).__name__}\n{node.name}"
        if node.name is not None
        else node.role.name
        for node in pos
    }

    nx.draw_networkx_labels(graph, pos, labels=labels, ax=axis, font_size=10)


def _draw_edges(graph, axis, pos):
    """Adds edges to the figure."""

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=graph.edges,
        edge_color="#aaaaaa",
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
    ]

    axis.legend(handles=legend_elements, loc="best")
