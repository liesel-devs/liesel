.. _model_overview:

Model Building (liesel.model)
=============================

This is an overview of the most important building blocks for setting up your model
graph with ``liesel.model``.

The import path for ``liesel.model`` is::

    import liesel.model as lsl


The model building workflow in Liesel consists of two steps:

1. Set up the nodes and variables that make up your model.
2. Set up a :class:`.GraphBuilder`, add your root node(s) to it, and call :meth:`.GraphBuilder.build_model` to build your model graph.

.. note::
    This document provides an overview of the most important classes for model building.
    You find more guidance on *how* to use them in the respective API documentation
    and in the :doc:`tutorials <tutorials_overview>`.


Nodes and Variables
-------------------

The fundamental building blocks of your model graph are given by just three classes
and two functions. Each of these building blocks is documented with examples, so make
sure to check them out.

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.model.nodes.Data
    ~liesel.model.nodes.Calc
    ~liesel.model.nodes.Dist
    ~liesel.model.nodes.obs
    ~liesel.model.nodes.param


Build and plot your model
-------------------------

The most important class here is the :class:`.GraphBuilder`.

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.model.model.GraphBuilder
    ~liesel.model.model.Model
    ~liesel.model.viz.plot_vars
    ~liesel.model.viz.plot_nodes
