.. _model_overview:

Model Building (liesel.model)
=============================

This is an overview of the most important building blocks for setting up your model
graph with ``liesel.model``.

We usually import ``liesel.model`` as follows::

    import liesel.model as lsl

Additionnaly, it often makes sense to import ``jax.numpy`` and tensorflow probability::

    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax.distributions as tfd


The model building workflow in Liesel consists of two steps:

1. Set up the nodes and variables that make up your model.
2. Initialize a :class:`.Model` with your root variable(s).

.. note::
    This document provides an overview of the most important classes for model building.
    You find more guidance on *how* to use them in the respective API documentation
    and in the :doc:`tutorials <tutorials_overview>`.


Variables: The model building blocks
------------------------------------

The fundamental building blocks of your model graph are given by just two classes.
Both are documented with examples, so make sure to check them out.

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.model.nodes.Var
    ~liesel.model.nodes.Dist

To set up :class:`.Var` objects, Liesel provides four constructors:

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.model.nodes.Var.new_param
    ~liesel.model.nodes.Var.new_obs
    ~liesel.model.nodes.Var.new_calc
    ~liesel.model.nodes.Var.new_value

We recommend to always use one of these constructors when initializing a variable.
This makes sure that the respective
:attr:`.Var.observerd` and :attr:`.Var.parameter` flags are correctly set. This in
turn ensures that the :attr:`.Var.log_prob` of an *observed* variable will be included
in the :attr:`.Model.log_lik` and the :attr:`.Var.log_prob` of a *parameter* variable
will be included in the :attr:`.Model.log_prior`.


Build and plot your model
-------------------------

The most important class here is the :class:`.Model`.

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.model.model.Model
    ~liesel.model.viz.plot_vars

For advanced users, further interesting functionality can be found here:

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.model.model.GraphBuilder
    ~liesel.model.viz.plot_nodes
