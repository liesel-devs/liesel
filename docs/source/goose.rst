.. _goose_overview:

MCMC Sampling (liesel.goose)
============================

This is an overview of the most important building blocks for sampling from the
posterior of your model ``liesel.goose``.

We usually import ``liesel.goose`` as follows::

    import liesel.goose as gs

The workflow for MCMC sampling goose consists of the following steps:

1. Set up your model
2. Set up an :class:`.Engine` with MCMC kernels for your parameters and draw posterior samples
3. Inspect your results

.. note::
    This document provides an overview of the most important classes for MCMC sampling.
    You find more guidance on *how* to use them in the respective API documentation
    and in the :doc:`tutorials <tutorials_overview>`.

Set up your model
-----------------

A natural option for setting up your model is the use of ``liesel.model``. See
:doc:`Model Building with liesel.model <model>` for more. However, you are not locked
in to using :class:`.Model`. Goose currently includes the following interfaces:

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.goose.interface.LieselInterface
    ~liesel.goose.interface.DataclassInterface
    ~liesel.goose.interface.DictInterface
    ~liesel.goose.interface.NamedTupleInterface
    ~liesel.experimental.pymc.PyMCInterface


Find good starting values
-------------------------

It can often be beneficial to find good starting values to get your MCMC sampling scheme
going. Goose provides the function :func:`.optim` for this purpose, which allows you
to run stochastic gradient descent on a liesel model.

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.goose.optim.optim
    ~liesel.goose.optim.OptimResult


Set up an MCMC Engine and draw posterior samples
------------------------------------------------

To set up an MCMC engine, goose provides the :class:`.EngineBuilder`. Please refer to
the linked EngineBuilder documentation to learn how to use it.

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.goose.builder.EngineBuilder
    ~liesel.goose.engine.Engine


.. rubric:: Available MCMC kernels

Goose makes it easy for you to combine different MCMC kernels for different blocks of
model parameters. Currently, the available MCMC kernels are:

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.goose.rw.RWKernel
    ~liesel.goose.iwls.IWLSKernel
    ~liesel.goose.hmc.HMCKernel
    ~liesel.goose.nuts.NUTSKernel
    ~liesel.goose.gibbs.GibbsKernel

You can also define your own kernel by implementing the :class:`.Kernel` protocol.

To draw samples from your posterior, you will want to call
:meth:`.Engine.sample_all_epochs`. Once sampling is done, you can obtain the results
with :meth:`.Engine.get_results`, which will return a :class:`.SamplingResults`
instance.


Inspect your results
--------------------

The two central classes for handling your sampling results are:

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.goose.engine.SamplingResults
    ~liesel.goose.summary_m.Summary

You can obtain your posterior samples as a dictionary via
:meth:`.SamplingResults.get_posterior_samples`. There is also experimental support
for turning your samples into an ``arviz.InferenceData`` object via
:func:`.to_arviz_inference_data`.

.. rubric:: Plot posterior samples

Goose comes with a number of plotting functions that give you quick acccess to important
diagnostics.

.. autosummary::
    :toctree: generated
    :recursive:
    :nosignatures:

    ~liesel.goose.summary_viz.plot_param
    ~liesel.goose.summary_viz.plot_trace
    ~liesel.goose.summary_viz.plot_density
    ~liesel.goose.summary_viz.plot_pairs
    ~liesel.goose.summary_viz.plot_cor
