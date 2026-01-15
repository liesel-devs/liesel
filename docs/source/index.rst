Liesel: A Probabilistic Programming Framework
=============================================

.. include:: welcome.md
   :parser: myst_parser.sphinx_

.. toctree::
   :caption: Guides
   :hidden:
   :maxdepth: 1

   tutorials_overview


API Reference
-------------

This is an overview of the central classes in Liesel.

Model Basics
************

The fundamental building blocks of your model graph are given by just three classes.
Both are documented with examples, so make sure to check them out.

The model building workflow in Liesel consists of the following steps:

1. Set up the nodes and variables that make up your model.
2. Initialize a :class:`.Model` with your root variable(s).

.. autosummary::
    :toctree: generated
    :caption: Model Basics
    :recursive:
    :nosignatures:

    ~liesel.model.Model
    ~liesel.model.Var
    ~liesel.model.Dist

MCMC Setup
************

To set up an MCMC engine, goose provides the :class:`~.goose.EngineBuilder`. Please refer to
the linked EngineBuilder documentation to learn how to use it.

A recent addition is the :class:`~.goose.MCMCSpec`, which can be passed to the
``inference`` argument of a :class:`.model.Var` upon initialization to tell the variable
directly how it should be sampled. You can then use the method
:meth:`~.goose.LieselMCMC.get_engine_builder` of :class:`~.goose.LieselMCMC` to
conveniently initialize your :class:`~.goose.EngineBuilder`.

.. autosummary::
    :toctree: generated
    :caption: MCMC Setup
    :recursive:
    :nosignatures:

    ~liesel.goose.LieselMCMC
    ~liesel.goose.MCMCSpec
    ~liesel.goose.EngineBuilder
    ~liesel.goose.Engine

MCMC Kernels
*************


Goose makes it easy for you to combine different MCMC kernels for different blocks of
model parameters. You can also define your own kernel by implementing
the :class:`.Kernel` protocol.

To draw samples from your posterior, you will want to call
:meth:`~.goose.Engine.sample_all_epochs`. Once sampling is done, you can obtain the results
with :meth:`~.goose.Engine.get_results`, which will return a :class:`~.goose.SamplingResults`
instance.

.. autosummary::
    :toctree: generated
    :caption: MCMC Kernels
    :recursive:
    :nosignatures:

    ~liesel.goose.IWLSKernel
    ~liesel.goose.NUTSKernel
    ~liesel.goose.HMCKernel
    ~liesel.goose.RWKernel
    ~liesel.goose.MHKernel
    ~liesel.goose.MHProposal
    ~liesel.goose.GibbsKernel
    ~liesel.goose.Kernel

Summary & Plots
****************************

The central classes for handling your sampling results are:

.. autosummary::
    :toctree: generated
    :caption: MCMC Results & Summary
    :recursive:
    :nosignatures:

    ~liesel.goose.SamplingResults
    ~liesel.goose.Summary
    ~liesel.goose.SamplesSummary

You can obtain your posterior samples as a dictionary via
:meth:`~.goose.SamplingResults.get_posterior_samples`. There is also experimental support
for turning your samples into an ``arviz.InferenceData`` object via
:func:`.to_arviz_inference_data`.

Goose also comes with a number of plotting functions that give you quick
acccess to important diagnostics.

.. autosummary::
    :toctree: generated
    :caption: Plots
    :recursive:
    :nosignatures:

    ~liesel.goose.plot_trace
    ~liesel.goose.plot_cor
    ~liesel.goose.plot_pairs
    ~liesel.goose.plot_scatter
    ~liesel.goose.plot_density
    ~liesel.goose.plot_param

Optimization
*************

It can often be beneficial to find good starting values to get your MCMC sampling scheme
going. Goose provides the function :func:`.optim_flat` for this purpose, which allows you
to run stochastic gradient descent on a liesel model.

.. autosummary::
    :toctree: generated
    :caption: Optimization
    :recursive:
    :nosignatures:

    ~liesel.goose.optim_flat
    ~liesel.goose.Stopper
    ~liesel.goose.history_to_df
    ~liesel.goose.OptimResult

Model (Advanced)
************************

.. autosummary::
    :toctree: generated
    :caption: Model (Advanced)
    :recursive:
    :nosignatures:

    ~liesel.model.Calc
    ~liesel.model.Node
    ~liesel.model.Value
    ~liesel.model.PIT
    ~liesel.distributions.GaussianCopula
    ~liesel.model.TransientCalc
    ~liesel.model.TransientDist
    ~liesel.model.TransientIdentity
    ~liesel.model.TransientNode
    ~liesel.model.InputGroup
    ~liesel.model.GraphBuilder

Model Interfaces
************************

A natural option for setting up your model is the use of ``liesel.model``.
However, you are not locked
in to using :class:`.Model`. Goose currently includes the following interfaces:

.. autosummary::
    :toctree: generated
    :caption: Model Interfaces
    :recursive:
    :nosignatures:

    ~liesel.goose.LieselInterface
    ~liesel.goose.DictInterface
    ~liesel.goose.DataclasslInterface
    ~liesel.goose.NamedTupleInterface

P-Splines
************

.. autosummary::
    :toctree: generated
    :caption: P-Splines
    :recursive:
    :nosignatures:

    ~liesel.contrib.splines.basis_matrix
    ~liesel.contrib.splines.equidistant_knots
    ~liesel.contrib.splines.pspline_penalty
    ~liesel.distributions.MultivariateNormalDegenerate


Advanced MCMC functionality
*****************************

.. autosummary::
    :toctree: generated
    :caption: MCMC (Advanced)

    ~liesel.goose.da
    ~liesel.goose.mm
    ~liesel.goose.EpochConfig


Experimental API
************************

.. autosummary::
    :toctree: generated
    :caption: Experimental API
    :recursive:
    :nosignatures:

    ~liesel.experimental


Effort-Based Versioning
-----------------------


Starting with v0.4.0, we will be using effort-based versioning.
See the EffVer documentation at https://jacobtomlinson.dev/effver/

The JAX developers provide a wonderful summary:
https://docs.jax.dev/en/latest/jep/25516-effver.html

The following description is almost entirely quoted from the linked JAX page,
but it describes what we intend with effort-based versioning perfectly.

Effort-based versioning is a three-number versioning system,
similar to the better-known semantic versioning (SemVer: https://semver.org/).
It uses a three-number format: ``MACRO.MESO.MICRO``, where version numbers
are incremented based on the expected effort required to adapt to the change.

As an example, consider software with current version ``2.3.4``:

1. Increasing the *micro* version (i.e. releasing ``2.3.5``)
   signals to users that little to no effort is necessary on their part
   to adapt to the changes.
2. Increasing the *meso* version (i.e. releasing ``2.4.0``)
   signals to users that some small effort will be required
   for existing code to work with the changes.
3. Increasing the *macro* version (i.e. releasing ``3.0.0``)
   signals to users that significant effort may be required
   to update to the changes.

In some ways, this captures the essence of more commonly-used semantic versioning,
but avoids phrasing in terms of compatibility guarantees that are hard to meet in practice.

Zero Version
************

In addition, EffVer gives special meaning to the *zero version*.
Early releases of software are often versioned ``0.X.Y``, and in this case:

- ``X`` has the characteristics of the macro version.
- ``Y`` has the characteristics of the meso version.

Liesel has been in a zero-version state since its initial release,
and EffVer's zero-version case is a good post-facto description
of the implicit intent behind Liesel's releases to date.

In EffVer, bumping from ``0.X.Y`` to version ``1.0.0`` is recommended
when a certain level of stability has been reached in practice:
If you end up on a version like ``0.9.x`` for many months,
it is a good signal that things are pretty stable
and that it's time to switch to a ``1.0.0`` release.


Indices and Search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
