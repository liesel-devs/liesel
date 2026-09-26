Liesel: A Probabilistic Programming Framework
=============================================

.. include:: welcome.md
   :parser: myst_parser.sphinx_

.. toctree::
   :caption: Guides
   :hidden:
   :maxdepth: 1

   tutorials_overview
   optimization
   missing-values


API Reference
-------------

This is an overview of the central classes in Liesel.

.. _model-api:

.. _model-basics:

.. _model-advanced:

.. _custom-distributions:

.. _p-splines:

Models
******

The fundamental building blocks of your model graph are given by just three classes.
Both are documented with examples, so make sure to check them out.

The model building workflow in Liesel consists of the following steps:

1. Set up the nodes and variables that make up your model.
2. Initialize a :class:`Model <liesel.model.Model>` with your root variable(s).

.. autosummary::
   :toctree: generated
   :nosignatures:
   :caption: Models
   :recursive:

   ~liesel.model.Model
   ~liesel.model.Var
   ~liesel.model.Dist

.. toctree::
   :maxdepth: 1

   reference/model-nodes
   reference/model-density
   reference/model-distributions
   reference/model-splines

.. _mcmc-api:

.. _mcmc-setup:

.. _mcmc-kernels:

.. _model-interfaces:

.. _advanced-mcmc-functionality:

MCMC
****

To set up an MCMC engine, goose provides the :class:`EngineBuilder <liesel.goose.EngineBuilder>`. Please refer to
the linked EngineBuilder documentation to learn how to use it.

A recent addition is the :class:`MCMCSpec <liesel.goose.MCMCSpec>`, which can be passed to the
``inference`` argument of a :class:`model.Var <liesel.model.Var>` upon initialization to tell the variable
directly how it should be sampled. You can then use the method
:meth:`get_engine_builder <liesel.goose.LieselMCMC.get_engine_builder>` of :class:`LieselMCMC <liesel.goose.LieselMCMC>` to
conveniently initialize your :class:`EngineBuilder <liesel.goose.EngineBuilder>`.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :caption: MCMC
   :recursive:

   ~liesel.goose.LieselMCMC
   ~liesel.goose.MCMCSpec
   ~liesel.goose.EngineBuilder
   ~liesel.goose.Engine

.. toctree::
   :maxdepth: 1

   reference/mcmc-kernels
   reference/mcmc-interfaces
   reference/mcmc-advanced

.. _summary-plots:

MCMC results
************

The central classes for handling your sampling results are:

You can obtain your posterior samples as a dictionary via
:meth:`get_posterior_samples <liesel.goose.SamplingResults.get_posterior_samples>`. There is also experimental support
for turning your samples into an ArviZ data object via
:func:`to_arviz_inference_data <liesel.experimental.arviz.to_arviz_inference_data>`.

Goose also comes with a number of plotting functions that give you quick
acccess to important diagnostics.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :caption: MCMC results
   :recursive:

   ~liesel.goose.SamplingResults
   ~liesel.goose.Summary
   ~liesel.goose.SamplesSummary
   ~liesel.goose.loo

.. toctree::
   :maxdepth: 1

   reference/mcmc-plots

.. _optimizer-api:

Optimization
************

Start with the :doc:`optimization` guide. These pages describe the arguments
and defaults.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :caption: Optimization
   :recursive:

   ~liesel.optim.LieselOptim
   ~liesel.optim.OptimEngine
   ~liesel.optim.OptimCheckpoint
   ~liesel.optim.OptimHistory
   ~liesel.optim.OptimResult
   ~liesel.optim.OptimNaNDebugInfo
   ~liesel.optim.Stopper
   ~liesel.optim.EmaTrainLossMonitor
   ~liesel.optim.LossMonitor

.. toctree::
   :maxdepth: 1

   reference/optim-losses
   reference/optim-optimizers
   reference/optim-data
   reference/optim-laplace

.. _experimental-api:

Experimental
************

These integrations have experimental APIs.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :caption: Experimental

   ~liesel.experimental.arviz
   ~liesel.experimental.pymc

.. _legacy-optimization-deprecated:

Deprecated
**********

:func:`liesel.goose.optim_flat` is deprecated and emits a ``FutureWarning``.
Use :class:`liesel.optim.LieselOptim` for new code, including finding starting
values for MCMC. See :doc:`optimizer-migration` for a before-and-after example.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :caption: Deprecated
   :recursive:

   ~liesel.goose.optim_flat
   ~liesel.goose.Stopper
   ~liesel.goose.history_to_df
   ~liesel.goose.OptimResult

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
