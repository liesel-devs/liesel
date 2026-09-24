Work with posterior draws
=========================

The examples use the regression ``model`` and checked ``results`` from
:doc:`tutorials/md/01c-transform`.

Extract the posterior
---------------------

.. code-block:: python

   import liesel.goose as gs

   samples = results.get_posterior_samples()
   print(samples["beta"].shape)
   print(samples["sigma_sq"].shape)

The shapes are ``(4, 1000, 2)`` and ``(4, 1000)`` in the tutorial:
chains, draws, then any parameter dimensions. Preserve the first two axes
when computing diagnostics. ``get_samples()`` also includes initial values
and warmup; use ``get_posterior_samples()`` for posterior inference.

Choose what to record
---------------------

Goose records variables updated by kernels. Add a derived quantity through
``positions_included`` when running the sampler:

.. code-block:: python

   recorded = gs.LieselMCMC(model).run_for_epochs(
       seed=2026, num_chains=4, adaptation=1000, posterior=1000,
       positions_included=["sigma_sq"],
   )

Here NUTS samples ``log_sigma_sq``; ``sigma_sq`` is its positive transformation.
``positions_excluded`` overrides included positions and affects storage only.
It does not remove a kernel. Excluding sampled parameters can prevent later
predictions or diagnostics that need those draws.

For large intermediate arrays, retain the underlying parameter draws and
recompute quantities as needed instead:

.. code-block:: python

   variance = model.predict(samples, predict=["sigma_sq"])
   print(gs.SamplesSummary(variance).to_dataframe())

Predict a mean or simulate responses
------------------------------------

For the tutorial's design matrix variable ``X``, construct a grid with the
same two columns: intercept and covariate.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   x_grid = jnp.linspace(0.0, 1.0, 50)
   X_grid = jnp.column_stack([jnp.ones_like(x_grid), x_grid])
   mean = model.predict(samples, predict=["mu"], newdata={"X": X_grid})
   print(gs.SamplesSummary(mean).to_dataframe().head())

``mean["mu"]`` contains the regression mean at each grid point for every draw.
Its pointwise credible intervals describe uncertainty in the mean.

To include observation noise, simulate responses conditional on those draws:

.. code-block:: python

   parameter_draws = {name: samples[name] for name in model.parameters}
   predictive = model.sample(
       shape=(),
       seed=jax.random.key(2027),
       posterior_samples=parameter_draws,
       newdata={"X": X_grid, "y": jnp.zeros_like(x_grid)},
   )
   y_predictive = predictive["y"]
   print(y_predictive.shape)

Pass draws for the writable parameters to ``model.sample``. The recorded
``sigma_sq`` is a derived value in this model; it is recomputed from
``log_sigma_sq`` rather than set directly.

The placeholder ``y`` gives the response the new grid length while the model
updates its state. These values are replaced by simulated responses; they are
not additional observations used for fitting.

With ``shape=()``, there is one simulated response vector per posterior draw.
Predictive intervals also include observation variation and are generally
wider. See :meth:`~liesel.model.Model.predict` and
:meth:`~liesel.model.Model.sample` for new-data and shape rules. Additional
sample dimensions, if requested, precede the posterior chain/draw dimensions.

Save a finished run
-------------------

.. code-block:: python

   results.pkl_save("regression-samples.pkl")
   restored = gs.SamplingResults.pkl_load("regression-samples.pkl")

Load only trusted pickle files. Keep the model definition, data, and
:doc:`environment information <tutorials/md/05-reproducibility>` with the result.

Alternatively, pass ``save_path="regression-samples.pkl"`` to
``run_for_epochs``. If that path already exists, Goose **loads it and skips
sampling**, even when you changed the model, seed, or sampling settings.
Use a fresh path for a new fit. A saved ``SamplingResults`` object contains
results; it is not an engine checkpoint for resuming a chain.
