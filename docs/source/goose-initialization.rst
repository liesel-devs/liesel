Choose initial values
=====================

Goose starts from the model's current state. Choose values with finite log
density on the correct support, then disperse the chains enough to reveal
different behavior.

Check the starting model
------------------------

For an existing Liesel ``model``:

.. code-block:: python

   print(model.diagnose())

Inspect values and log probabilities for NaN or infinity. For positive
parameters sampled with NUTS, use a suitable
:doc:`transformation <tutorials/md/01c-transform>` and initialize on its
sampling scale. Finite density is necessary but does not ensure finite gradients
or good mixing.

Disperse the chains
-------------------

For the regression ``model`` from :doc:`tutorials/md/01c-transform`:

.. code-block:: python

   import tensorflow_probability.substrates.jax.distributions as tfd

   import liesel.goose as gs

   joint = gs.MCMCSpec(
       gs.NUTSKernel,
       kernel_group="regression",
       jitter_dist=tfd.Normal(0.0, 0.2),
       jitter_method="additive",
   )
   model.vars["beta"].inference = joint
   model.vars["log_sigma_sq"].inference = joint
   results = gs.LieselMCMC(model).run_for_epochs(
       seed=2026, num_chains=4, adaptation=1000, posterior=1000,
   )

Each chain receives its own perturbation. The scalar jitter distribution draws
one perturbation per element of a vector parameter. A distribution with batch
or event dimensions must match the parameter's shape.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``jitter_method``
     - Starting value
     - Check
   * - ``"additive"``
     - Current value plus a draw.
     - The perturbation must respect support; a log parameter is often convenient.
   * - ``"multiplicative"``
     - Current value times a draw.
     - Use positive draws to keep a positive parameter positive.
   * - ``"replacement"``
     - The draw replaces the current value.
     - Choose a distribution with appropriate support and scale.

``apply_jitter=True`` applies these configured distributions. It does not
create random starts when ``jitter_dist`` is absent. Set ``apply_jitter=False``
to use the existing starts without their configured perturbations.

Start near an optimized fit
---------------------------

For a model prepared for unconstrained optimization, fit first and then update
its state before building the sampler:

.. code-block:: python

   import liesel.optim as opt

   fit = opt.LieselOptim(
       model, optimizers="lbfgs", loss_monitor="train_full_data",
   ).fit()
   model.state = model.update_state(fit.position_min_monitor, model.state)

Fitting returns values separately; the assignment makes them the model's
starting state. The next ``LieselMCMC(model).run_for_epochs(...)`` call starts
there and applies any configured jitter. See :doc:`optimization` for fitting
choices. A mode can be a useful start, but it does not establish exploration
of the posterior or reveal other modes.

For individually supplied chain states, use
:meth:`~liesel.goose.EngineBuilder.set_initial_values` with
``multiple_chains=True``. Each state leaf then needs a leading chain axis.
Record starts and jitter settings along with the
:doc:`seed and environment <tutorials/md/05-reproducibility>`.
