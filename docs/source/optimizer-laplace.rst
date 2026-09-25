Integrate latent parameters with Laplace
========================================

:class:`liesel.optim.LaplaceLoss` integrates selected continuous parameters out
of the model's joint density, then fits the remaining parameters. Here we fit a
Poisson model's mean and random-effect scale while integrating eight group effects.

Build a model and fit the marginal posterior
--------------------------------------------

Use float64 for this example: enable it before creating arrays and pass
``to_float32=False`` to the model.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import tensorflow_probability.substrates.jax.bijectors as tfb
   import tensorflow_probability.substrates.jax.distributions as tfd
   import liesel.model as lsl
   import liesel.optim as opt

   jax.config.update("jax_enable_x64", True)
   counts = jnp.array([
       [1., 2., 1., 0.], [3., 2., 4., 3.], [6., 5., 4., 5.], [9., 7., 10., 8.],
       [13., 16., 12., 15.], [2., 4., 3., 2.], [5., 8., 6., 7.], [20., 23., 19., 21.],
   ])
   group = jnp.repeat(jnp.arange(8), 4)
   mu = lsl.Var.new_param(1., lsl.Dist(tfd.Normal, 0., 2.), name="mu")
   tau = lsl.Var.new_param(
       .5, lsl.Dist(tfd.LogNormal, jnp.log(.5), .7),
       bijector=tfb.Exp(), name="tau",
   )
   b = lsl.Var.new_param(jnp.zeros(8), lsl.Dist(tfd.Normal, 0., tau), name="b")
   log_rate = lsl.Var.new_calc(
       lambda mu, b: mu + b[group], mu, b, name="log_rate"
   )
   y = lsl.Var.new_obs(
       counts.ravel(), lsl.Dist(tfd.Poisson, log_rate=log_rate), name="y"
   )
   model = lsl.Model([y], to_float32=False)
   loss = opt.LaplaceLoss(model, latent=["b"])
   result = opt.LieselOptim(
       model, loss=loss, optimizers="lbfgs", loss_monitor="train_full_data",
       stopper=opt.Stopper(epochs=60, patience=10, rtol=1e-10),
       show_progress=False,
   ).fit()
   print(result.status)

The outer optimizer selects ``mu`` and the unconstrained ``tau_transformed``
coordinate. ``b`` keeps its prior. Fitting leaves the model unchanged.

Use full-data batches and ``"train_full_data"`` monitoring. The loss includes
priors, Jacobians, and normalization constants without rescaling;
see :doc:`optimizer-loss-scaling`.

Inspect the conditional mode and curvature
------------------------------------------

.. code-block:: python

   outer = result.position_min_monitor
   state = result.loss_state_min_monitor
   print(outer)
   print(state.latent_position["b"])
   print(state.n_iter, state.newton_decrement_squared / 2)
   latent_factor = state.latent_precision_cholesky
   print(state.latent_names, state.latent_shapes)

``latent_factor @ latent_factor.T`` is the conditional precision, ordered by
``latent_names`` and ``latent_shapes``. ``status == 1`` means the inner solve
succeeded. See :class:`~liesel.optim.LaplaceState` for the other diagnostics.

The best snapshot minimizes monitoring loss. For the last completed epoch, use
``position_final`` with its matching ``loss_state_final``.

Control inner warm starts
-------------------------

By default, each inner solve starts from the latent mode committed at the end of
the previous epoch. Set ``warm_start=False`` to start from the model's latent values:

.. code-block:: python

   cold_loss = opt.LaplaceLoss(model, latent=["b"], warm_start=False)

Pass ``cold_loss`` to a new fit to compare. Resuming an interrupted fit is a separate
operation: :doc:`optimizer-checkpointing` explains how to resume with saved states.

Construct joint uncertainty on request
--------------------------------------

.. code-block:: python

   posterior = loss.approximate_joint_posterior(result)  # at="best" by default
   print(posterior.names, posterior.shapes)
   covariance = posterior.covariance()
   draws = posterior.sample(jax.random.key(42), sample_shape=(1000,))
   predicted = model.predict(draws, predict=["tau", "log_rate"])
   tau_draws = predicted["tau"]
   rate_draws = jnp.exp(predicted["log_rate"])

The joint Gaussian includes outer uncertainty, conditional latent uncertainty, and
their cross-correlations. Its precision factor orders sorted outer names before
sorted latent names. A dense covariance is only constructed when requested.

Use ``at="final"`` for the final snapshot. Keep the same model, data, and loss
configuration; do not change fixed parameters between fitting and this call.
``Model.predict`` transforms draws from optimizer coordinates to the positive
``tau`` scale here. ``sample_shape=()`` gives one draw; ``(chains, draws)`` adds two axes.

The helper costs extra curvature work once per call and none during fitting. It
requires positive-definite curvature and an approximately stationary fit
(``stationarity_tol``, default ``1e-4``). Use this helper rather than differentiating
the loss twice.

Inspect a failure deliberately
------------------------------

An insufficient inner budget makes a useful diagnostic example:

.. code-block:: python

   short_loss = opt.LaplaceLoss(model, latent=["b"], inner_max_iter=1)
   failed = opt.LieselOptim(
       model, loss=short_loss, optimizers="lbfgs",
       loss_monitor="train_full_data", show_progress=False,
   ).fit()
   print(failed.status, failed.failure_reason)
   print(failed.failed_loss_state)
   diagnostic = short_loss.approximate_joint_posterior(
       failed, raise_on_failure=False
   )
   print(diagnostic.valid, diagnostic.diagnostics["reason"])

A numerical failure preserves the last completed valid position and state; if none
exists, the matching state is ``None``. Failed fits have no resumable checkpoint;
an earlier saved checkpoint remains available.

The posterior helper raises on failure by default. ``raise_on_failure=False`` returns
an invalid diagnostic object with available gradients, curvature, and the failure
reason. Its ``sample`` and ``covariance`` methods raise.

Choose numerical controls
-------------------------

``inner_max_iter`` defaults to 100. ``inner_tol`` bounds half the inner squared
Newton decrement: ``None`` chooses ``1e-6`` for float32 or ``1e-10`` for float64.
For difficult curvature, consider better initial values, float64, or an explicit
tolerance and budget; inspect the diagnostics when doing so.

The solver finds a local mode; warm and cold starts can find different modes.
Dense curvature uses O(d²) storage and O(d³) linear algebra for d latent coordinates.
This version targets a few hundred latents with a modest outer dimension.
