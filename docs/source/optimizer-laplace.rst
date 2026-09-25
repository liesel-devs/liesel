Integrate latent parameters with Laplace
========================================

:class:`liesel.optim.LaplaceLoss` integrates selected continuous parameters out
of the model's joint density, then fits the remaining parameters. Here we fit a
Poisson model's mean and random-effect scale while integrating eight group effects.

Build and inspect the model
---------------------------

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

   counts = jnp.array(
       [
           [1.0, 2.0, 1.0, 0.0],
           [3.0, 2.0, 4.0, 3.0],
           [6.0, 5.0, 4.0, 5.0],
           [9.0, 7.0, 10.0, 8.0],
           [13.0, 16.0, 12.0, 15.0],
           [2.0, 4.0, 3.0, 2.0],
           [5.0, 8.0, 6.0, 7.0],
           [20.0, 23.0, 19.0, 21.0],
       ]
   )
   group = jnp.repeat(jnp.arange(8), 4)

   mu = lsl.Var.new_param(1.0, lsl.Dist(tfd.Normal, 0.0, 2.0), name="mu")
   tau = lsl.Var.new_param(
       0.5,
       lsl.Dist(tfd.LogNormal, jnp.log(0.5), 0.7),
       bijector=tfb.Exp(),
       name="tau",
   )
   b = lsl.Var.new_param(jnp.zeros(8), lsl.Dist(tfd.Normal, 0.0, tau), name="b")
   log_rate = lsl.Var.new_calc(lambda mu, b: mu + b[group], mu, b, name="log_rate")
   y = lsl.Var.new_obs(counts.ravel(), lsl.Dist(tfd.Poisson, log_rate=log_rate), name="y")

   model = lsl.Model([y], to_float32=False)
   model.plot(width=8, height=6)

.. figure:: _static/optimizer-laplace-model.png
   :alt: Model graph connecting the mean mu and group effects b to Poisson observations y, with tau controlling the group-effect scale.

   The group effects ``b`` enter the log rate; ``tau`` controls their prior scale.

Fit the marginal posterior
--------------------------

.. code-block:: python

   loss = opt.LaplaceLoss(model, latent=["b"])
   stopper = opt.Stopper(epochs=60, patience=10, rtol=1e-10)
   result = opt.LieselOptim(
       model,
       loss=loss,
       optimizers="lbfgs",
       loss_monitor="train_full_data",
       stopper=stopper,
       show_progress=False,
   ).fit()

.. code-block:: pycon

   >>> result.status
   'early_stopping'

Use full-data batches and ``"train_full_data"`` monitoring. The outer coordinates
are ``mu`` and ``h(tau)``; ``b`` keeps its prior. The unscaled loss includes priors,
Jacobians, and normalization constants; see :doc:`optimizer-loss-scaling`.

Inspect the conditional mode and curvature
------------------------------------------

.. code-block:: pycon

   >>> outer = result.position_min_monitor
   >>> state = result.loss_state_min_monitor
   >>> print(f"mu = {outer['mu']:.3f}, log(tau) = {outer['h(tau)']:.3f}")
   mu = 1.679, log(tau) = -0.229
   >>> print(state.latent_position["b"].round(3))
   [-1.272 -0.514 -0.064  0.441  0.934 -0.586  0.182  1.328]
   >>> int(state.status), int(state.n_iter)
   (1, 1)
   >>> float(state.newton_decrement_squared / 2) < loss.inner_tol
   True
   >>> latent_factor = state.latent_precision_cholesky
   >>> state.latent_names, state.latent_shapes, latent_factor.shape
   (('b',), ((8,),), (8, 8))

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

.. code-block:: pycon

   >>> posterior = loss.approximate_joint_posterior(result)
   >>> posterior.names, posterior.shapes
   (('h(tau)', 'mu', 'b'), ((), (), (8,)))
   >>> covariance = posterior.covariance()
   >>> covariance.shape
   (10, 10)
   >>> draws = posterior.sample(jax.random.key(42), sample_shape=(1000,))
   >>> predicted = model.predict(draws, predict=["tau", "log_rate"])
   >>> tau_draws = predicted["tau"]
   >>> rate_draws = jnp.exp(predicted["log_rate"])
   >>> tau_draws.shape, rate_draws.shape
   ((1000,), (1000, 32))

The joint Gaussian includes outer and latent uncertainty and their correlations.
A dense covariance is only constructed when requested.

The default is ``at="best"``; use ``at="final"`` for the final snapshot.
Keep the same model, data, and loss configuration; do not change fixed parameters
between fitting and this call.
``Model.predict`` transforms optimizer-coordinate draws to the positive ``tau`` scale.

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
       model,
       loss=short_loss,
       optimizers="lbfgs",
       loss_monitor="train_full_data",
       show_progress=False,
   ).fit()

.. code-block:: pycon

   >>> failed.status
   'numerical_failure'
   >>> failed.failure_reason
   'Inner Laplace optimization failed: iteration limit.'
   >>> int(failed.failed_loss_state.status), int(failed.failed_loss_state.n_iter)
   (2, 1)
   >>> diagnostic = short_loss.approximate_joint_posterior(
   ...     failed, raise_on_failure=False
   ... )
   >>> diagnostic.valid
   False

Failures preserve the last valid snapshot; without one, the loss state is ``None``.
Failed fits cannot resume; earlier saved checkpoints remain available.

The posterior helper raises on failure by default. ``raise_on_failure=False`` returns
an invalid diagnostic object with available gradients, curvature, and the failure
reason. Its ``sample`` and ``covariance`` methods raise.

Choose numerical controls
-------------------------

``inner_max_iter`` defaults to 100. ``inner_tol`` bounds half the inner squared
Newton decrement: ``None`` chooses ``1e-6`` for float32 or ``1e-10`` for float64.

The solver finds a local mode; warm and cold starts can find different modes.
Dense curvature uses O(d²) storage and O(d³) linear algebra for d latent coordinates.
This version targets a few hundred latents with a modest outer dimension.
