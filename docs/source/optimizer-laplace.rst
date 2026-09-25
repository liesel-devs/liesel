Integrate latent parameters with Laplace
============================================

:class:`liesel.optim.LaplaceLoss` integrates selected continuous parameters out
of the model's joint density, then fits the remaining parameters. Here we fit a
Poisson model's mean and random-effect scale while integrating eight group effects.

Build a model and fit the marginal posterior
------------------------------------------------

Use float64 for this example. Enable it before constructing arrays and preserve
it when building the model; the loss itself never changes JAX's precision.

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
coordinate automatically. ``b`` keeps its prior and parameter status in the model.
Fitting leaves the model unchanged. Explicit optimizer lists can select a smaller
outer subset; omitted parameters stay fixed at their model values.

The loss uses the full, unscaled joint density, including priors, transformation
Jacobians, and normalization constants. ``LieselOptim(scale_loss=...)`` does not
rescale a supplied loss. Use full-data batches and ``"train_full_data"`` monitoring;
minibatches, validation monitoring, and EMA monitoring are unsupported here.

Inspect the conditional mode and curvature
----------------------------------------------

.. code-block:: python

   outer = result.position_min_monitor
   state = result.loss_state_min_monitor
   print(outer)
   print(state.latent_position["b"])
   print(state.n_iter, state.newton_decrement_squared / 2)
   latent_factor = state.latent_precision_cholesky
   print(state.latent_names, state.latent_shapes)

``latent_factor @ latent_factor.T`` is the dense conditional precision at this
outer position. Its flattened order follows ``latent_names`` and ``latent_shapes``.
Only ``status == 1`` denotes a successful inner solve. The state's gradient norm
and ``n_resolution_steps`` help inspect convergence and floating-point safeguards.

The best snapshot minimizes recorded monitoring loss. Use ``position_final`` with
``loss_state_final`` for the last completed epoch; keep each position with its own
state. Factors are retained for these snapshots, rather than for every history row.

Control inner warm starts
-----------------------------

The default ``warm_start=True`` starts each inner solve from the last committed
latent mode. The seed stays fixed throughout an outer line search. One successful
full-training evaluation commits state after each epoch, even with several outer
updates. That monitor repeats the selected-point solve; it does not recompute the
outer gradient. Stateful L-BFGS refreshes its gradient when starting a new step.

To compare with starts from the original latent values, supply a new loss to a new fit:

.. code-block:: python

   cold_loss = opt.LaplaceLoss(model, latent=["b"], warm_start=False)

This controls inner initialization. To resume an interrupted fit, use an optimizer
checkpoint instead; see :doc:`optimizer-checkpointing`. Checkpoints preserve the
committed and best latent states and require the same latent coordinates and
inner-solver settings.

Construct joint uncertainty on request
------------------------------------------

.. code-block:: python

   posterior = loss.approximate_joint_posterior(result)  # at="best" by default
   print(posterior.names, posterior.shapes)
   covariance = posterior.covariance()
   draws = posterior.sample(jax.random.key(42), sample_shape=(1000,))
   predicted = model.predict(draws, predict=["tau", "log_rate"])
   tau_draws = predicted["tau"]
   rate_draws = jnp.exp(predicted["log_rate"])

The mean combines the selected outer fit with its conditional latent mode. The
Gaussian combines marginal outer curvature with the local conditional Gaussian,
including cross-correlations and the outer parameters' contribution to latent
uncertainty. Its precision factor orders sorted outer names before sorted latent
names. A dense covariance is only constructed when requested.

Use ``at="final"`` to select the matched final snapshot. Keep the same model,
training data, and loss configuration; do not change fixed parameters between
fitting and this call. Samples use optimizer coordinates. ``Model.predict`` applies
the model's transformations, including the positive ``tau`` scale here. Empty
``sample_shape=()`` returns one draw; ``(chains, draws)`` adds two leading axes.

This optional calculation adds higher implicit derivatives and curvature work
once per call. Ordinary fitting uses first derivatives and reuses the final latent
factor; differentiating the fitting loss twice raises an error. The helper requires
true positive-definite conditional and marginal curvature and checks outer
stationarity using half the squared Newton decrement (default bound ``1e-4``).

Inspect a failure deliberately
----------------------------------

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

A handled numerical failure preserves the last completed valid position and state.
If no valid evaluation exists, the matching state is ``None``. The failed proposal
is separate and cannot become a warm-start seed. Failed fits have no resumable
checkpoint; an earlier saved checkpoint remains available.

The posterior helper raises on failure by default. ``raise_on_failure=False``
consciously requests an invalid diagnostic object with whatever gradients and raw
curvature could be evaluated. Its ``sample`` and ``covariance`` methods raise.
No jitter or eigenvalue clipping silently converts invalid curvature into success.

Choose numerical controls
-----------------------------

``inner_max_iter`` defaults to 100. ``inner_tol`` bounds half the inner squared
Newton decrement on the unscaled objective: ``None`` chooses ``1e-6`` for float32
or ``1e-10`` for float64. For difficult curvature, consider better initial values,
float64, or an explicit tolerance and budget; inspect the diagnostics when doing so.
An early-stopping result alone does not certify posterior stationarity.

The solver finds a local conditional mode. Non-concave problems may select different
modes from warm and cold starts. Dense curvature uses O(d²) storage and O(d³) linear
algebra for d latent coordinates; this version targets a few hundred latents with
a modest outer dimension. It has no sparse or block approximation.
