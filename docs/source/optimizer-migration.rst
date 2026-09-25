Migrate optim_flat
==================

:func:`liesel.goose.optim_flat` is deprecated since version 0.8.0 and emits a
``FutureWarning`` when called. Use :class:`liesel.optim.LieselOptim` instead.

Fit the full data
-----------------

Both examples use the same model and optimize only ``loc``:

.. code-block:: python

   import jax.numpy as jnp
   import optax
   import tensorflow_probability.substrates.jax.distributions as tfd

   import liesel.model as lsl

   loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
   y = lsl.Var.new_obs(
       jnp.array([1.0, 2.0, 3.0]),
       lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
       name="y",
   )
   model = lsl.Model(y)

Before (deprecated):

.. code-block:: python

   import liesel.goose as gs

   result = gs.optim_flat(
       model,
       params=["loc"],
       optimizer=optax.adam(0.01),
       stopper=gs.Stopper(max_iter=1000, patience=20, atol=0.001),
       scale_loss=False,
       progress_bar=False,
   )
   position = result.position
   fitted_state = result.model_state

After:

.. code-block:: python

   import liesel.optim as opt

   engine = opt.LieselOptim(
       model,
       optimizers=[opt.Optimizer(["loc"], optax.adam(0.01))],
       stopper=opt.Stopper(epochs=1000, patience=20, atol=0.001),
       loss_monitor="train_full_data",
       scale_loss=False,
       show_progress=False,
       save_position_history=True,
   ).build_engine()
   result = engine.fit()
   position = result.position_min_monitor
   fitted_state = model.update_state(position, model.state)

``fitted_state`` can be passed to
:meth:`liesel.goose.EngineBuilder.set_initial_values` for an MCMC warm start.
Optimization does not assign the fitted state to the original model.

What changes
------------

* To optimize only selected parameters (the old ``params=``), wrap the
  transformation in ``opt.Optimizer(params, transformation)`` as above. A bare
  Optax transformation optimizes all model parameters.
* ``LieselOptim`` requires ``optimizers``. Pass a configured Optax transformation
  for all parameters, ``"lbfgs"``, or explicit per-parameter optimizers.
* Replace ``batch_seed`` with ``seed``. The new default, ``seed=0``, is
  deterministic; ``seed=None`` uses the current time. A separately constructed
  split has its own seed.
* Replace ``gs.Stopper(max_iter=...)`` with ``opt.Stopper(epochs=...)``. An epoch
  runs all configured batches. The new history starts after the first epoch;
  the old history started before any updates. Runs need not stop at the same time.
* Choose ``loss_monitor`` explicitly. ``"train_full_data"`` evaluates the full
  training objective after each epoch, as before when no validation model was used.
* Keep ``scale_loss=False`` to retain the old default objective scale. The new
  builder otherwise divides by the training sample size. Set stopping tolerances
  explicitly too; their defaults differ.
* Use ``position_final`` to replace ``restore_best_position=False``. Use
  ``position_min_monitor`` for the best monitoring loss across the run. The old
  default selected within the final patience window. Use ``model.update_state``
  to get a fitted model state.
* Use ``result.plot_loss_overview()`` or ``result.history.loss_df()`` instead of
  ``gs.history_to_df(result.history)``.

Split and batch data
--------------------

Models with custom aggregate likelihood, prior, or probability nodes require a
custom :class:`~liesel.optim.Loss`. The built-in loss supports only the standard
sum of observed likelihoods and parameter priors. A manual split alone does not
replace ``optim_flat``'s handling of custom aggregate objectives or its optional
decomposition-check override.

Use :class:`liesel.optim.PositionSplit` instead of ``model_validation`` and set
``loss_monitor="validation"``. Keep responses and matching covariates together.
Build the split from your existing holdout to keep the same evaluation data.
Validation now uses likelihood only; set ``validation_strategy="log_prob"`` to
include priors as before.

``batch_size`` remains available. See the
:doc:`basic tutorial <tutorials/notebooks/09-liesel-optim-basic>` for a complete
validation and minibatch example.

Track model outputs
-------------------

``optim_flat(track_keys=...)`` has no direct argument replacement. Set
``save_position_history=True`` and evaluate deterministic model quantities from
the saved parameter history after fitting. For the example above, this records
the scalar total log likelihood and the array of individual log likelihoods:

.. code-block:: python

   import jax

   track_keys = ["_model_log_lik", "y_log_prob"]
   history = {
       key: values[:result.n_epochs]
       for key, values in result.history.position.items()
   }


   def extract_quantities(position):
       state = model.update_state(position | engine.split.train, model.state)
       return model.extract_position(track_keys, state)


   derived_history = jax.vmap(extract_quantities)(history)

Explicitly supplying ``engine.split.train`` makes data-dependent quantities refer
to complete training data, even after a minibatch fit. Slicing to
``result.n_epochs`` excludes unused history entries when history pruning is
disabled. This reconstructs deterministic model quantities at each saved epoch
position; it cannot reconstruct transient optimizer internals or past stochastic
draws.
