Migrating from optim_flat
=========================

:func:`liesel.goose.optim_flat` is deprecated since version 0.8.0 and emits a
``FutureWarning`` when called. Use :class:`liesel.optim.LieselOptim` instead.

Full-data optimization
----------------------

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
   model = lsl.Model([y])

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

   result = opt.LieselOptim(
       model,
       optimizers=[opt.Optimizer(["loc"], optax.adam(0.01))],
       stopper=opt.Stopper(epochs=1000, patience=20, atol=0.001),
       loss_monitor="train_full_data",
       scale_loss=False,
       show_progress=False,
   ).fit()
   position = result.position_min_monitor
   fitted_state = model.update_state(position, model.state)

``fitted_state`` can be passed to
:meth:`liesel.goose.EngineBuilder.set_initial_values` for an MCMC warm start.
Optimization does not assign the fitted state to the original model.

What changes
------------

* Wrap ``params`` and the Optax optimizer in :class:`liesel.optim.Optimizer`.
  Without this, the builder fits all parameters with Adam at learning rate
  ``0.02``. The old default rate was ``0.01``.
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

Validation and minibatches
--------------------------

Use :class:`liesel.optim.PositionSplit` instead of ``model_validation`` and set
``loss_monitor="validation"``. Keep responses and matching covariates together.
Build the split from your existing holdout to keep the same evaluation data.
Validation now uses likelihood only; set ``validation_strategy="log_prob"`` to
include priors as before.

``batch_size`` remains available; ``batch_seed`` becomes ``seed``. See the
:doc:`basic tutorial <tutorials/notebooks/09-liesel-optim-basic>` for a complete
validation and minibatch example.
