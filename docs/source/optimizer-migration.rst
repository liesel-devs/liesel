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

* Wrap the old ``params`` and Optax optimizer in :class:`liesel.optim.Optimizer`.
  This preserves the selected parameter subset. The default builder instead
  optimizes all model parameters with Adam at learning rate ``0.001``; the old
  default learning rate was ``0.01``.
* Replace ``gs.Stopper(max_iter=...)`` with ``opt.Stopper(epochs=...)``. An epoch
  runs all configured batches. The old history included the initial position at
  iteration zero; the new history records completed epochs. Budgets and early
  stopping therefore need not produce identical runs.
* Choose ``loss_monitor`` explicitly. ``"train_full_data"`` evaluates the full
  training objective after each epoch, matching the old monitoring choice when
  no separate validation model was supplied.
* Keep ``scale_loss=False`` to retain the old default objective scale. The new
  builder otherwise divides the loss by the training sample size. Set tolerances
  explicitly when migrating; the stopper defaults also differ.
* Use ``position_final`` to replace ``restore_best_position=False``. Use
  ``position_min_monitor`` for the best recorded monitoring loss across the run;
  the old ``restore_best_position=True`` selected within the final patience
  window. Rebuild a model state with ``model.update_state`` as shown above.
* Use ``result.plot_loss()`` or ``result.history.loss_df()`` instead of
  ``gs.history_to_df(result.history)``.

Validation and minibatches
--------------------------

Replace ``model_validation`` with a :class:`liesel.optim.PositionSplit` holding
training and validation data, and select ``loss_monitor="validation"``. Include
responses and their matching covariates in the split. To retain an existing
holdout, construct the split from those data rather than drawing a new split.
The new default validation objective uses likelihood only; set
``validation_strategy="log_prob"`` to include priors as ``optim_flat`` did.

``batch_size`` remains available; ``batch_seed`` becomes ``seed``. See the
:doc:`basic tutorial <tutorials/notebooks/09-liesel-optim-basic>` for a complete
validation and minibatch example.
