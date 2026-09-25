Monitoring and early stopping
=============================

``loss_monitor`` chooses the loss used for early stopping and for saving the
best fit. Choose it when creating :class:`liesel.optim.LieselOptim`.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Setting
     - Use it when
     - What is measured
   * - ``"validation"``
     - You have held-out validation data.
     - Full validation loss after each epoch.
   * - ``"train_full_data"``
     - You want to check the full training objective.
     - Full training loss after each epoch.
   * - ``opt.EmaTrainLossMonitor(effective_window=2.0)``
     - Full-data checks are too expensive.
     - A running average of minibatch training losses.

An epoch runs all configured batches. The two full-data monitors each add one
loss evaluation at the end of every epoch. Validation uses likelihood only by
default; set ``validation_strategy="log_prob"`` to include priors.

:class:`.LieselVI` accepts the EMA and full-training monitors, but no validation
monitor. Its full-training ELBO still uses variational draws and can fluctuate;
see :doc:`variational-inference`.

Choose when to stop
-------------------

.. code-block:: python

   import optax

   import liesel.optim as opt

   stopper = opt.Stopper(epochs=500, patience=20, rtol=1e-4)

This allows at most 500 epochs. Within the last 20 epochs, fitting stops when
there is no worthwhile improvement over the oldest loss in that window.
``atol`` measures absolute improvement; ``rtol`` measures relative improvement.
See :class:`~liesel.optim.Stopper` for the exact rule.

Read the result
---------------

* ``result.position_min_monitor`` holds the parameters saved at the epoch with
  the lowest finite monitoring loss. It raises ``RuntimeError`` if no such loss
  was recorded. An earlier best position remains available after a later failure.
* ``result.position_final`` holds the parameters at the end of the run.
* ``result.plot_loss_overview()`` shows the full loss history and a closer
  view of recent epochs.
* ``result.plot_params()`` shows saved parameter paths.

Both position properties raise ``RuntimeError`` if their parameters contain NaN
or infinity. History, status, and diagnostics remain available for inspection.
See :ref:`optimizer-debug-nans` to capture information about a NaN failure.

The training curve averages the losses seen before each update. Parameters
change during an epoch, so this curve is not the full training loss at its end.
Use ``result.history.loss_df()`` to inspect the recorded values.

Smooth minibatch losses
-----------------------

For an existing ``model``:

.. code-block:: python

   split = opt.PositionSplit.from_model(model)
   batches = opt.Batches.from_split(split, batch_size=32)
   result = opt.LieselOptim(
       model,
       optimizers=optax.adam(0.01),
       split=split,
       batches=batches,
       loss_monitor=opt.EmaTrainLossMonitor(effective_window=2.0),
       stopper=stopper,
       seed=42,
   ).fit()
   result.plot_loss_overview()

A larger ``effective_window`` smooths more and reacts more slowly. Its unit is
an epoch's worth of batches. Older losses fade gradually; they are not dropped
at a fixed age. The average continues across epochs and reuses losses already
computed for optimizer updates.

An EMA combines losses from several parameter positions. Its best saved position
is the snapshot at the end of that epoch, not a position whose exact loss equals
the plotted average. See :class:`~liesel.optim.EmaTrainLossMonitor` for the formula
and the alternative :meth:`~liesel.optim.EmaTrainLossMonitor.from_half_life` setting.

.. _optimizer-debug-nans:

Investigate a NaN failure
-------------------------

For a configured ``LieselOptim`` builder, enable first-NaN reproduction capture
on its engine before fitting:

.. code-block:: python

   engine = builder.build_engine()
   engine.debug_nans = True
   result = engine.fit()
   debug_info = result.nan_debug

If a NaN is detected, ``debug_info`` contains information for reproducing it;
otherwise it is ``None``. See :class:`~liesel.optim.OptimNaNDebugInfo` for its
contents. Enable ``debug_nans`` on the engine returned by ``build_engine()``.
The captured information helps investigate the failure; it does not correct
poor starting values.
