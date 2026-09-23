Choose optimizers
=================

:class:`liesel.optim.LieselOptim` uses Adam by default. Pass ``optimizers="lbfgs"``
for a full-data, deterministic fit. L-BFGS cannot use minibatches.

Set a learning rate or schedule
-------------------------------

For an existing ``model``, wrap an Optax optimizer in
:class:`liesel.optim.Optimizer`:

.. code-block:: python

   import optax
   import liesel.optim as opt

   schedule = optax.exponential_decay(
       init_value=0.01, transition_steps=100, decay_rate=0.9
   )
   optimizer = opt.Optimizer(list(model.parameters), optax.adam(schedule))
   result = opt.LieselOptim(
       model,
       optimizers=[optimizer],
       loss_monitor="train_full_data",
   ).fit()

The schedule advances on optimizer updates, not epochs. With minibatches, it can
advance several times per epoch. For a constant rate, use ``optax.adam(0.01)``.

Use different optimizers for different parameters
-------------------------------------------------

For the ``beta`` and ``log_sigma`` parameters in the basic tutorial:

.. code-block:: python

   optimizers = [
       opt.Optimizer(["beta"], optax.adam(0.01)),
       opt.Optimizer(["log_sigma"], optax.adam(0.001)),
   ]

Pass this list as ``optimizers=optimizers`` to ``LieselOptim``. Each optimizer
updates its own parameters in list order on every batch. Their parameter names
must not overlap. Parameters left out of the list stay fixed.

Change engine settings
----------------------

Use ``build_engine()`` when you need settings beyond the builder's arguments:

.. code-block:: python

   engine = opt.LieselOptim(
       model, loss_monitor="train_full_data"
   ).build_engine()
   engine.save_position_history = False
   result = engine.fit()

This saves memory by skipping parameter paths; the final and best positions are
still available. See :class:`~liesel.optim.OptimEngine` for all settings, or
:doc:`optimizer-checkpointing` to pause and resume a fit.
