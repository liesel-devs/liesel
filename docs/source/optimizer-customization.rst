Choose optimizers
=================

:class:`liesel.optim.LieselOptim` uses Adam with learning rate ``0.02`` by default.
Pass ``optimizers="lbfgs"``
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

Control history memory
----------------------

Disable parameter history when you only need the final and best positions:

.. code-block:: python

   result = opt.LieselOptim(
       model, loss_monitor="train_full_data", save_position_history=False
   ).fit()

History reserves memory for the maximum epoch budget before fitting, even when
early stopping ends the run sooner. Its parameter storage costs approximately
``epochs * total_parameter_bytes``: one million float32 parameters across 1,000
epochs take about 4 GB. Disabling it retains scalar losses and final/best positions.

Use ``build_engine()`` for settings beyond the wrapper's arguments.
See :class:`~liesel.optim.OptimEngine` for all settings, or
:doc:`optimizer-checkpointing` to pause and resume a fit.
