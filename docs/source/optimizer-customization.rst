Choose optimizers
=================

:class:`liesel.optim.LieselOptim` requires an explicit optimizer. Pass a
configured Optax transformation, such as ``optimizers=optax.adam(0.01)``, to use
it for all parameters, or ``optimizers="lbfgs"`` for a full-data, deterministic
fit. L-BFGS cannot use minibatches.

Set the learning rate
---------------------

For an existing ``model``, pass a configured Optax optimizer directly:

.. code-block:: python

   import optax

   import liesel.optim as opt

   schedule = optax.exponential_decay(
       init_value=0.01, transition_steps=100, decay_rate=0.9
   )
   result = opt.LieselOptim(
       model,
       optimizers=optax.adam(schedule),
       loss_monitor="train_full_data",
   ).fit()

The schedule advances on optimizer updates, not epochs. With minibatches, it can
advance several times per epoch. For a constant rate, use ``optax.adam(0.01)``.
Minibatch fits can retain optimization noise. For precise point estimates,
check convergence and consider a smaller rate or a final full-data fit.

Choose parameters
-----------------

For the ``beta`` and ``log_sigma`` parameters in the basic tutorial:

.. code-block:: python

   optimizers = [
       opt.Optimizer(["beta"], optax.adam(0.01)),
       opt.Optimizer(["log_sigma"], optax.adam(0.001)),
   ]

Pass this list as ``optimizers=optimizers`` to ``LieselOptim``. Each optimizer
updates its own parameters in list order on every batch. Their parameter names
must not overlap. Parameters left out of the list stay fixed.
Separate blocks each evaluate the loss and gradient, adding work.

L-BFGS must be the only optimizer because other parameter updates invalidate its
cached objective and curvature history. Use ``optimizers="lbfgs"`` for all model
parameters, or ``optimizers=[opt.LBFGS(["beta", "log_sigma"])]`` for a selected
subset.

Choose the source
-----------------

A prior may be attached to a weak parameter computed from a strong source variable.
That prior remains in the default loss, including its derivatives through the weak
parameter. Automatic parameter selection cannot decide which source to estimate
and raises an informative error. Supply the strong names explicitly, for example
``optimizers=[opt.Optimizer(["source"], optax.adam(0.01))]`` or
``optimizers=[opt.LBFGS(["source"])]``. The source does not need to be marked as a
parameter. The weak parameter itself is recomputed during fitting.

Switch to L-BFGS
----------------

Use two separate fits to switch from Adam to L-BFGS. The optimizer option
``activate_after_epochs`` delays activation; it does not deactivate Adam.
For an existing ``model`` with the same parameter names in both fits:

.. code-block:: python

   first = opt.LieselOptim(
       model, optimizers=optax.adam(0.01), loss_monitor="train_full_data"
   ).fit()
   model.state = model.update_state(first.position_final, model.state)
   result = opt.LieselOptim(
       model, optimizers="lbfgs", loss_monitor="train_full_data"
   ).fit()

Assign the returned model state before constructing the second fit:
``update_state`` leaves the model unchanged by default. L-BFGS starts with fresh
optimizer state at Adam's final parameter values and uses full-data batches here.
Do not pass Adam's checkpoint to the L-BFGS fit.

Save less history
-----------------

Disable parameter history when you only need the final and best positions:

.. code-block:: python

   result = opt.LieselOptim(
       model,
       optimizers=optax.adam(0.01),
       loss_monitor="train_full_data",
       save_position_history=False,
   ).fit()

History reserves memory for the maximum epoch budget before fitting, even when
early stopping ends the run sooner. Its parameter storage costs approximately
``epochs * total_parameter_bytes``: one million float32 parameters across 1,000
epochs take about 4 GB. Disabling it retains scalar losses and final/best positions.

Use ``build_engine()`` for settings beyond the wrapper's arguments.
See :class:`~liesel.optim.OptimEngine` for all settings, or
:doc:`optimizer-checkpointing` to pause and resume a fit.
