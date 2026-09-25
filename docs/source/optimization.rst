Optimization
============

Use :class:`liesel.optim.LieselOptim` to fit model parameters or find starting
values for MCMC. It minimizes the negative log posterior: likelihood plus priors.
Without priors, this gives a maximum likelihood fit.

For an existing Liesel ``model``, a full-data fit takes:

.. code-block:: python

   import liesel.optim as opt

   optim = opt.LieselOptim(
       model,
       optimizers="lbfgs",
       loss_monitor="train_full_data",
   )
   result = optim.fit()
   position = result.position_min_monitor

L-BFGS needs the full data and a deterministic loss. For minibatches, use Adam.
The ``optimizers`` argument is required. Pass a configured Optax transformation,
such as ``optimizers=optax.adam(0.01)``, to optimize all parameters with it.
The fitted values are returned separately; fitting does not change your model.

After a joint MAP fit, ``optim.loss.approximate_joint_posterior(result)``
can construct a Gaussian approximation for the optimized parameters. It uses
full-training curvature and checks that the selected position is a stationary
point with positive curvature; a validation or EMA minimum may not qualify.
See :meth:`~liesel.optim.NegLogProbLoss.approximate_joint_posterior` for controls
and :doc:`optimizer-laplace` for a walkthrough of draws and uncertainty in a
hierarchical model.

Start here
----------

.. toctree::
   :maxdepth: 1

   tutorials/notebooks/09-liesel-optim-basic
   Fit two data groups <tutorials/notebooks/10-liesel-optim-advanced>

Common tasks
------------

.. toctree::
   :maxdepth: 1

   optimizer-monitoring
   optimizer-customization
   optimizer-laplace
   optimizer-splitting
   optimizer-batching
   optimizer-loss-scaling
   optimizer-weighted-batching
   optimizer-checkpointing
   optimizer-migration

For arguments and defaults, see the :ref:`optimizer-api`.
``goose.optim_flat`` is deprecated and will be removed in 0.8.0; use the migration guide
above to update existing code.
