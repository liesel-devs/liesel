Optimization
============

Use :class:`liesel.optim.LieselOptim` to fit model parameters or find starting
values for MCMC. It minimizes the negative log posterior: likelihood plus priors.
Without priors, this gives a maximum likelihood fit.

For an existing Liesel ``model``, a full-data fit takes:

.. code-block:: python

   import liesel.optim as opt

   result = opt.LieselOptim(
       model,
       optimizers="lbfgs",
       loss_monitor="train_full_data",
   ).fit()
   position = result.position_min_monitor

L-BFGS needs the full data and a deterministic loss. For minibatches, use Adam.
The fitted values are returned separately; fitting does not change your model.

Start here
----------

.. toctree::
   :maxdepth: 1

   tutorials/notebooks/09-liesel-optim-basic
   tutorials/notebooks/10-liesel-optim-advanced

Common tasks
------------

.. toctree::
   :maxdepth: 1

   optimizer-monitoring
   optimizer-customization
   optimizer-data-flow
   optimizer-weighted-batching
   optimizer-checkpointing
   optimizer-migration

For arguments and defaults, see the :ref:`optimizer-api`.
``goose.optim_flat`` is deprecated as of version 0.8.0; use the migration guide
above to update existing code.
