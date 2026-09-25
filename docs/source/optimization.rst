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

Approximate posterior uncertainty
---------------------------------

After a joint MAP fit, the default :class:`~liesel.optim.NegLogProbLoss` can
construct a Gaussian posterior approximation and draw parameter dictionaries:

.. code-block:: python

   import jax

   posterior = optim.loss.approximate_joint_posterior(result)
   draws = posterior.sample(1000, seed=jax.random.key(42))

The default ``at="min_monitor"`` selects ``position_min_monitor``;
``at="final"`` selects ``position_final``. Both use full-training curvature,
regardless of loss scaling or monitoring strategy. A validation or EMA minimum
may not be a posterior mode; the helper checks stationarity and raises if the
selected position is unsuitable. Only optimized parameters enter the
approximation, with omitted parameters held fixed at their model values.

Use ``model.predict`` to transform draws and evaluate derived quantities.
:class:`~liesel.optim.LaplaceApproximation` also provides the full covariance and
named marginal covariance, marginal precision factor, and conditional precision
blocks. See :doc:`optimizer-laplace` for examples of drawing and using joint
uncertainty. In hierarchical models, joint MAP can favor vanishing scale
parameters; that guide shows how to integrate selected effects before fitting.

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
