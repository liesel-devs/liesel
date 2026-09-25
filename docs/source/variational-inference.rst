Variational inference
=====================

Use :class:`liesel.optim.LieselVI` to fit an approximate posterior for a Liesel
model. It minimizes the negative evidence lower bound (ELBO) by adjusting the
parameters of a variational distribution. The fitted values describe that
distribution; draw samples from it to summarize the target model's parameters.

Fit and sample a Gaussian approximation
---------------------------------------

For an existing Liesel ``model`` with parameters on the real line:

.. code-block:: python

   import jax
   import optax
   import liesel.optim as opt

   loss = opt.NegElboLoss.mvn_diag(model, nsamples=10, scale=True)
   result = opt.LieselVI(
       model,
       loss=loss,
       optimizers=optax.adam(0.001),
       loss_monitor=opt.EmaTrainLossMonitor(effective_window=2.0),
       seed=42,
   ).fit()
   samples = loss.vdist.sample(
       jax.random.key(43),
       sample_shape=(1_000,),
       at_position=result.position_final,
   )

``mvn_diag`` creates a diagonal Gaussian over the model parameters and stores
its :class:`.VDist` in ``loss.vdist``. Samples use the target parameter names.
Transform constrained parameters to the real line before constructing this
approximation, and transform draws back when summarizing them. Fitting leaves
the supplied model unchanged.

Use ``mvn_tril`` for a dense covariance or :class:`.CompositeVDist` to combine
families for different parameter blocks. An explicit loss keeps its own sample
count, scaling, entropy and prior settings; configure them on the loss.

Choose a monitoring loss
------------------------

Both ``optimizers`` and ``loss_monitor`` are required. An
:class:`.EmaTrainLossMonitor` smooths the losses evaluated before optimizer
updates and carries that average across epochs. ``"train_full_data"`` adds an
evaluation on all training rows after each epoch. That evaluation still draws
variational samples, so it can fluctuate even without minibatches. Analytic
entropy reduces one source of Monte Carlo noise; it does not make the whole
ELBO deterministic. See :doc:`optimizer-monitoring` for stopping and histories.

ELBO losses do not support validation splits. Use a train/test split for a final
predictive check, as in the basic tutorial. ``position_final`` is the final
iterate; ``position_min_monitor`` is the saved epoch-end position with the
smallest monitoring value. With an EMA, that value combines losses from several
positions. Inspect convergence and posterior predictions before choosing a fit.

Work through the examples
-------------------------

.. toctree::
   :maxdepth: 1

   tutorials/notebooks/11-liesel-vi-basic
   tutorials/notebooks/12-liesel-vi-advanced

Configure data and optimizers
-----------------------------

* :doc:`optimizer-splitting` explains explicit splits, axes and seeds.
* :doc:`optimizer-batching` covers aligned row groups and fixed computed data.
* :doc:`optimizer-loss-scaling` explains sample counts and normalization.
* :doc:`optimizer-weighted-batching` covers unequal sampling probabilities.
* :doc:`optimizer-customization` covers learning-rate schedules and parameter
  blocks. For VI, explicit optimizers select names in ``loss.q.parameters``.
* :doc:`optimizer-checkpointing` explains how to resume a fit.

For constructor arguments, ELBO estimation and sampling, see
:class:`.LieselVI`, :class:`.NegElboLoss`, :class:`.VDist` and
:class:`.CompositeVDist` in the :ref:`optimizer-api`.
