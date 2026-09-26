.. _optimizer-likelihood-scaling:

Understand loss scaling
=======================

The default training loss combines likelihood and priors. A minibatch likelihood
is scaled up to represent its full training group; priors are not scaled up.
Each group gets its own factor, so different batch sizes do not change the
relative weight of the groups.

Validation and test likelihoods include only the split branches and are scaled
to the corresponding training size. Unsplit observed likelihoods contribute to
training only, including observations explicitly marked as passthrough.
:class:`LieselOptim <liesel.optim.LieselOptim>` then divides losses by the total training sample size by default;
use ``scale_loss=False`` to keep the sum. Sample size counts likelihood terms,
which need not equal the number of array elements.

Validation leaves out priors by default. Use ``validation_strategy="log_prob"``
to include them. See :class:`~liesel.optim.NegLogProbLoss` for details.

With :class:`~liesel.optim.LaplaceLoss`, the objective is always the unscaled
full-training marginal posterior approximation. It includes priors and Jacobians;
the wrapper's ``scale_loss`` setting does not change it. See :doc:`optimizer-laplace`.

.. raw:: html

   <iframe
     class="interactive-visualization"
     data-visualization="likelihood"
     src="_static/visualizations/likelihood-scaling.html"
     title="Interactive overview of likelihood scaling through splitting and batching"
     loading="lazy"
     sandbox="allow-scripts allow-same-origin">
   </iframe>

`Open the likelihood-scaling overview in a separate page
<_static/visualizations/likelihood-scaling.html>`__.

Choose sample sizes for reduced likelihoods
-------------------------------------------

For an existing ``model``, observed distributions with ``per_obs=False`` prevent
automatic sample-size inference. Construct the split explicitly:

.. code-block:: python

   import optax
   import liesel.optim as opt

   split = opt.PositionSplit.from_model(
       model, infer_sample_sizes=False, multi_size="manager", shuffle=False
   )
   optim = opt.LieselOptim(
       model, split=split, optimizers=optax.adam(0.01),
       loss_monitor="train_full_data",
   )

This chooses split-axis counts for scaling. Alternatively, supply effective
``sample_sizes`` to the split factory. Setting ``scale_loss=False`` on
:class:`LieselOptim <liesel.optim.LieselOptim>` only disables final loss normalization; it does not disable
split inference or specify batch scaling.

Handle custom objectives
-------------------------

Custom aggregate likelihood, prior, or probability nodes need a loss that evaluates
those aggregates. A manual split specifies data grouping and scaling; it does not
change the objective used by
:class:`~liesel.optim.NegLogProbLoss`. That loss accepts the standard sums
of observed distribution factors and parameter priors, including weak observed
variables. Other distribution factors must be classified appropriately or handled
by a custom loss. :class:`LaplaceLoss <liesel.optim.LaplaceLoss>` uses the model's actual joint density,
including custom aggregates, with full training observations substituted. If
automatic split inference fails, supply an explicit full-training split.

For the data setup, see :doc:`optimizer-splitting` and
:doc:`optimizer-batching`. :doc:`optimizer-weighted-batching` explains the
additional correction for unequal sampling probabilities.

.. raw:: html

   <script src="_static/visualizations/resize-frames.js"></script>
