.. _optimizer-likelihood-scaling:

Scale the loss
==============

The default training loss combines likelihood and priors. A minibatch likelihood
is scaled up to represent its full training group; priors are not scaled up.
Each group gets its own factor, so different batch sizes do not change the
relative weight of the groups.

Validation and test likelihoods include only the split branches and are scaled
to the corresponding training size. Unsplit observed likelihoods contribute to
training only, including observations explicitly marked as passthrough.
``LieselOptim`` then divides losses by the total training sample size by default;
use ``scale_loss=False`` to keep the sum. Sample size counts likelihood terms,
which need not equal the number of array elements.

:class:`.LieselVI` applies the same training-likelihood scaling to
:class:`.NegElboLoss`. Its default ``scale_loss=True`` divides the complete
negative ELBO by the training sample size, including entropy and prior terms.
For an explicitly constructed loss, set ``scale`` on that loss instead.

Validation leaves out priors by default. Use ``validation_strategy="log_prob"``
to include them. See :class:`~liesel.optim.NegLogProbLoss` for details.

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

Set sample sizes
----------------

For an existing ``model``, observed distributions with ``per_obs=False`` prevent
automatic sample-size inference. Construct the split explicitly:

.. code-block:: python

   import optax

   import liesel.optim as opt

   split = opt.PositionSplit.from_model(
       model, infer_sample_sizes=False, multi_size="manager", shuffle=False
   )
   optim = opt.LieselOptim(
       model,
       split=split,
       optimizers=optax.adam(0.01),
       loss_monitor="train_full_data",
   )

This chooses split-axis counts for scaling. Alternatively, supply effective
``sample_sizes`` to the split factory. Setting ``scale_loss=False`` on
``LieselOptim`` only disables final loss normalization; it does not disable
split inference or specify batch scaling.

Use a custom loss
-----------------

Custom aggregate likelihood, prior, or probability nodes require a custom
:class:`~liesel.optim.Loss` and an explicit split. A manual split specifies data
grouping and scaling; it does not change the objective used by
:class:`~liesel.optim.NegLogProbLoss`. The built-in loss accepts the standard sums
of observed distribution factors and parameter priors, including weak observed
variables. Other distribution factors must be classified appropriately or handled
by a custom loss.

For the data setup, see :doc:`optimizer-splitting` and
:doc:`optimizer-batching`. :doc:`optimizer-weighted-batching` explains the
additional correction for unequal sampling probabilities.

.. raw:: html

   <script src="_static/visualizations/resize-frames.js"></script>
