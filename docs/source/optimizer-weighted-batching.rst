Weighted minibatch sampling
===========================

Sampling weights make some training rows appear more often. The built-in loss
corrects for this, so the fit still targets the original likelihood on average.
These weights change how often rows are sampled, not their importance in the
statistical model.

Weight the training rows
------------------------

Starting with a ``model`` and its ``split``, build batches and pass them to
``LieselOptim``. Here, ``X`` and ``y`` have matching rows:

.. code-block:: python

   import optax

   import liesel.optim as opt

   weights = opt.Batches.weights_binned(split.train["y"], bins=10)
   batches = opt.Batches(
       split.split_position_keys,
       axis_size=split.train_axis_size,
       sample_size=split.train_sample_size,
       batch_size=32,
       sample_with_replacement=True,
       sampling_weights=weights,
   )
   result = opt.LieselOptim(
       model,
       optimizers=optax.adam(0.01),
       split=split,
       batches=batches,
       loss_monitor="train_full_data",
       seed=42,
   ).fit()

Weights must be finite, positive, and in the training split's current row order.
One vector applies to all aligned arrays in a group. Sampling uses replacement:
a row can appear twice in a batch, and an epoch need not visit every row.
Probabilities stay fixed for the run, including after checkpoint recovery.

Choose weights
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Helper
     - Effect
   * - :meth:`~liesel.optim.Batches.weights_balanced`
     - Sample rare categories more often. At ``strength=1``, categories have
       equal total sampling probability.
   * - :meth:`~liesel.optim.Batches.weights_for_shares`
     - Choose expected category shares, such as 70% common and 30% rare.
   * - :meth:`~liesel.optim.Batches.weights_binned`
     - Sample sparse numeric intervals more often. Choose ``bins`` and, if
       needed, ``strength`` between zero and one.

Pass labels or values from the training split. Stronger balancing can emphasize
outliers. These helpers preserve row order and return weights; they do not
change a batch object themselves. Their API pages describe accepted inputs.

Several groups
--------------

Give one weight vector per group, keyed by one selected variable in that group:

.. code-block:: python

   children = [
       opt.Batches(
           child.split_position_keys,
           axis_size=child.train_axis_size,
           sample_size=child.train_sample_size,
           batch_size=32,
           sample_with_replacement=True,
       )
       for child in split.splits
   ]
   batches = opt.BatchManager(
       children,
       epoch_size="max",
       sampling_weights={"y_a": weights_a, "y_b": weights_b},
   )

The split must already contain those groups. Omitted groups use uniform sampling.
If your model already holds the training rows, :meth:`~liesel.optim.Batches.from_model`
and :meth:`~liesel.optim.BatchManager.from_model` can infer sizes for you.
See :class:`~liesel.optim.BatchManager` for direct construction and overrides.

Custom likelihoods and losses
-----------------------------

The built-in correction gives a sampled row with probability ``p`` an extra
factor ``1 / (N * p)``, where ``N`` is the group's row count. This acts before
summing the likelihood, in addition to the usual minibatch scaling. Full-data
and validation evaluations need no sampling correction.

Keep per-observation likelihoods (``Dist.per_obs=True``). For custom reductions
or transposed likelihood arrays, set ``likelihood_axes`` to the axis that follows
the sampled rows. A likelihood already summed to a scalar cannot be corrected.
Custom losses must apply :meth:`~liesel.optim.Batches.scaled_log_lik` or
:meth:`~liesel.optim.Batches.correction_factors` themselves.
