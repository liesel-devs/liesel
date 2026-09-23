.. _optimizer-data-flow:

Splitting, batching, and loss scaling
=====================================

A split decides which data belong to training, validation, and testing.
Batches take smaller pieces of the training data for each update. These are
separate choices: shuffling batches never changes the held-out data.

.. _optimizer-split-overview:

Keep matching rows together
---------------------------

Split responses and their covariates together. For an existing ``model``:

.. code-block:: python

   import liesel.optim as opt

   split = opt.PositionSplit.from_model(
       model,
       position_keys=["X", "y"],
       validate_axis_share=0.2,
       test_axis_share=0.1,
       seed=42,
   )

This puts 70% of rows in training, 20% in validation, and 10% in testing, subject
to rounding. Splits shuffle by default; set ``shuffle=False`` for an ordered
split.

The split's ``seed`` chooses which rows go into each part. The seed passed to
``LieselOptim`` controls batch shuffling or random batch sampling during fitting.
Custom losses and optimizers can also use its random key through ``carry.key``.
Starting parameter values come from the model. Set both seeds to repeat the
split and fitting randomness; seed any random data or starting values separately.

For separate groups, make the grouping explicit:

.. code-block:: python

   split = opt.PositionSplitManager.from_model(
       model,
       position_keys=[["X_a", "y_a"], ["X_b", "y_b"]],
       validate_axis_share=0.2,
       seed=42,
   )

Arrays within a group share row indices. Different groups split independently,
even if their lengths happen to match. Flat or omitted ``position_keys`` group
observed arrays by length; use nested groups when equal length does not mean
matching rows. Every group must have validation data if any group does.

Set ``split_axes`` when creating a split for observations on an axis other than
zero. A value of ``None`` keeps a shared table unchanged in every split and out
of automatic batches. Keep per-observation covariates, weights, and offsets with
the response.
Set the corresponding ``batch_axes`` when creating batches too.

``PositionSplit`` holds the split data. ``Split`` holds reusable row indices;
call ``split_position()`` to apply them. Manager classes handle several groups.
See :meth:`~liesel.optim.PositionSplit.from_model` for the factory options.

.. raw:: html

   <iframe
     class="interactive-visualization"
     data-visualization="split"
     src="_static/visualizations/split-api-overview.html"
     title="Interactive overview of the Liesel split API"
     loading="lazy"
     sandbox="allow-scripts allow-same-origin">
   </iframe>

`Open the split API overview in a separate page
<_static/visualizations/split-api-overview.html>`__.

.. _optimizer-batch-manager-overview:

Batch one or several groups
---------------------------

Create batches from the training split, then pass both to ``LieselOptim``:

.. code-block:: python

   batches = opt.Batches.from_split(split, batch_size=32)
   result = opt.LieselOptim(
       model,
       split=split,
       batches=batches,
       loss_monitor="validation",
       seed=43,
   ).fit()

Omit ``batches`` to use all training data in each update. Set batch size,
shuffling, axes, and epoch policy when creating the batches.

The engine checks each training array's length along its batch axis before
fitting. Splitting and batching may use different axes. When holding out rows
on the same axis, build batches from the split so they use the training size.

Only complete batches are used. With shuffling, the leftover rows can change
between epochs. For multiple groups, a :class:`~liesel.optim.BatchManager`
supplies one batch from each group at every update.

The ``Batches.from_split`` default, ``epoch_size="max"``, follows the group with
the most batches. Smaller groups start another shuffled pass as needed. Other
choices are ``"min"`` (stop with the shortest group), ``"strict"`` (require equal batch
counts), or a positive number of steps. Direct managers default to ``"strict"``.

``sample_with_replacement=True`` draws rows independently, so duplicates are
possible. ``Batches.from_split`` uses this for a group smaller than the requested
batch size. Such an epoch need not visit every row.

.. raw:: html

   <iframe
     class="interactive-visualization"
     data-visualization="batch"
     src="_static/visualizations/batch-manager-overview.html"
     title="Interactive BatchManager epoch overview"
     loading="lazy"
     sandbox="allow-scripts allow-same-origin">
   </iframe>

`Open the BatchManager overview in a separate page
<_static/visualizations/batch-manager-overview.html>`__.

.. _optimizer-likelihood-scaling:

What the loss means
-------------------

The default training loss combines likelihood and priors. A minibatch likelihood
is scaled up to represent its full training group; priors are not scaled up.
Each group gets its own factor, so different batch sizes do not change the
relative weight of the groups.

Validation and test likelihoods are scaled to the corresponding training size.
``LieselOptim`` then divides losses by the total training sample size by default;
use ``scale_loss=False`` to keep the sum. Sample size counts likelihood terms,
which need not equal the number of array elements.

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

.. raw:: html

   <script>
     window.addEventListener("message", (event) => {
       if (event.data?.type !== "liesel:visualization-height") return;
       const frame = document.querySelector(
         `[data-visualization="${event.data.visualization}"]`
       );
       if (
         !frame ||
         event.source !== frame.contentWindow?.frames[0] ||
         !Number.isFinite(event.data.height) ||
         event.data.height < 0 ||
         event.data.height > 10000
       ) return;
       frame.style.height = `${Math.ceil(event.data.height) + 32}px`;
     });
   </script>
