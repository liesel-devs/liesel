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

   import optax

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
Both seeds default to ``0``, making split and fitting randomness reproducible in
the same environment. Choose other integer seeds for different runs. Explicit
``seed=None`` opts into Unix time in whole seconds, so calls in the same second
can share a seed. Starting parameter values come from the model; seed any random
data or starting values separately.

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

Automatic grouping requires an observed likelihood in each group. If a group
has none, specify its row keys as an explicit nested group, or mark shared data
as passthrough. Matching lengths alone do not establish row alignment: lookup
tables must be passthrough even when their length matches a response.

Set ``split_axes`` when creating a split for observations on an axis other than
zero. A value of ``None`` keeps a shared table unchanged in every split and out
of automatic batches. Keep per-observation covariates, weights, and offsets with
the response.
Set the corresponding ``batch_axes`` when creating batches too.

For a model with a lookup table ``z`` indexed by row-level group IDs, construct
``PositionSplit.from_model(model, split_axes={"z": None})`` and pass the result
as ``LieselOptim(..., split=split)``. This also works for scalar constants.
Automatic setup raises an informative error for scalars or inferred groups
without a likelihood instead of guessing how to split them.

``PositionSplit`` holds the split data. ``Split`` holds reusable row indices;
call ``split_position()`` to apply them. Manager classes handle several groups.
See :meth:`~liesel.optim.PositionSplit.from_model` for the factory options.

For observed distributions with ``per_obs=False`` or a custom ``log_lik_node``,
automatic sample-size inference is unavailable. Construct the split explicitly:

.. code-block:: python

   split = opt.PositionSplit.from_model(
       model, infer_sample_sizes=False, multi_size="manager", shuffle=False
   )
   optim = opt.LieselOptim(
       model, split=split, optimizers=optax.adam(0.01),
       loss_monitor="train_full_data",
   )

This chooses split-axis counts for scaling. Alternatively, supply effective
``sample_sizes`` to the split factory. Setting ``scale_loss=False`` on
``LieselOptim`` only disables final loss normalization; it does not disable
split inference or specify batch scaling.

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
       optimizers=optax.adam(0.01),
       split=split,
       batches=batches,
       loss_monitor="validation",
       seed=43,
   ).fit()

For default batch settings, pass ``batch_size=32`` directly to ``LieselOptim``.
This calls ``Batches.from_split(split, batch_size=32)``. Omit both ``batch_size``
and ``batches`` to use all training data in each update. For custom shuffling,
axes, or epoch policy, create ``batches`` explicitly instead.

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

Pass ``sample_with_replacement=True`` to draw rows independently, allowing
duplicates. Managers also enable this for groups smaller than the requested batch
size. Such an epoch need not visit every row. For weighted sampling, pass
``sampling_weights`` in training-row order.

:meth:`~liesel.optim.Batches.from_split` also accepts custom sample sizes and
likelihood axes. :meth:`~liesel.optim.BatchManager.from_split` accepts the same
options and always returns a manager. For different settings per group, create
each child with ``Batches.from_split`` and combine them with ``BatchManager``.

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

Validation and test likelihoods include only the split branches and are scaled
to the corresponding training size. Unsplit observed likelihoods contribute to
training only, including observations explicitly marked as passthrough.
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
