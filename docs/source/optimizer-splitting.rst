Split training, validation, and test data
=========================================

A split decides which rows belong to training, validation, and testing.
Create it before :doc:`configuring minibatches <optimizer-batching>`;
shuffling batches never changes the held-out data. Selecting rows must
preserve their :ref:`likelihood contributions <optimizer-row-wise>`.
The optimizer does not check this property.

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
to rounding. Splits with validation or test data shuffle by default; set
``shuffle=False`` for an ordered split. Without holdouts, splits preserve the
original row order and ignore the seed, even with ``shuffle=True``. This is also
the behavior of ``LieselOptim``'s automatic full-training split.

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
Set the corresponding ``batch_axes`` when :doc:`creating batches <optimizer-batching>` too.

For a model with a lookup table ``z`` indexed by row-level group IDs, construct
``PositionSplit.from_model(model, split_axes={"z": None})`` and pass the result
as ``LieselOptim(..., split=split)``. This also works for scalar constants.
Automatic setup raises an informative error for scalars or inferred groups
without a likelihood instead of guessing how to split them.

``PositionSplit`` holds the split data. ``Split`` holds reusable row indices;
call ``split_position()`` to apply them. Manager classes handle several groups.
See :meth:`~liesel.optim.PositionSplit.from_model` for the factory options.

For reduced likelihoods or custom objectives, see :doc:`optimizer-loss-scaling`.
For computed observations such as copulas, see :ref:`optimizer-weak-observations`.

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

Next, :doc:`create minibatches <optimizer-batching>` or read how
:doc:`loss scaling <optimizer-loss-scaling>` handles held-out data.

.. raw:: html

   <script src="_static/visualizations/resize-frames.js"></script>
