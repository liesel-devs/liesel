Split data
==========

A split decides which rows belong to training, validation, and testing.
Create it before :doc:`configuring minibatches <optimizer-batching>`;
shuffling batches never changes the held-out data. Selecting rows must
preserve their :ref:`likelihood contributions <optimizer-row-wise>`.
The optimizer does not check this property.

.. _optimizer-split-overview:

Keep rows together
------------------

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
``LieselOptim`` controls batch sampling during fitting. Both default to ``0``.
Starting parameter values come from the model; seed any random data or starting
values separately.

Split several groups
--------------------

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

Automatic grouping requires an observed likelihood in each group. For row data
without a likelihood, supply an explicit nested group.

Keep shared data
----------------

Shared tables and scalar constants must stay unchanged in every split. For a
lookup table ``z`` indexed by row-level group IDs, use
``PositionSplit.from_model(model, split_axes={"z": None})`` and pass the result
as ``LieselOptim(..., split=split)``. This also keeps ``z`` out of automatic batches,
even if its length matches a response. Keep per-row covariates, weights, and
offsets with their response.

For observations on an axis other than zero, set ``split_axes`` and the
corresponding ``batch_axes`` when :doc:`creating batches <optimizer-batching>`.

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
