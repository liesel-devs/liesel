.. _optimizer-data-flow:

Splitting, batching, and likelihood scaling
============================================

These interactive diagrams show how the optimizer represents data splits, coordinates
mini-batches with different sample sizes, and scales likelihood contributions during
training and validation.

.. _optimizer-split-overview:

Choosing a split representation
-------------------------------

Use this overview to choose between :class:`~liesel.optim.Split`,
:class:`~liesel.optim.PositionSplit`, and their manager variants, and to compare their
constructors and typical use cases.

Explicit observation groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass nested ``position_keys`` to choose which arrays share split or minibatch
indices. Separate groups stay separate even when they contain equally many rows:

.. code-block:: python

   import liesel.optim as opt

   manager = opt.SplitManager.from_model(
       model,
       position_keys=[["x_a", "y_a"], ["x_b", "y_b"]],
       validate_axis_share=0.20,
       test_axis_share=0.10,
       shuffle=True,
       seed=42,
   )
   split = manager.split_position(model.extract_position(manager.position_keys))
   batches = opt.Batches.from_split(split, batch_size=64)

Arrays within each group must have matching lengths along their configured axes,
but their complete shapes may differ. Use ``split_axes`` and ``batch_axes`` for
non-leading observation axes. When using ``Batches.from_split`` directly, pass the
appropriate ``batch_axes`` as well. Group order is preserved and affects the random
keys assigned to children; repeating the same grouping and seed repeats the split.

The same nested syntax is accepted by ``BatchManager.from_model`` and
``PositionSplitManager.from_model``. ``Batches.from_model`` and
``PositionSplit.from_model`` require ``multi_size="manager"`` for multiple groups,
including equal-sized groups. With one group, these two factories still return a
single object; manager factories always return a manager.

Flat keys retain automatic grouping by axis length. Omitting ``position_keys``
selects all observed variables and groups them automatically. Mixed flat/nested
inputs, empty groups, duplicate keys, and incompatible lengths within an explicit
group are rejected. Factory options apply to all groups; construct child objects
manually when different groups need different settings or scalar size overrides.

Shuffling and reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The split default is now** ``shuffle=True`` **instead of** ``False``. This applies
to ``Split`` and all split factories, for both flat and nested selections. Set
``shuffle=False`` explicitly to retain ordered train/validation/test partitions,
for example in a chronological evaluation. Supply a seed for reproducible random
holdouts; without one, shuffled splits use the current Unix time in seconds.

Full-data splits, with no validation or test observations, preserve row order and
do not generate or use a split seed, regardless of ``shuffle``. This also applies
when requested shares round down to zero observations. ``LieselOptim`` explicitly
uses ``shuffle=False`` for its automatic full-data setup. Minibatch shuffling is
separate and remains controlled by the optimizer's seed; it does not change which
observations belong to training or validation.

Keeping shared entries unsplit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Map a selected key to ``None`` in ``split_axes`` when every split needs that entry
in full:

.. code-block:: python

   import liesel.optim as opt

   split = opt.PositionSplit.from_model(
       model,
       position_keys=["y", "group_id", "group_table"],
       validate_axis_share=0.2,
       split_axes={"group_table": None},
       shuffle=True,
       seed=42,
   )

``group_table`` is included unchanged in ``split.train``, ``split.validate``, and
``split.test``. It is not split, does not contribute to split likelihood scaling,
and is not included in batches derived automatically from the split. Use passthrough
for shared lookup tables or constants. Split per-observation covariates, weights,
and offsets alongside the response.

Passthrough entries can accompany any explicit observation group and remain global
to the resulting split. A group containing only passthrough entries is rejected;
each explicit group must contain at least one entry that is actually split.

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

Coordinating multiple batch streams
-----------------------------------

A :class:`~liesel.optim.BatchManager` combines child batches into joint optimizer
steps. Change the sample sizes, batch sizes, and epoch-size strategy to see which
observations contribute to each step.

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

Understanding likelihood scaling
--------------------------------

Explore how :class:`~liesel.optim.NegLogProbLoss` combines likelihood and prior
contributions for mini-batch training, validation, and full-data training. The diagram
also identifies the methods that supply each scaling factor and likelihood value.

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
