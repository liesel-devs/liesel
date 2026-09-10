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
   )

``group_table`` is included unchanged in ``split.train``, ``split.validate``, and
``split.test``. It is not split, does not contribute to split likelihood scaling,
and is not included in batches derived automatically from the split. Use passthrough
for shared lookup tables or constants. Split per-observation covariates, weights,
and offsets alongside the response.

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
