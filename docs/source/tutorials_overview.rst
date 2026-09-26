Tutorials
=========

Follow a complete workflow, then use its task guides when a specific question
arises. Each tutorial starts with its own data and model; do not carry fitted
objects between tutorials unless a page explicitly asks you to.

Build and predict
-----------------

Start with :doc:`tutorials/notebooks/11-model-building` to define variables,
inspect a graph, and change model values. Continue with
:doc:`tutorials/notebooks/12-model-predictions` to fit a model whose mean and
spread vary with covariates and distinguish mean uncertainty from new responses.
The :doc:`model-building` route links to focused guides for common model tasks.

Sample a posterior
------------------

Use :doc:`tutorials/md/01c-transform` for a first complete NUTS fit, then
:doc:`tutorials/md/01d-gibbs-sampling` to combine NUTS with an exact Gibbs update.
The :doc:`sampling` route covers kernel choices, initialization, diagnostics,
reproducibility, and extending Goose.

Optimize parameters
-------------------

Follow :doc:`tutorials/notebooks/09-liesel-optim-basic` for a first point estimate
and :doc:`tutorials/notebooks/10-liesel-optim-advanced` for two data groups.
The :doc:`optimization` route covers splitting, batching, monitoring, and more.

Explore applications
--------------------

The :doc:`examples` apply these tools to GEV responses, a sampler comparison,
variable selection with PyMC, and measurement-error correction.

.. toctree::
   :hidden:

   Overview <self>
   tutorials/md/01a-lin-reg
   tutorials/md/01b-model
   tutorials/md/02-ls-reg
