Tutorials
=========

Start with :doc:`model-building` to construct and inspect a model, then follow
:doc:`sampling` for posterior inference or :doc:`optimization` for point
estimates. Each route links to complete tutorials and short task guides.
The model guides also cover prediction and simulation.

Model walkthroughs
------------------

The model-building route introduces the current workflow. These earlier
walkthroughs provide additional regression examples:

.. toctree::
   :maxdepth: 1

   tutorials/md/01a-lin-reg
   tutorials/md/02-ls-reg

Draw posterior samples
----------------------

The :doc:`sampling` guide links to a first regression posterior and a tutorial
combining NUTS and Gibbs updates. Its task guides cover parameter blocks,
initialization, warmup, diagnostics, results, reproducibility, and custom kernels.

Explore applications
--------------------

.. toctree::
   :maxdepth: 1

   tutorials/md/03-gev
   tutorials/md/04-mcycle
   tutorials/md/06-pymc
   tutorials/md/07-error-correction

.. toctree::
   :hidden:

   tutorials/md/01b-model
