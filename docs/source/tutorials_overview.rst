Tutorials
=========

Start with the model-building basics, then follow the :doc:`sampling` or
:doc:`optimization` route for inference. Each route links to complete worked
examples and short guides for common tasks.

Build a model
-------------

.. toctree::
   :maxdepth: 1

   tutorials/md/01a-lin-reg
   tutorials/md/01b-model
   tutorials/md/02-ls-reg

Draw posterior samples
----------------------

The :doc:`sampling` guide starts with a complete regression posterior and
then combines NUTS and Gibbs updates. It also covers parameter blocks,
initialization, warmup, diagnostics, results, and custom kernels.

Explore applications
--------------------

.. toctree::
   :maxdepth: 1

   tutorials/md/03-gev
   tutorials/md/04-mcycle
   tutorials/md/06-pymc
   tutorials/md/07-error-correction
