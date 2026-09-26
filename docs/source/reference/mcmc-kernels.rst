Kernels
=======

Goose makes it easy for you to combine different MCMC kernels for different blocks of
model parameters. You can also define your own kernel by implementing
the :class:`Kernel <liesel.goose.Kernel>` protocol.

To draw samples from your posterior, you will want to call
:meth:`sample_all_epochs <liesel.goose.Engine.sample_all_epochs>`. Once sampling is done, you can obtain the results
with :meth:`get_results <liesel.goose.Engine.get_results>`, which will return a :class:`SamplingResults <liesel.goose.SamplingResults>`
instance.

.. toctree::
   :maxdepth: 1

   Overview <self>

.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :recursive:

   ~liesel.goose.IWLSKernel
   ~liesel.goose.NUTSKernel
   ~liesel.goose.HMCKernel
   ~liesel.goose.RWKernel
   ~liesel.goose.MHKernel
   ~liesel.goose.MHProposal
   ~liesel.goose.GibbsKernel
