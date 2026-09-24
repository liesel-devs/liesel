Choose kernels and parameter blocks
===================================

A kernel updates one block of parameters conditional on the current values of
the other blocks. Goose applies the kernels in sequence within each iteration.
You can use different methods for different blocks without changing the model's
priors or likelihood.

Choose an update
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Kernel
     - Use it for
     - What you supply or check
   * - :class:`~liesel.goose.NUTSKernel`
     - Continuous parameters with gradients.
     - Unconstrained sampling coordinates; trajectory length is selected automatically.
   * - :class:`~liesel.goose.HMCKernel`
     - Hamiltonian updates with a chosen trajectory length.
     - Gradients and the number of integration steps.
   * - :class:`~liesel.goose.IWLSKernel`
     - Blocks where local curvature gives a useful Gaussian proposal.
     - A differentiable density with suitable curvature; larger blocks make matrix operations more expensive.
   * - :class:`~liesel.goose.RWKernel`
     - A simple Gaussian random-walk update.
     - A useful parameter scale; mixing can be slow for strongly dependent parameters.
   * - :class:`~liesel.goose.GibbsKernel`
     - A block whose full conditional you can sample directly.
     - A function drawing from that full conditional.
   * - :class:`~liesel.goose.MHKernel`
     - A custom Metropolis-Hastings proposal.
     - A proposal function and the backward-minus-forward log proposal correction.

See :doc:`tutorials/md/01c-transform` for a positive parameter sampled on a
log scale, and :doc:`tutorials/md/01d-gibbs-sampling` for an exact Gibbs update.
For custom proposals, start with :doc:`tutorials/md/08-custom-kernel`.

Inspect the configured parameters
----------------------------------

For an existing Liesel ``model``:

.. code-block:: python

   import liesel.goose as gs

   for name, param in model.parameters.items():
       print(name, param.inference)

   mcmc = gs.LieselMCMC(model)
   for kernel in mcmc.get_kernel_list():
       print(type(kernel).__name__, kernel.position_keys)

Check that every parameter you intend to estimate appears in a kernel.
A parameter without an inference specification stays at its current value
unless you add a kernel manually through an :doc:`engine builder <goose-engine>`.

Choose separate or joint updates
---------------------------------

For the ``model`` from :doc:`tutorials/md/01c-transform`, give the coefficients
and log variance separate NUTS kernels:

.. code-block:: python

   beta = model.vars["beta"]
   log_variance = model.vars["log_sigma_sq"]
   beta.inference = gs.MCMCSpec(gs.NUTSKernel, order=1)
   log_variance.inference = gs.MCMCSpec(gs.NUTSKernel, order=2)

One iteration first updates ``beta`` given the old log variance, then updates
the log variance given the new ``beta``. The vector ``beta`` is already one
block: both coefficients are proposed together.

To update both variables jointly, give them an explicit shared group:

.. code-block:: python

   import tensorflow_probability.substrates.jax.distributions as tfd

   joint = gs.MCMCSpec(
       gs.NUTSKernel,
       kernel_group="regression",
       kernel_kwargs={"da_target_accept": 0.9},
       jitter_dist=tfd.Normal(0.0, 0.2),
   )
   beta.inference = joint
   log_variance.inference = joint

The shared specification keeps the group's kernel, arguments, and order
consistent. Reusing a specification without ``kernel_group`` does **not**
combine variables. Assigning a new specification replaces the previous one,
including its jitter settings.

Joint updates can help with posterior dependence but cost more per update.
Inspect :doc:`diagnostics <goose-diagnostics>` after changing the grouping.
A higher ``da_target_accept`` generally adapts toward smaller steps; it is not
a substitute for a well-scaled model.

Set the update order
---------------------

Smaller ``order`` values run first. With equal values, the high-level helper
uses reverse graph order, starting nearer the responses. Inspect
``get_kernel_list()`` if the order matters, and use the same order for every
member of a joint group. See :class:`~liesel.goose.MCMCSpec` and
:meth:`~liesel.goose.LieselMCMC.get_kernel_groups` for the exact grouping rules.

.. raw:: html

   <figure aria-label="Two sequential parameter updates in one iteration">
     <div style="display:flex;flex-wrap:wrap;gap:.6rem;align-items:center;padding:1rem;border:1px solid #999;border-radius:.4rem">
       <span>Current β and σ²</span><span aria-hidden="true">→</span>
       <strong>Update β given σ²</strong><span aria-hidden="true">→</span>
       <strong>Update σ² given new β</strong><span aria-hidden="true">→</span>
       <span>Store the iteration</span>
     </div>
     <figcaption>Each block sees updates already made in this iteration. Separate chains run their own sequences.</figcaption>
   </figure>
