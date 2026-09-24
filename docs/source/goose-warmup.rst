Configure warmup and sampling
=============================

An iteration applies every configured kernel once. A chain repeats those
iterations; an epoch is a run of iterations with the same phase and storage
settings.

Set the phase lengths
----------------------

For an existing ``model`` with complete inference specifications:

.. code-block:: python

   import liesel.goose as gs

   results = gs.LieselMCMC(model).run_for_epochs(
       seed=2026,
       num_chains=4,
       adaptation=1000,
       burnin=200,
       posterior=2000,
   )

These counts apply to **each chain**. Adaptation tunes kernel settings.
Optional burnin lets the chains continue with tuning finished. Both belong to
warmup. ``get_posterior_samples()`` returns only the posterior phase: here,
2,000 draws from each of four chains. The additional recorded initial state
is not a posterior draw.

.. raw:: html

   <figure aria-label="Sampling phases from initialization to posterior draws">
     <div style="display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;margin:1rem 0">
       <div style="border:1px solid #999;padding:.7rem">Initial<br>state</div>
       <div style="border:2px solid #397e92;padding:.7rem;flex:2">
         <strong>Adaptation</strong><br>Fast → expanding slow epochs → fast
       </div>
       <div style="border:2px solid #397e92;padding:.7rem">Optional<br>burnin</div>
       <div style="border:2px solid #497a32;padding:.7rem;flex:2">
         <strong>Posterior sampling</strong><br>Draws used for inference
       </div>
     </div>
     <figcaption>Warmup includes adaptation and burnin. Box widths do not represent recommended durations.</figcaption>
   </figure>

During adaptation, kernels can tune within iterations and between epochs.
NUTS and HMC tune step size and, when enabled, their mass matrix in slow
epochs. RW and IWLS tune step size; Gibbs has nothing to tune.
See :doc:`goose-engine` to customize the schedule.

The example lengths are a starting point. Use
:doc:`diagnostics <goose-diagnostics>` to assess the result. More posterior
iterations do not repair invalid starting values or a problematic model.

Choose how many draws to store
-------------------------------

Keep ``posterior_thinning=1`` unless storing the draws is too expensive.
For example, requesting 2,000 posterior iterations with
``posterior_thinning=2`` retains 1,000 draws per chain; all 2,000 transitions
still run. Thinning discards information and does not fix poor mixing.

Avoid adaptation thinning for kernels that rely on sample history for tuning.
See :meth:`~liesel.goose.EngineBuilder.add_adaptation` for constraints.

Allow time for compilation
---------------------------

Goose uses JAX to compile the sampling calculations. Compilation can make the
first part of a run slower than later iterations. Multiple chains have
separate states and random keys; they do not imply one physical CPU core per
chain. Record the backend and numerical environment for
:doc:`reproducibility <tutorials/md/05-reproducibility>`.
