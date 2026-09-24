Inspect chains and diagnose problems
====================================

Check sampler errors and chain behavior before interpreting posterior estimates.
The snippets below use ``model`` and ``results`` from
:doc:`tutorials/md/01c-transform`.

Read a compact summary
-----------------------

.. code-block:: python

   import liesel.goose as gs

   summary = gs.Summary(results, selected=["beta", "sigma_sq"])
   print(summary.to_dataframe())
   print(summary.aggregate_diagnostics())
   print(summary.error_df(per_chain=True))

``aggregate_diagnostics()`` reports the smallest bulk/tail ESS and the largest
R-hat within each parameter variable. This keeps a coefficient vector's worst
diagnostics visible without printing every element.

* **R-hat** compares variation within and between chains.
* **Bulk and tail ESS** describe how much information the dependent draws
  contain about the center and tails.
* **MCSE** measures Monte Carlo uncertainty in an estimated summary, not the
  posterior uncertainty represented by its standard deviation or interval.

As a practical check, aim for R-hat below 1.01 across several chains and enough
ESS for the summaries you need. Compare MCSE with the precision your analysis
requires. See the `Stan diagnostic guidance
<https://mc-stan.org/learn-stan/diagnostics-warnings.html>`_ for thresholds and
their limitations. Good diagnostics are evidence, not proof of convergence.

Inspect the paths
-----------------

.. code-block:: python

   gs.plot_trace(results, params=["beta", "sigma_sq"])
   gs.plot_cor(results, params=["beta"])

Look for drift, chains occupying different regions, and long stretches with
little movement. Autocorrelation helps identify slow exploration.
:func:`~liesel.goose.plot_pairs` can reveal dependence between parameters;
:func:`~liesel.goose.plot_param` combines several views of one parameter.

Connect a symptom to a next check
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Next check
   * - Non-finite starting density or acceptance probability
     - Inspect ``model.diagnose()``, parameter support, calculations, and starting values.
   * - Divergent HMC/NUTS transitions
     - Investigate posterior geometry, scaling, and parameterization; a higher target acceptance may help but is not a general repair.
   * - Maximum NUTS tree depth
     - Check mixing and parameter dependence. Longer trajectories cost more, so investigate before increasing the limit.
   * - Poor chain overlap or high R-hat
     - Inspect the affected parameters, starting regions, and possible separate modes.
   * - Small ESS or large MCSE
     - Inspect autocorrelation and blocking. Once exploration is satisfactory, more posterior draws can improve precision.

Divergences can indicate biased exploration; a tree-depth limit is primarily an
efficiency concern. Kernel error codes differ, so use their accompanying
messages rather than comparing numeric codes across methods.

Inspect acceptance separately
-----------------------------

.. code-block:: python

   print(summary.acceptance_prob_df())
   moved = results.get_posterior_position_moved()

The first reports acceptance probabilities; ``moved`` records whether each
transition changed the position. These are different quantities. High
acceptance alone does not establish good mixing, and an exact Gibbs update
does not need an accept/reject step.

Record tuning when investigating warmup
---------------------------------------

Kernel states must be recorded during sampling. For the tutorial's single
joint NUTS kernel, run:

.. code-block:: python

   import matplotlib.pyplot as plt

   recorded = gs.LieselMCMC(model).run_for_epochs(
       seed=2026, num_chains=4, adaptation=1000, posterior=1000,
       store_kernel_states=True,
   )
   states = recorded.get_warmup_kernel_states()
   kernel_id = recorded.get_kernels_by_pos_key()["beta"]
   step_size = states[kernel_id]["step_size"]
   plt.plot(step_size.T)
   plt.xlabel("Warmup iteration")
   plt.ylabel("Step size")
   plt.show()

The columns plotted are chains. Inspect whether step sizes settle toward the
end of adaptation; changes at epoch boundaries can reflect tuning.
Recording kernel states uses additional memory, especially for mass matrices.
It cannot be enabled retroactively for an existing result.
