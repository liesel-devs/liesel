Compare predictive accuracy
============================

Use pointwise log likelihoods to evaluate how a fitted model predicts held-out
observations with Pareto-smoothed importance sampling leave-one-out
cross-validation (PSIS-LOO). Start with :doc:`checked chains <goose-diagnostics>`;
LOO does not diagnose MCMC convergence.

Evaluate each observation
-------------------------

For ``model`` and ``results`` from :doc:`tutorials/md/01c-transform`:

.. code-block:: python

   import liesel.goose as gs
   import liesel.model as lsl

   samples = results.get_posterior_samples()
   log_lik = lsl.log_prob_pointwise({"y": model.vars["y"]}, samples)
   log_lik_y = log_lik["y_log_prob"]
   print(log_lik_y.shape)
   loo_result = gs.loo(log_lik_y)
   print(loo_result)

The shape is ``(4, 1000, 500)``: chains, draws, observations.
Select the response likelihood, not the total log posterior or its priors.
The distribution must retain per-observation log probabilities.
See :func:`~liesel.model.log_prob_pointwise` for the exact rules.

Read the result
----------------

``elpd_loo`` estimates expected log predictive density under leave-one-out
prediction; larger is better on the default log scale. ``se`` describes
uncertainty in that estimate. Inspect the reported Pareto-k diagnostics and
warnings: unreliable importance weights can make the approximation unsuitable.
The `ArviZ PSIS-LOO documentation
<https://python.arviz.org/en/stable/api/generated/arviz.loo.html>`_ explains
the result fields and diagnostics.

The optional ``samples`` argument to :func:`~liesel.goose.loo` is retained for
compatibility but ignored. By default, relative MCMC efficiency is estimated
from likelihood values. Use ``gs.loo(log_lik_y)`` for this workflow.

Compare like with like
----------------------

Fit each candidate model and evaluate the **same observations**, in the same
order and on the same likelihood scale. Compare their predictive scores with
uncertainty, not just a ranking. The standard error for an individual score
is not the standard error of a paired difference between two models.

The observation axis defines what is left out. Leaving out one response is
a different prediction task from leaving out a group or a future time interval.
Do not flatten dependent likelihood factors into purported independent
observations without checking that the resulting task matches your analysis.
