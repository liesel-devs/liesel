---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Compare predictions

Use pointwise log likelihoods to evaluate how a fitted model predicts held-out
observations with Pareto-smoothed importance sampling leave-one-out
cross-validation (PSIS-LOO). Start with {doc}`checked chains <goose-diagnostics>`;
LOO does not diagnose MCMC convergence.

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
```

## Prepare the example

These examples require the regression `model` and sampling `results` from
{doc}`tutorials/md/01c-transform`.

```{code-cell} ipython3
:load: _examples/goose-regression.py.inc
:tags: [remove-cell]
```

```{code-cell} ipython3
:tags: [remove-cell]

results = gs.LieselMCMC(model).run_for_epochs(
    seed=1,
    num_chains=4,
    adaptation=1000,
    posterior=1000,
    positions_included=["sigma_sq"],
    show_progress=False,
)
```

## Compute log likelihoods

For `model` and `results` from {doc}`tutorials/md/01c-transform`:

```{code-cell} ipython3
samples = results.get_posterior_samples()
log_lik = lsl.log_prob_pointwise({"y": model.vars["y"]}, samples)
log_lik_y = log_lik["y_log_prob"]
loo_result = gs.loo(log_lik_y)
```

```{code-cell} ipython3
log_lik_y.shape
```

```{code-cell} ipython3
loo_result
```

The shape is `(4, 1000, 500)`: chains, draws, observations.
Select the response likelihood, not the total log posterior or its priors.
The distribution must retain per-observation log probabilities.
See {func}`~liesel.model.log_prob_pointwise` for the exact rules.

## Read the result

`elpd_loo` estimates expected log predictive density under leave-one-out
prediction; larger is better on the default log scale. `se` describes
uncertainty in that estimate. Inspect the reported Pareto-k diagnostics and
warnings: unreliable importance weights can make the approximation unsuitable.
Here all 500 Pareto-k values are in the reported good range. `p_loo` is the
estimated effective number of parameters (about 3.07 here); it gives context
for model complexity and is not the predictive score to maximize.
The [ArviZ PSIS-LOO documentation](https://python.arviz.org/en/stable/api/generated/arviz.loo.html) explains
the result fields and diagnostics.

The optional `samples` argument to {func}`~liesel.goose.loo` is retained for
compatibility but ignored. By default, relative MCMC efficiency is estimated
from likelihood values. Use `gs.loo(log_lik_y)` for this workflow.

## Compare like with like

Fit each candidate model and evaluate the **same observations**, in the same
order and on the same likelihood scale. Compare their predictive scores with
uncertainty, not just a ranking. The standard error for an individual score
is not the standard error of a paired difference between two models.

The observation axis defines what is left out. Leaving out one response is
a different prediction task from leaving out a group or a future time interval.
Do not flatten dependent likelihood factors into purported independent
observations without checking that the resulting task matches your analysis.
