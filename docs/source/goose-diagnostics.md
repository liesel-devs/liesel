---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Check diagnostics

Check sampler errors and chain behavior before interpreting posterior estimates.

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
{doc}`tutorials/md/01c-transform`, with `sigma_sq` included in the stored draws.

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

## Read the summary

```{code-cell} ipython3
summary = gs.Summary(results, selected=["beta", "sigma_sq"])
```

```{code-cell} ipython3
summary.to_dataframe().set_index("var_fqn")[
    ["mean", "sd", "mcse_mean", "ess_bulk", "rhat"]
].round(3)
```

```{code-cell} ipython3
summary.aggregate_diagnostics().round(3)
```

```{code-cell} ipython3
summary.error_df().reset_index().filter(["error_msg", "phase", "count"])
```

`aggregate_diagnostics()` reports the smallest bulk/tail ESS and the largest
R-hat within each parameter variable. This keeps a coefficient vector's worst
diagnostics visible without printing every element.

* **R-hat** compares variation within and between chains.
* **Bulk and tail ESS** describe how much information the dependent draws
  contain about the center and tails.
* **MCSE** measures Monte Carlo uncertainty in an estimated summary, not the
  posterior uncertainty represented by its standard deviation or interval.

As a practical check, aim for R-hat below 1.01 across several chains and enough
ESS for the summaries you need. Compare MCSE with the precision your analysis
requires. See the [Stan diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html) for thresholds and
their limitations. Good diagnostics are evidence, not proof of convergence.

## Inspect the paths

The {doc}`first tutorial <tutorials/md/01c-transform>` shows trace plots
of all three estimated quantities. Goose also provides
{func}`~liesel.goose.plot_trace` and {func}`~liesel.goose.plot_cor`
for quick trace and autocorrelation plots.

Look for drift, chains occupying different regions, and long stretches with
little movement. Autocorrelation helps identify slow exploration.
{func}`~liesel.goose.plot_pairs` can reveal dependence between parameters;
{func}`~liesel.goose.plot_param` combines several views of one parameter.

## Investigate problems

| Symptom | Next check |
| --- | --- |
| Non-finite starting density or acceptance probability | Inspect `model.diagnose()`, parameter support, calculations, and starting values. |
| Divergent HMC/NUTS transitions | Investigate posterior geometry, scaling, and parameterization; a higher target acceptance may help but is not a general repair. |
| Maximum NUTS tree depth | Check mixing and parameter dependence. Longer trajectories cost more, so investigate before increasing the limit. |
| Poor chain overlap or high R-hat | Inspect the affected parameters, starting regions, and possible separate modes. |
| Small ESS or large MCSE | Inspect autocorrelation and blocking. Once exploration is satisfactory, more posterior draws can improve precision. |

Divergences can indicate biased exploration; a tree-depth limit is primarily an
efficiency concern. Kernel error codes differ, so use their accompanying
messages rather than comparing numeric codes across methods.

## Check acceptance

```{code-cell} ipython3
summary.acceptance_prob_df().reset_index()[
    ["phase", "acceptance_probability", "position_moved"]
].round(3)
```

The table separates acceptance probabilities from the fraction of transitions
that moved. NUTS does not report a move indicator, so its summary shows `NaN`.
For kernels that report movement, `position_moved` is the fraction of
transitions that changed the position.

High acceptance alone does not establish good mixing, and an exact Gibbs
update does not need an accept/reject step.

To investigate adaptation, {ref}`record kernel states <goose-record-tuning>`
during sampling.
