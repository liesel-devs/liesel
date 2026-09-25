---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Use posterior draws

Extract draws, compute predictions, and save a finished run.

## Prepare the example

These examples require the regression `model` and sampling `results` from
{doc}`tutorials/md/01c-transform`, with `sigma_sq` included in the stored draws.

```{code-cell} ipython3
:load: _examples/goose-regression.py
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

## Extract the posterior

```{code-cell} ipython3
import pandas as pd

import liesel.goose as gs

samples = results.get_posterior_samples()
sample_shapes = pd.DataFrame(
    {
        "Variable": ["beta", "sigma_sq"],
        "Shape": [samples["beta"].shape, samples["sigma_sq"].shape],
    },
)
```

```{code-cell} ipython3
sample_shapes
```

The shapes are `(4, 1000, 2)` and `(4, 1000)` in the tutorial:
chains, draws, then any parameter dimensions. Preserve the first two axes
when computing diagnostics. `get_samples()` also includes initial values
and warmup; use `get_posterior_samples()` for posterior inference.

## Record derived values

Goose records variables updated by kernels. The tutorial also sets
`positions_included=["sigma_sq"]` to record the derived variance.

Here NUTS samples `log_sigma_sq`; `sigma_sq` is its positive transformation.
`positions_excluded` overrides included positions and affects storage only.
It does not remove a kernel. Excluding sampled parameters can prevent later
predictions or diagnostics that need those draws.

For large intermediate arrays, retain the underlying parameter draws and
recompute quantities as needed instead:

```{code-cell} ipython3
variance = model.predict(samples, predict=["sigma_sq"])
```

```{code-cell} ipython3
gs.SamplesSummary(variance).to_dataframe()[["mean", "sd"]].round(3)
```

## Predict new responses

For the tutorial's design matrix variable `X`, construct a grid with the
same two columns: intercept and covariate.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

x_grid = jnp.linspace(0.0, 1.0, 50)
X_grid = jnp.column_stack([jnp.ones_like(x_grid), x_grid])

mean = model.predict(
    samples,
    predict=["mu"],
    newdata={"X": X_grid},
)
mean_summary = gs.SamplesSummary(mean, which=["quantiles"]).to_dataframe()
mean_summary["x"] = x_grid
```

```{code-cell} ipython3
mean_summary.iloc[[0, -1]][["x", "q_0.05", "q_0.5", "q_0.95"]].round(3)
```

`mean["mu"]` contains the regression mean at each grid point for every draw.
The displayed endpoints show how the fitted mean rises across the grid.
The 5th and 95th percentiles form a pointwise 90% credible interval for
the mean, as plotted in the first tutorial.

To include observation noise, simulate responses conditional on those draws:

```{code-cell} ipython3
parameter_draws = {name: samples[name] for name in model.parameters}
predictive = model.sample(
    shape=(),
    seed=jax.random.key(2027),
    posterior_samples=parameter_draws,
    newdata={"X": X_grid, "y": jnp.zeros_like(x_grid)},
)
y_predictive = predictive["y"]
```

```{code-cell} ipython3
y_predictive.shape
```

Pass draws for the writable parameters to `model.sample`. The recorded
`sigma_sq` is a derived value in this model; it is recomputed from
`log_sigma_sq` rather than set directly.

The placeholder `y` gives the response the new grid length while the model
updates its state. These values are replaced by simulated responses; they are
not additional observations used for fitting.

With `shape=()`, there is one simulated response vector per posterior draw.
Predictive intervals also include observation variation and are generally
wider. See {meth}`~liesel.model.Model.predict` and
{meth}`~liesel.model.Model.sample` for new-data and shape rules. Additional
sample dimensions, if requested, precede the posterior chain/draw dimensions.

## Save a finished run

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "regression-samples.pkl"
    results.pkl_save(path)
    restored = gs.SamplingResults.pkl_load(path)
```

This example uses a temporary directory; choose a persistent path to keep your
run. Load only trusted pickle files. Keep the model definition, data, and
{doc}`environment information <tutorials/md/05-reproducibility>` with the result.

Alternatively, pass `save_path="regression-samples.pkl"` to
`run_for_epochs`. If that path already exists, Goose **loads it and skips
sampling**, even when you changed the model, seed, or sampling settings.
Use a fresh path for a new fit. A saved `SamplingResults` object contains
results; it is not an engine checkpoint for resuming a chain.
