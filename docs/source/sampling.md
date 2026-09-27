---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# MCMC sampling

Use `liesel.goose` to draw posterior samples. Choose a sampling method
(a *kernel*) for each parameter block, then let Goose run the chains.

This complete example estimates a normal mean with a normal prior:

```{code-cell} ipython3
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl

mu = lsl.Var.new_param(
    0.0,
    dist=lsl.Dist(tfd.Normal, 0.0, 2.0),
    name="mu",
    inference=gs.MCMCSpec(gs.NUTSKernel, jitter_dist=tfd.Normal(0.0, 0.2)),
)
y = lsl.Var.new_obs(
    jnp.array([0.8, 1.3, 0.9, 1.6, 1.1]),
    dist=lsl.Dist(tfd.Normal, mu, 1.0),
    name="y",
)
model = lsl.Model(y)
```

```{code-cell} ipython3
results = gs.LieselMCMC(model).run_for_epochs(
    seed=2026,
    num_chains=4,
    adaptation=1000,
    posterior=1000,
    show_progress=False,
)

summary = gs.Summary(results)
samples = results.get_posterior_samples()
```

```{code-cell} ipython3
summary.to_dataframe()[["mean", "sd", "ess_bulk", "rhat"]].round(3)
```

The summary reports estimates and diagnostics; check both before using the
draws. `samples["mu"]` has shape `(4, 1000)`: chains first, then posterior
draws. Adaptation tunes the sampler and is excluded from these draws.

Every parameter you intend to sample needs a kernel. The high-level helper
reads {class}`~liesel.goose.MCMCSpec` objects from the model; it does not assign
a default kernel to parameters without one.

## Start here

```{toctree}
:maxdepth: 1

Overview <self>
First posterior <tutorials/md/01c-transform>
NUTS and Gibbs <tutorials/md/01d-gibbs-sampling>
```

## Common tasks

```{toctree}
:maxdepth: 1

goose-kernels
goose-initialization
goose-warmup
goose-diagnostics
goose-results
goose-model-comparison
goose-engine
Reproducibility <tutorials/md/05-reproducibility>
Custom kernels <tutorials/md/08-custom-kernel>
```

For exact arguments and defaults, see the {ref}`mcmc-api`.
