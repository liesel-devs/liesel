---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
---

# Optimization

Use {py:class}`liesel.optim.LieselOptim` to fit model parameters or find starting
values for MCMC. It minimizes the negative log posterior: likelihood plus priors.
Without priors, this gives a maximum likelihood fit.

To fit an approximate posterior distribution, use {doc}`variational-inference`.

For a small Normal model, a full-data fit takes:

```{code-cell} ipython3
import logging

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

logging.getLogger("liesel").setLevel(logging.WARNING)

loc = lsl.Var.new_param(0.0, name="loc")
y = lsl.Var.new_obs(
    jnp.array([1.0, 2.0, 3.0]), lsl.Dist(tfd.Normal, loc, 1.0), name="y"
)
model = lsl.Model(y)

result = opt.LieselOptim(
    model,
    optimizers="lbfgs",
    loss_monitor="train_full_data",
    show_progress=False,
).fit()
```

```{code-cell} ipython3
result.position_min_monitor
```

The fitted mean is 2, the average of these three observations.

L-BFGS needs the full data and a deterministic loss. For minibatches, use Adam.
The `optimizers` argument is required. Pass a configured Optax transformation,
such as `optimizers=optax.adam(0.01)`, to optimize all parameters with it.
The fitted values are returned separately; fitting does not change your model.

## Start here

```{toctree}
:maxdepth: 1

tutorials/notebooks/09-liesel-optim-basic
Fit two data groups <tutorials/notebooks/10-liesel-optim-advanced>
```

## Common tasks

```{toctree}
:maxdepth: 1

optimizer-monitoring
optimizer-customization
optimizer-splitting
optimizer-batching
optimizer-loss-scaling
optimizer-weighted-batching
optimizer-checkpointing
optimizer-migration
```

For arguments and defaults, see the {ref}`optimizer-api`.
`goose.optim_flat` is deprecated and will be removed in 0.8.0; use the migration guide
above to update existing code.
