---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
  execution_timeout: 180
  execution_raise_on_error: true
---

# Simulate responses

{meth}`~liesel.model.Model.sample` draws from the distributions attached to a
model, following their dependencies. With posterior parameter draws supplied,
it can generate posterior predictive responses. It does not run posterior MCMC.

## Draw from the priors

Use `model` from {doc}`tutorials/notebooks/12-model-predictions`, whose mean
and log-scale coefficients have proper priors:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
```

```{code-cell} ipython3
:load: _examples/model-predictions.py.inc
:tags: [remove-cell]
```

```{code-cell} ipython3
key = jax.random.key(27)
key, prior_key = jax.random.split(key)
prior_draws = model.sample(shape=(6,), seed=prior_key)
```

```{code-cell} ipython3
prior_draws["y"].shape
```

This returns six response datasets, with shape `(6, 120)`. Each uses newly
drawn coefficients and the existing covariates. These simulations reveal the
range of data allowed by your priors; very large or implausible values suggest
revisiting their scales. Use a fresh split key for another draw.

## Use posterior draws

For the posterior `samples` from the same tutorial:

```{code-cell} ipython3
x_grid = jnp.linspace(-1.0, 1.0, 60)

key, response_key = jax.random.split(key)
replicated = model.sample(
    shape=(),
    seed=response_key,
    posterior_samples=samples,
    newdata={"x": x_grid, "y": jnp.zeros_like(x_grid)},
)
```

```{code-cell} ipython3
replicated["y"].shape
```

`shape=()` requests one dataset per posterior draw. The result has shape
`(4, 500, 60)`. Posterior sample arrays must have two consistent leading axes
for chain and draw, followed by the parameter's own shape. The response
placeholder gives the new observation shape; random responses replace its values.

Supply all parameters you mean to condition on. A parameter with an attached
distribution but absent from `posterior_samples` may be sampled from that
distribution instead. Keys cannot occur in both `posterior_samples` and
`newdata`. See {doc}`model-prediction` for intervals on calculated quantities;
intervals computed from these replicated responses include response variation.

## Hold parameters fixed

To simulate responses conditional on the current coefficient values:

```{code-cell} ipython3
key, conditional_key = jax.random.split(key)
conditional = model.sample(
    shape=(6,),
    seed=conditional_key,
    fixed=["beta", "gamma"],
)
```

```{code-cell} ipython3
conditional["y"].shape
```

This keeps `beta` and `gamma` fixed and returns six datasets. Fixed entries
must not also appear in `posterior_samples`. Strong variables without distributions,
including fixed covariates, retain their supplied values. An unbounded parameter
without a distribution has an improper constant prior: retaining its current
value is conditional simulation, not drawing from that prior.

Calculated inputs are recomputed as needed. The returned dictionary contains
sampled quantities; use `predict` for derived quantities you want to inspect.
The `chunk_size` option limits parallel intermediate calculations, but all
returned samples still need to fit in memory.
