---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
---

# Sample with weights

Sampling weights make some training rows appear more often. The built-in loss
corrects for this, so the fit still targets the original likelihood on average.
These weights change how often rows are sampled, not their importance in the
statistical model.

This page uses a small Gaussian regression. Expand the setup to run the
examples from top to bottom.

```{code-cell} ipython3
import logging

import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
```

```{code-cell} ipython3
:tags: [hide-input]

logging.getLogger("liesel").setLevel(logging.WARNING)

rng = np.random.default_rng(42)
x = np.linspace(-1.0, 1.0, 128)
X = lsl.Var.new_obs(jnp.asarray(x), name="X")
beta = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="beta")
log_sigma = lsl.Var.new_param(0.0, name="log_sigma")
sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")
mu = lsl.Var.new_calc(lambda X, beta: X * beta, X, beta, name="mu")
y = lsl.Var.new_obs(
    jnp.asarray(0.5 * x + rng.normal(scale=0.7, size=x.size)),
    lsl.Dist(tfd.Normal, mu, sigma),
    name="y",
)
model = lsl.Model(y)
```

## Weight the training rows

Split `X` and `y` together, then build batches from the training rows:

```{code-cell} ipython3
split = opt.PositionSplit.from_model(model, position_keys=["X", "y"])
weights = opt.Batches.weights_binned(split.train["y"], bins=10)
batches = opt.Batches.from_split(
    split,
    batch_size=32,
    sample_with_replacement=True,
    sampling_weights=weights,
)
result = opt.LieselOptim(
    model,
    optimizers=optax.adam(0.01),
    split=split,
    batches=batches,
    loss_monitor="train_full_data",
    show_progress=False,
    seed=42,
).fit()
```

```{code-cell} ipython3
result.position_min_monitor
```

Weights must be finite, positive, and in the training split's current row order.
One vector applies to all aligned arrays in a group. Sampling uses replacement:
a row can appear twice in a batch, and an epoch need not visit every row.
Probabilities stay fixed for the run, including after checkpoint recovery.

## Choose weights

| Helper | Effect |
| --- | --- |
| {py:meth}`~liesel.optim.Batches.weights_balanced` | Sample rare categories more often. At `strength=1`, categories have equal total sampling probability. |
| {py:meth}`~liesel.optim.Batches.weights_for_shares` | Choose expected category shares, such as 70% common and 30% rare. |
| {py:meth}`~liesel.optim.Batches.weights_binned` | Sample sparse numeric intervals more often. Choose `bins` and, if needed, `strength` between zero and one. |

Pass labels or values from the training split. Stronger balancing can emphasize
outliers. Their API pages describe accepted inputs.

## Several groups

For several groups, pass a mapping such as
`sampling_weights={"y_a": weights_a, "y_b": weights_b}` to
{meth}`Batches.from_split <liesel.optim.Batches.from_split>`. Each vector must follow its group's training-row order.

Use one selected variable name as the key for each group. Omitted groups use
uniform sampling. See {py:meth}`~liesel.optim.Batches.from_split` for other settings.

## Use a custom loss

The built-in correction gives a sampled row with probability `p` an extra
factor `1 / (N * p)`, where `N` is the group's row count. This acts before
summing the likelihood, in addition to the usual minibatch scaling. Full-data
and validation evaluations need no sampling correction.

Keep per-observation likelihoods (`Dist.per_obs=True`). For custom reductions
or transposed likelihood arrays, set `likelihood_axes` to the axis that follows
the sampled rows. A likelihood already summed to a scalar cannot be corrected.
Custom losses must apply {py:meth}`~liesel.optim.Batches.scaled_log_lik` or
{py:meth}`~liesel.optim.Batches.correction_factors` themselves.
