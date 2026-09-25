---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
---

# Fit a model with two data groups

Some models use groups with different numbers of rows. Here, a zero-inflated
gamma model has one response for whether a value is zero and another for its
positive amount. We keep matching rows together within each group.

Start with [the first tutorial](09-liesel-optim-basic.md) if you are new to
`LieselOptim`.

```{code-cell} ipython3
import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

logging.getLogger("liesel").setLevel(logging.WARNING)
```

## Simulate the two groups

The zero indicator has 1,000 rows. The gamma response contains only positive
observations, so it has fewer rows and its own design matrix.

```{code-cell} ipython3
rng = np.random.default_rng(202406)
x = rng.normal(size=1000)
X_zero_values = np.column_stack([np.ones_like(x), x])
alpha_true = np.array([-0.55, 1.10])
zero_probability = 1 / (1 + np.exp(-(X_zero_values @ alpha_true)))
zero_values = rng.binomial(1, zero_probability)

X_positive_values = X_zero_values[zero_values == 0]
beta_true = np.array([1.0, 0.65])
positive_mean = np.exp(X_positive_values @ beta_true)
positive_values = rng.gamma(shape=5.0, scale=positive_mean / 5.0)
```

```{code-cell} ipython3
pd.DataFrame(
    {"rows": [len(zero_values), len(positive_values)]}, index=["all", "positive"]
)
```

## Build the model

Each group has its own coefficients. We fix the gamma concentration at 5 to
keep the example focused on handling the data.

```{code-cell} ipython3
alpha = lsl.Var.new_param(
    jnp.zeros(2), lsl.Dist(tfd.Normal, loc=0.0, scale=5.0), name="alpha"
)
X_zero = lsl.Var.new_obs(jnp.asarray(X_zero_values), name="X_zero")
logits = lsl.Var.new_calc(lambda X, a: X @ a, X_zero, alpha, name="logits")
zero = lsl.Var.new_obs(
    jnp.asarray(zero_values),
    lsl.Dist(tfd.Binomial, total_count=1.0, logits=logits),
    name="zero",
)

beta = lsl.Var.new_param(
    jnp.zeros(2), lsl.Dist(tfd.Normal, loc=0.0, scale=5.0), name="beta"
)
X_positive = lsl.Var.new_obs(jnp.asarray(X_positive_values), name="X_positive")
rate = lsl.Var.new_calc(
    lambda X, b: 5.0 / jnp.exp(X @ b), X_positive, beta, name="rate"
)
positive = lsl.Var.new_obs(
    jnp.asarray(positive_values),
    lsl.Dist(tfd.Gamma, concentration=5.0, rate=rate),
    name="positive",
)
model = lsl.Model(zero, positive)
```

Plot the model to see how its variables depend on one another:

```{code-cell} ipython3
---
mystnb:
  image:
    alt: "Two model branches: X_zero and alpha determine the zero probabilities; X_positive and beta determine the gamma rate for positive responses."
---
model.plot()
```

## Split each group

Each inner list names arrays that must share row indices. Groups split
independently. This example fits separate parameters for the two responses;
for a joint held-out prediction of the same people or events, split by that
shared identity before forming the groups.

```{code-cell} ipython3
split = opt.PositionSplitManager.from_model(
    model,
    position_keys=[["X_zero", "zero"], ["X_positive", "positive"]],
    validate_axis_share=0.2,
    seed=123,
)
```

```{code-cell} ipython3
pd.DataFrame(
    [
        {"keys": child.position_keys, "training_rows": child.train_axis_size}
        for child in split.splits
    ]
)
```

## Fit both groups together

`Batches.from_split()` builds a `BatchManager` for the two groups. Pass it to
`LieselOptim`. Each update uses 32 rows from each group. The default epoch length follows the larger group; the smaller
group starts another shuffled pass when needed. The loss scales each group's
batch to its training size.

```{code-cell} ipython3
batches = opt.Batches.from_split(split, batch_size=32)
result = opt.LieselOptim(
    model,
    split=split,
    batches=batches,
    optimizers=optax.adam(0.03),
    loss_monitor="validation",
    stopper=opt.Stopper(epochs=250, patience=30, rtol=1e-4),
    show_progress=False,
    seed=321,
).fit()
position = result.position_min_monitor
```

```{code-cell} ipython3
pd.DataFrame(
    {"zero": position["alpha"], "positive": position["beta"]},
    index=["intercept", "slope"],
).round(3)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: "Full and recent training and validation loss histories, with the best validation epoch marked by a vertical line."
---
result.plot_loss_overview()
```

The zero-response coefficients (about −0.59 and 1.14) are close to the
simulated values −0.55 and 1.10. The positive-response coefficients (1.00 and
0.61) are also close to their targets 1.00 and 0.65.

Both loss curves fall sharply in the first few epochs and then fluctuate
around a stable level. The vertical line marks the lowest validation loss;
later updates bring little sustained improvement. The two panels show the
same range here because the run is shorter than the recent-history window.
These curves summarize both groups together, so they cannot show which
response accounts for any remaining lack of fit.

One optimizer can fit both groups. Use [separate optimizers](../../optimizer-customization.md)
only when you want different update rules or learning rates. See
[splitting](../../optimizer-splitting.md) for custom data axes,
[batching](../../optimizer-batching.md) for other epoch lengths, and
[loss scaling](../../optimizer-loss-scaling.md) for how each group contributes.
