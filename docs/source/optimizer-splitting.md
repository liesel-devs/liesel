---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
---

# Split data

A split decides which rows belong to training, validation, and testing.
Create it before {doc}`configuring minibatches <optimizer-batching>`;
shuffling batches never changes the held-out data. Selecting rows must
preserve their {ref}`likelihood contributions <optimizer-row-wise>`.
The optimizer does not check this property.

The examples assume a Gaussian regression `model` with observed `X` and `y`,
as in the {doc}`first tutorial <tutorials/notebooks/09-liesel-optim-basic>`.

```{code-cell} ipython3
:tags: [remove-cell]

import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

logging.getLogger("liesel").setLevel(logging.WARNING)

rng = np.random.default_rng(42)
x = np.linspace(-1.0, 1.0, 128)

X = lsl.Var.new_obs(jnp.asarray(x), name="X")
beta = lsl.Var.new_param(
    0.0,
    dist=lsl.Dist(tfd.Normal, 0.0, 5.0),
    name="beta",
)

log_sigma = lsl.Var.new_param(0.0, name="log_sigma")
sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")

mu = lsl.Var.new_calc(
    lambda X, beta: X * beta,
    X,
    beta,
    name="mu",
)

y = lsl.Var.new_obs(
    jnp.asarray(0.5 * x + rng.normal(scale=0.7, size=x.size)),
    dist=lsl.Dist(tfd.Normal, mu, sigma),
    name="y",
)
model = lsl.Model(y)
```

(optimizer-split-overview)=

## Keep rows together

Split responses and their covariates together:

```{code-cell} ipython3
split = opt.PositionSplit.from_model(
    model,
    position_keys=["X", "y"],
    validate_axis_share=0.2,
    test_axis_share=0.1,
    seed=42,
)
```

```{code-cell} ipython3
pd.DataFrame(
    {
        "rows": [
            split.train_axis_size,
            split.validate_axis_size,
            split.test_axis_size,
        ],
    },
    index=["training", "validation", "test"],
)
```

This puts 70% of rows in training, 20% in validation, and 10% in testing, subject
to rounding. Splits with validation or test data shuffle by default; set
`shuffle=False` for an ordered split. Without holdouts, splits preserve the
original row order and ignore the seed, even with `shuffle=True`. This is also
the behavior of `LieselOptim`'s automatic full-training split.

The split's `seed` chooses which rows go into each part. The seed passed to
`LieselOptim` controls batch sampling during fitting. Both default to `0`.
Starting parameter values come from the model; seed any random data or starting
values separately.

(optimizer-split-groups)=

## Split several groups

By default, `LieselOptim` raises when observed arrays have different lengths.
This gives you a chance to check which arrays share rows and which are shared
data. For independent groups, opt in to grouping by length and inspect the result:

```{code-cell} ipython3
y_a = lsl.Var.new_obs(
    rng.normal(size=80),
    dist=lsl.Dist(tfd.Normal, 0.0, 1.0),
    name="y_a",
)

y_b = lsl.Var.new_obs(
    rng.normal(size=40),
    dist=lsl.Dist(tfd.Normal, 0.0, 1.0),
    name="y_b",
)
grouped_model = lsl.Model([y_a, y_b])

grouped_split = opt.PositionSplit.from_model(
    grouped_model,
    multi_size="manager",
    validate_axis_share=0.2,
    seed=42,
)
```

```{code-cell} ipython3
pd.DataFrame(
    [
        {
            "keys": group.position_keys,
            "training": group.train_axis_size,
            "validation": group.validate_axis_size,
        }
        for group in grouped_split.splits
    ],
)
```

Pass the checked split to `LieselOptim` as `split=grouped_split`. Matching lengths
do not establish row alignment: flat or omitted `position_keys` group arrays by
length only. To choose groups explicitly, use `PositionSplitManager.from_model`
with nested keys, such as `position_keys=[["X_a", "y_a"], ["X_b", "y_b"]]`.
The {doc}`two-group tutorial <tutorials/notebooks/10-liesel-optim-advanced>` shows
a complete fit.

Arrays within a group share row indices. Different groups split independently,
even if their lengths happen to match. Every group must have validation data if
any group does. Keep shared arrays out of row groups with `split_axes={key: None}`,
as described below.

Automatic grouping requires an observed likelihood in each group. For row data
without a likelihood, supply an explicit nested group.

## Keep shared data

Shared tables and scalar constants must stay unchanged in every split. For a
lookup table `z` indexed by row-level group IDs, use
`PositionSplit.from_model(model, split_axes={"z": None})` and pass the result
as `LieselOptim(..., split=split)`. This also keeps `z` out of automatic batches,
even if its length matches a response. Keep per-row covariates, weights, and
offsets with their response.

For observations on an axis other than zero, set `split_axes` and the
corresponding `batch_axes` when {doc}`creating batches <optimizer-batching>`.

`PositionSplit` holds the split data. `Split` holds reusable row indices;
call `split_position()` to apply them. Manager classes handle several groups.
See {py:meth}`~liesel.optim.PositionSplit.from_model` for the factory options.

For reduced likelihoods or custom objectives, see {doc}`optimizer-loss-scaling`.
For computed observations such as copulas, see {ref}`optimizer-weak-observations`.

<iframe
  class="interactive-visualization"
  data-visualization="split"
  src="_static/visualizations/split-api-overview.html"
  title="Interactive overview of the Liesel split API"
  loading="lazy"
  sandbox="allow-scripts allow-same-origin">
</iframe>

[Open the split API overview in a separate page](_static/visualizations/split-api-overview.html).

Next, {doc}`create minibatches <optimizer-batching>` or read how
{doc}`loss scaling <optimizer-loss-scaling>` handles held-out data.

<script src="_static/visualizations/resize-frames.js"></script>
