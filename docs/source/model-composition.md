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

# Combine models

A variable belongs to one model at a time. Copy a model or its components when
you want an independent version; share variables deliberately when combining
two response models.

## Copy a submodel

For `model` from {doc}`tutorials/notebooks/11-model-building`:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
```

```{code-cell} ipython3
:load: _examples/model-building.py.inc
:tags: [remove-cell]
```

```{code-cell} ipython3
mean_model = model.parental_submodel("mu")
```

```{code-cell} ipython3
sorted(mean_model.vars)
```

The parental submodel contains copies of `mu` and its ancestors: covariate
`x`, coefficients `beta`, and their prior scale. It excludes `y` and the
response variance. Changing the copy does not change the original variables.

## Add a response

This complete example creates two groups of observations:

```{code-cell} ipython3
mean_a = lsl.Var.new_param(
    0.0,
    dist=lsl.Dist(tfd.Normal, loc=0.0, scale=2.0),
    name="mean",
)
y_a = lsl.Var.new_obs(
    jnp.array([0.8, 1.2]),
    dist=lsl.Dist(tfd.Normal, loc=mean_a, scale=1.0),
    name="y_a",
)
group_a = lsl.Model(y_a)

mean_b = lsl.Var.new_param(
    0.0,
    dist=lsl.Dist(tfd.Normal, loc=0.0, scale=2.0),
    name="mean",
)
y_b = lsl.Var.new_obs(
    jnp.array([1.4, 1.6]),
    dist=lsl.Dist(tfd.Normal, loc=mean_b, scale=0.5),
    name="y_b",
)
group_b = lsl.Model(y_b)

separate = group_a.copy()
other = group_b.copy()
other.prefix_names("b_")
separate.add(other, copy=True);
```

```{code-cell} ipython3
---
{"mystnb": {"image": {"alt": "Two independent response groups, each with its own mean parameter."}}}
---
separate.plot(width=7, height=4)
```

```{code-cell} ipython3
sorted(separate.parameters)
```

This model has two means, `mean` and `b_mean`. Prefixing the second group's
names avoids collisions: `add` requires unique names. `copy=True` preserves
the source model; the default `copy=False` moves its contents and empties it.

## Share a parameter

```{code-cell} ipython3
shared = group_a.copy()
shared.join(group_b, by=["mean"], copy=True);
```

```{code-cell} ipython3
sorted(shared.parameters)
```

```{code-cell} ipython3
shared.vars["y_b"].dist_node["loc"] is shared.vars["mean"]
```

```{code-cell} ipython3
---
{"mystnb": {"image": {"alt": "Both response groups depend on one shared mean parameter and its prior."}}}
---
shared.plot(width=7, height=5)
```

There is one `mean` parameter, and the identity check returns `True`. Both
responses depend on the receiving model's mean and its prior. The second
model's same-named mean is replaced, including its prior; joining does not
multiply two priors for the shared parameter.

Names listed in `by` are shared. Other overlapping names receive suffixes
(`.x` and `.y` by default). `copy=True` keeps `group_b` usable;
without it, joining consumes that model. See {meth}`~liesel.model.Model.join`
for the exact collision rules.
