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

# Change a model

Use `model` from {doc}`tutorials/notebooks/11-model-building` for these
examples. Make a copy when you want to compare model variants.

## Change a value

```{code-cell} ipython3
:tags: [remove-cell]

%run tutorials/notebooks/11-model-building.ipynb
```

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np

changed = model.copy()
changed.vars["beta"].value = jnp.array([0.5, 1.5])
```

```{code-cell} ipython3
np.asarray(changed.vars["mu"].value[:3]).round(3)
```

```{code-cell} ipython3
round(float(changed.log_lik), 3)
```

The mean and likelihood update automatically. The connections stay the same.
Retrieve variables from `changed` when editing the copy: a Python variable
that points to the original `beta` still belongs to the original model.

## Replace a prior

```{code-cell} ipython3
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl

new_width = lsl.Var.new_value(1.0, name="new_prior_scale")
changed.vars["beta"].dist_node["scale"] = new_width
```

```{code-cell} ipython3
changed.vars["beta"].dist_node["scale"].name
```

The coefficient prior now depends on `new_prior_scale`. To replace the whole
prior, assign a new distribution node:

```{code-cell} ipython3
changed.vars["beta"].dist_node = lsl.Dist(tfd.Normal, loc=0.0, scale=1.5)
changed.rebuild_graph();
```

```{code-cell} ipython3
sorted(changed.vars)
```

The old prior-scale variables disappear because they are no longer inputs of
the model's seed response `y`. `rebuild_graph()` rediscovers the graph from
its seeds; components added with `add_to_seeds=False` can also disappear.
`update()` instead recomputes outdated values without pruning the graph.

The same input indexing works for calculations. Here, `sigma` takes its
variance as its first positional input:

```{code-cell} ipython3
changed.vars["sigma"].value_node[0] = lsl.Var.new_value(0.5, name="fixed_variance")
changed.rebuild_graph();
```

```{code-cell} ipython3
round(float(changed.vars["sigma"].value), 3)
```

The response standard deviation is now about 0.707. This changes one connection.
It also removes the unused variance parameter and its prior from this model.

## Replace a variable

Start from another copy when comparing this change with the previous one:

```{code-cell} ipython3
fixed = model.copy()
fixed.replace("variance", lsl.Var.new_value(0.5, name="variance"));
```

```{code-cell} ipython3
sorted(fixed.parameters)
```

Only `beta` remains a parameter. {meth}`~liesel.model.Model.replace` redirects
uses of the replaced variable throughout the model, whereas assigning one
`value_node` or `dist_node` input changes just that connection. Both choices
change the statistical model here: variance is fixed and its prior is removed.
