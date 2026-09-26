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

# Inspect a model

The examples below use `model` from
{doc}`tutorials/notebooks/11-model-building`, with `beta`, `variance`,
`sigma`, `mu`, `x`, and `y`. Inspect the model at its current values
before fitting it.

## Find inputs

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
sorted(model.parameters)
```

```{code-cell} ipython3
sorted(model.observed)
```

```{code-cell} ipython3
response = model.vars["y"]
```

```{code-cell} ipython3
response.dist_node["scale"].name
```

```{code-cell} ipython3
model.vars["sigma"].value_node[0].name
```

The parameters are `beta` and `variance`; the observed variables are `x`
and `y`. Their order in the mappings is not significant. The two input
lookups return `sigma` and `variance`. `"scale"` is the distribution's
argument name, whereas `"sigma"` is the variable's name. Use an integer for
an input passed positionally and its argument name for an input passed by keyword.

Strong variables receive their values directly; weak variables calculate them
from other variables. Observed and parameter flags describe a separate choice:
the statistical role of a variable. A weak variable can also carry a likelihood
or prior; see {doc}`model-distributions`.

## Read log densities

```{code-cell} ipython3
response.log_prob.shape
```

```{code-cell} ipython3
{
    "likelihood": round(float(model.log_lik), 3),
    "prior": round(float(model.log_prior), 3),
    "total": round(float(model.log_prob), 3),
}
```

```{code-cell} ipython3
model.vars["x"].log_prob
```

The response contributes one log density per observation in this scalar normal
example. `model.log_lik` sums densities on observed variables;
`model.log_prior` sums densities on parameters. Here their sum is
`model.log_prob`. Variables without a distribution, such as `x`, contribute
zero. A parameter without a distribution has a constant prior contribution;
on an unbounded domain this is an improper prior, with no distribution to sample.

The total normally sums all distribution nodes. A distribution not assigned
an observed or parameter role can therefore break the likelihood/prior
decomposition. Custom aggregate nodes can also change it. See
{doc}`model-distributions` before using such a model for optimization.

## Trace invalid values

Probe a copy so the original model keeps its valid values:

```{code-cell} ipython3
probe = model.copy()
probe.vars["variance"].value = -1.0
diagnostics = probe.diagnose().sort_values("name")
```

```{code-cell} ipython3
diagnostics[["name", "value_n_nan", "log_prob_n_nan", "log_prob_n_inf"]]
```

```{code-cell} ipython3
probe.vars["variance"].value = 1.0
```

The negative variance makes its square root invalid. The table lets you trace
the NaN from `sigma` into the response density. Missing entries (`NaN`) mean that the
quantity does not apply, for example a log density on a variable without a
distribution. An infinite log density can instead signal a value outside a
distribution's support; check the distribution and its inputs rather than
replacing every non-finite result with a number.

Use {meth}`~liesel.model.Model.plot` to follow dependencies and
{meth}`~liesel.model.Model.plot_nodes` when you need to see the underlying value
and distribution nodes separately. Finite values establish neither a suitable
statistical model nor convergence of an inference algorithm.
