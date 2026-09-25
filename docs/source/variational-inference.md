---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
  execution_raise_on_error: true
---

# Variational inference

Use {class}`liesel.optim.LieselVI` to fit an approximate posterior for a Liesel
model. It minimizes the negative evidence lower bound (ELBO) by adjusting the
parameters of a variational distribution. The fitted values describe that
distribution; draw samples from it to summarize the target model's parameters.

## Fit a Gaussian family

This example estimates a Normal mean with known observation scale. All parameters
are on the real line. Transform constrained parameters before constructing the
variational family, and transform draws back when summarizing them.

```{code-cell} python
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

loc = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="loc")
y = lsl.Var.new_obs(
    jnp.array([0.9, 1.4, 0.8, 1.2, 0.7, 1.0]),
    lsl.Dist(tfd.Normal, loc, 1.0),
    name="y",
)
model = lsl.Model(y)
```

```{code-cell} python
---
mystnb:
  image:
    alt: "Normal mean model: the parameter loc determines the location of observed y."
---
model.plot()
```

Choose an initial scale appropriate for the parameter's units. Here 0.5 is close
to the posterior scale. We use an explicit iteration budget because stochastic
loss fluctuations can trigger early stopping before the distribution stabilizes.
Sixteen draws per step estimate the ELBO; more draws cost more computation.

```{code-cell} python
loss = opt.NegElboLoss.mvn_diag(model, nsamples=16, scale_diag=0.5, scale=True)
result = opt.LieselVI(
    model,
    loss=loss,
    optimizers=optax.adam(0.01),
    loss_monitor=opt.EmaTrainLossMonitor(effective_window=2.0),
    stopper=opt.Stopper(epochs=300, patience=300),
    seed=42,
    show_progress=False,
).fit()
```

```{code-cell} python
---
mystnb:
  image:
    alt: "Training and smoothed monitoring losses over the variational fit."
---
result.plot_loss()
```

`mvn_diag` creates a diagonal Gaussian over the model parameters and stores its
{class}`~liesel.optim.VDist` in `loss.vdist`. Fitting leaves the supplied model
unchanged. Inspect the curve and posterior predictions before treating the fit
as converged; a decreasing loss alone does not establish posterior accuracy.

## Sample the fitted family

```{code-cell} python
posterior = loss.approximate_joint_posterior(result)
samples = posterior.sample(2_000, seed=jax.random.key(43))
```

```{code-cell} python
pd.DataFrame(
    {"Mean": [samples["loc"].mean()], "SD": [samples["loc"].std()]},
    index=["loc"],
).astype(float).round(3)
```

The exact posterior mean in this example is 6/7 and its standard deviation is
1/sqrt(7). Monte Carlo summaries fluctuate around the fitted family's values.

`approximate_joint_posterior` selects the saved position with the smallest
monitoring loss by default. Pass `at="final"` to select the last iterate instead.
The returned {class}`~liesel.optim.VariationalApproximation` keeps those values
without retaining fit history. Variational parameters omitted from the fit are
bound to their current model values. Keep its variational graph, nonparameter
values, and position mapping unchanged, and use a result from the same loss.

Sampling uses the learned distribution, including custom families and mappings.
It does not refit the model or compute a Hessian. Finite fitted values do not
certify convergence. The default `sample(seed=key)` returns one draw in the
original parameter shapes. An integer or tuple adds leading sample axes:

```{code-cell} python
samples = posterior.sample((1, 1_000), seed=jax.random.key(44))
predicted = model.predict(samples)
```

```{code-cell} python
samples["loc"].shape
```

These leading axes also work when predicting derived quantities or transforming
constrained parameters. The fitted approximation API also accepts a directly
constructed {class}`~liesel.optim.NegElboLoss` without a `VDist`. Custom transformed
families remain subject to [issue #418](https://github.com/liesel-devs/liesel/issues/418).

Use `mvn_tril` for a dense covariance or {class}`~liesel.optim.CompositeVDist` for
independent blocks. An explicit loss keeps its own sample count, scaling, entropy,
and prior settings; configure them on that loss.

## Choose a monitor

Both `optimizers` and `loss_monitor` are required. An
{class}`~liesel.optim.EmaTrainLossMonitor` smooths losses evaluated before optimizer
updates and carries the average across epochs. `"train_full_data"` adds an
evaluation on all training rows after each epoch. It still draws variational
samples and can fluctuate without minibatches. Analytic entropy reduces one
source of Monte Carlo noise; it does not make the whole ELBO deterministic.
See {doc}`optimizer-monitoring` for stopping and histories.

ELBO losses do not support validation splits. Use a train/test split for a final
predictive check, as in the basic tutorial. `position_final` is the final iterate;
`position_min_monitor` is the saved epoch-end position with the smallest monitoring
value. With an EMA, that value combines losses from several positions.

## Work through examples

```{toctree}
:maxdepth: 1

variational-models
Laplace initialization <variational-laplace>
tutorials/notebooks/11-liesel-vi-basic
tutorials/notebooks/12-liesel-vi-advanced
```

## Configure the fit

* {doc}`optimizer-splitting` explains explicit splits, axes, and seeds.
* {doc}`optimizer-batching` covers aligned row groups and fixed computed data.
* {doc}`optimizer-loss-scaling` explains sample counts and normalization.
* {doc}`optimizer-weighted-batching` covers unequal sampling probabilities.
* {doc}`optimizer-customization` covers learning rates and parameter blocks. For
  VI, explicit optimizers select names in `loss.q.parameters`.
* {doc}`optimizer-checkpointing` explains how to resume a fit.

For arguments, ELBO estimation, and sampling, see {class}`~liesel.optim.LieselVI`,
{class}`~liesel.optim.NegElboLoss`, {class}`~liesel.optim.VDist`, and
{class}`~liesel.optim.CompositeVDist` in the {ref}`optimizer-api`.
