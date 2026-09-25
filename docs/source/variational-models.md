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

# Build a variational model

Use a Liesel {class}`~liesel.model.Model` as the variational distribution when its
parameters or conditional dependencies are easier to express as a graph. The
Gaussian builders are optional: {class}`~liesel.optim.NegElboLoss` accepts a target
model and a variational model directly.

## Define the target

We fit an intercept and slope to five measurements with known observation scale.
Both parameters are on the real line. The data make the intercept and slope
correlated, so the variational family will allow dependence between their draws.

```{code-cell} python
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

alpha = lsl.Var.new_param(0.0, dist=lsl.Dist(tfd.Normal, 0.0, 2.0), name="alpha")
beta = lsl.Var.new_param(0.0, dist=lsl.Dist(tfd.Normal, 0.0, 2.0), name="beta")

x = lsl.Var.new_value(jnp.arange(5.0), name="x")
mean = lsl.Var.new_calc(
    lambda a, b, x: a + b * x,
    alpha,
    beta,
    x,
    name="mean",
)

y = lsl.Var.new_obs(
    jnp.array([1.1, 1.7, 3.0, 3.8, 5.2]),
    dist=lsl.Dist(tfd.Normal, mean, 0.5),
    name="y",
)
model = lsl.Model(y)
```

```{code-cell} python
---
mystnb:
  image:
    alt: "Regression target graph with intercept alpha, slope beta, predictor x, and observed y."
---
model.plot()
```

## Define conditional draws

The variational model first draws `alpha`, then draws `beta` conditionally on it.
A learned coefficient controls their dependence. Observed variables in `q` are
placeholders for variational draws, not training observations from the target.

```{code-cell} python
alpha_loc = lsl.Var.new_param(
    0.0,
    dist=lsl.Dist(tfd.Normal, 0.0, 1.0),
    name="alpha_loc",
)
beta_loc = lsl.Var.new_param(0.0, name="beta_loc")
dependence = lsl.Var.new_param(0.0, name="dependence")

log_scale = lsl.Var.new_param(-0.7, name="log_scale")
scale = lsl.Var.new_calc(jnp.exp, log_scale, name="scale")

q_alpha = lsl.Var.new_obs(
    0.0, dist=lsl.Dist(tfd.Normal, alpha_loc, scale), name="alpha"
)

beta_mean = lsl.Var.new_calc(
    lambda loc, coefficient, parent: loc + coefficient * parent,
    beta_loc,
    dependence,
    q_alpha,
    name="beta_mean",
)
q_beta = lsl.Var.new_obs(
    0.0, dist=lsl.Dist(tfd.Normal, beta_mean, scale), name="beta"
)
q = lsl.Model(q_beta)
```

```{code-cell} python
---
mystnb:
  image:
    alt: "Variational graph: alpha is drawn first, and beta's mean depends on alpha through a learned coefficient. Strong free inputs control both draws."
---
q.plot(width=8, height=4, legend=False)
```

Mark strong, directly settable inputs as parameters. Derived scales and conditional
means stay computed variables. Optimizing `log_scale` keeps `scale` positive.
{class}`~liesel.optim.NegElboLoss` rejects computed variables marked as parameters;
mark their strong free inputs instead.

After fixing variational parameters, every distribution actually sampled must
belong to an observed `q` variable and support fully reparameterized draws.
Construction rejects unsupported distributions and unclassified sampled variables.
Discrete distributions need a different gradient estimator. Priors on variational
parameters, if present, do not cause those parameters to be sampled. They are
ignored by the default `regularize_q_prior=False`, so the Normal prior attached to
`alpha_loc` above does not change this fit. Opting into their penalties changes
the objective; see {ref}`vi-q-prior-penalties`. Target priors on `alpha` and `beta`
remain included.

## Fit and sample

Here the observed names and shapes in `q` already match the target parameters,
so the identity mapping suffices. An explicit loss must belong to the same target
model object passed to {class}`~liesel.optim.LieselVI`.

```{code-cell} python
loss = opt.NegElboLoss(model, q, nsamples=16, scale=True)

result = opt.LieselVI(
    model,
    loss=loss,
    optimizers=optax.adam(0.01),
    loss_monitor=opt.EmaTrainLossMonitor(2.0),
    stopper=opt.Stopper(epochs=500, patience=500),
    seed=42,
    show_progress=False,
).fit()

posterior = loss.approximate_joint_posterior(result)
draws = posterior.sample(1_000, seed=jax.random.key(43))
```

```{code-cell} python
pd.DataFrame(draws).agg(["mean", "std"]).round(3)
```

```{code-cell} python
---
mystnb:
  image:
    alt: "Loss history for the conditional variational model, including stochastic training loss and its smoothed monitor."
---
result.plot_loss()
```

Inspect loss and parameter paths before using the approximation. A finite fit and
a decreasing monitored loss do not establish posterior accuracy.
The conditional mean of `beta` given `alpha` is
`beta_loc + dependence * alpha`. Sharing one positive conditional scale is a
modeling choice; use separate scale parameters when that restriction is unsuitable.

`entropy="auto"` adds conditional entropies and averages over sampled parents.
`entropy="mc"` uses sampled negative log densities. An explicit loss keeps its own
settings when passed to `LieselVI`. See {doc}`variational-inference` for monitoring
and {doc}`optimizer-customization` for separate optimizer blocks.

(vi-q-prior-penalties)=

## Choose q-parameter penalties

By default, {class}`~liesel.optim.NegElboLoss` optimizes the ordinary ELBO:
the expected target log likelihood **and target log prior**, plus the entropy of
variational draws. `regularize_q_prior=False` does not remove target priors.

A prior attached to a fixed optimization parameter in `q`, such as `alpha_loc`,
is different: it is not part of the density of the draws `alpha` and `beta`.
Set `regularize_q_prior=True` explicitly to add its log density as a penalty.
This can shift the fitted distribution and need not optimize the ordinary ELBO.
Built-in Gaussian families have no such priors, so this flag has no effect on them.

Compare both objectives at the same fitted parameters and with the same draws:

```{code-cell} python
regularized_loss = opt.NegElboLoss(model, q, nsamples=16, regularize_q_prior=True)

params = result.position_final
key = jax.random.key(45)
elbo = loss.estimate_elbo(params, key)
regularized = regularized_loss.estimate_elbo(params, key)
```

```{code-cell} python
pd.DataFrame(
    {"Value": [elbo, regularized, regularized - elbo]},
    index=["Ordinary ELBO", "Regularized objective", "Added q-prior term"],
).astype(float).round(3)
```

The difference is the Normal log prior evaluated at the fitted `alpha_loc`.
To optimize this penalized objective, pass `regularized_loss` as `loss` to
`LieselVI`. An explicit loss keeps its own setting; a wrapper flag does not
modify it. Target-model priors are included in both rows.

## Map names and shapes

For different names or a flattened block, supply `q_to_p` to `NegElboLoss`.
The mapping operates on one draw; Liesel applies it across sample axes. Return
parameter names and original shapes accepted by the target model. The
{class}`~liesel.optim.VDist` builders already provide this unflattening step.

The mapping must be one-to-one and preserve the draw's probability density.
The ELBO does not add a change-of-variables Jacobian for `q_to_p`: use it for
structural renaming and reshaping. Put nonlinear distribution transformations
inside `q`, with their Jacobians included in its density. Auxiliary random
variables cannot simply be discarded without an appropriate target density.
A custom aggregate likelihood in `q` must be the normalized joint log density
of its actual draws. Normalization and arbitrary mapping correctness are the
caller's responsibility; construction cannot verify them mechanically.

Custom transformed families remain subject to the TFP/JAX cache limitation in
[issue #418](https://github.com/liesel-devs/liesel/issues/418).

## Use another objective

`LieselVI` configures ELBO optimization. For a different variational objective,
use {class}`~liesel.optim.OptimEngine` with the public {class}`~liesel.optim.Loss`
interface. {class}`~liesel.optim.LossMixin` supplies gradient helpers. The custom
loss owns its variational model, maps optimizer keys to initial values through
`position`, and evaluates its objective in `loss_train_batched`. Follow the return
contract documented by {meth}`~liesel.optim.LossMixin.loss_train_batched` and
implement `loss_train` for full-data monitoring.

Choose explicit {class}`~liesel.optim.Optimizer` blocks over the custom loss's
parameter keys and supply the evaluation state it expects. Keep sampling and
density calculations in the loss; the engine manages batches, optimizer steps,
monitoring, and checkpoints. This route needs no new variational-distribution
builder or changes to the ELBO implementation.
