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

# Choose a variational family

Start with {class}`~liesel.optim.VDist` to approximate a selected group of target
parameters, or combine several groups with {class}`~liesel.optim.CompositeVDist`.
These helpers build the variational model, create its trainable parameters, and
map draws back to the target's names and shapes. They let you choose which
posterior dependencies the family can represent.

This guide assumes the fitting and sampling workflow in
{doc}`variational-inference`. It first builds Gaussian families for a regression
model. The {ref}`advanced graph example <vi-custom-graph>` then shows how to use
an ordinary Liesel {class}`~liesel.model.Model` for conditional distributions that
you want to define directly.

## Define the target

We fit an intercept and slope to five measurements with known observation scale.
Both parameters are on the real line. The data make the intercept and slope
correlated, so a dense Gaussian can represent dependence that an independent
family omits.

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

(vi-gaussian-shortcuts)=

## Use Gaussian loss shortcuts

When one Gaussian family should cover **all** target parameters, construct the
loss directly. Both shortcuts build a `VDist` internally and connect it to the
negative ELBO; no separate `build()` call is needed.

Use {meth}`~liesel.optim.NegElboLoss.mvn_diag` for independent components, each
with its own learned mean and standard deviation:

```{code-cell} python
diagonal_loss = opt.NegElboLoss.mvn_diag(
    model,
    scale_diag=0.5,
    nsamples=16,
    scale=True,
)
```

Use {meth}`~liesel.optim.NegElboLoss.mvn_tril` to learn correlations as well:

```{code-cell} python
correlated_loss = opt.NegElboLoss.mvn_tril(
    model,
    scale_tril=0.5,
    nsamples=16,
    scale=True,
)
```

`scale_diag` contains initial standard deviations. `scale_tril` is an initial
lower Cholesky factor: its covariance is `scale_tril @ scale_tril.T`. For either
shortcut, the scalar `0.5` starts with SD 0.5 in every component. The dense family
starts with zero correlations but can learn them; the diagonal family cannot.
A dense factor takes quadratic storage in the total number of scalar components,
whereas the diagonal scale takes linear storage.

Both losses include `alpha` and `beta` here, using their current target values as
initial means. Pass either loss to the same fitting and sampling workflow in
{doc}`variational-inference`. The built helper is accessible as `loss.vdist`,
and its variational graph as `loss.q`. The Boolean `scale=True` normalizes the
training loss; it is independent of the Gaussian's initial scale.

Use an explicit `VDist` when selecting a subset of target parameters or supplying
a custom distribution. Use `CompositeVDist` to choose independent groups, as
shown below. {meth}`~liesel.optim.NegElboLoss.mvn_blocked` is also available when
you want exactly one dense block per target parameter: vector components can
correlate within a parameter, but different parameter names remain independent.

## Build one Gaussian block

A `VDist` governs the target names given in its first argument. One block may
contain several parameters, including vectors and matrices. Internally it flattens
their values; its sampling interface restores the original names and shapes.
Use the names on the unconstrained scale for transformed target parameters.

Choose a Gaussian form according to the dependence you want to represent:

| Builder | Variational family | Dependence within the block |
| --- | --- | --- |
| `mvn_diag(scale_diag=...)` | One multivariate Normal with diagonal covariance | None |
| `mvn_tril(scale_tril=...)` | One multivariate Normal with dense covariance | Learned correlations |

The constructors on `VDist` configure a block; unlike the loss shortcuts above,
they return the helper itself, not a loss. For independent `alpha` and `beta`, use:

```{code-cell} python
diagonal = opt.VDist(["alpha", "beta"], model).mvn_diag(scale_diag=0.5).build()
```

To let the fit learn their correlation, use the same names in a dense block:

```{code-cell} python
dense = opt.VDist(["alpha", "beta"], model).mvn_tril(scale_tril=0.5).build()
```

```{code-cell} python
dense
```

`mvn_tril` creates the trainable locations and Cholesky factor. The scalar
`scale_tril=0.5` starts that factor at `0.5 * I`; fitting can learn nonzero
correlations. Locations start at the current target values, with a separate learned
mean for each flattened component. A scalar `loc` would instead tie their means;
use a vector if supplying distinct initial means. See {ref}`vi-initial-scale` for
choosing scales in your parameter units.

`build()` creates the underlying Liesel model in `dense.q` and returns the same
helper. Its `parameters` property lists the variational parameter names that
optimizers adjust, rather than the target names `alpha` and `beta`:

```{code-cell} python
dense.parameters
```

You can sample the initial family to inspect its shape before fitting:

```{code-cell} python
initial_draws = dense.sample(4, seed=jax.random.key(40))
```

```{code-cell} python
pd.DataFrame(initial_draws)
```

Each row contains one draw in the target's representation. These are initial
variational draws, not draws from a fitted posterior.

## Combine independent blocks

`CompositeVDist` combines initialized `VDist` objects into one variational model.
It has no `.mvn_diag()` or `.mvn_tril()` constructor of its own: choose the
distribution on each `VDist` block. Each block may use a different distribution
or initial scale. Blocks must refer
to the same target model, and no target name may appear in more than one block.
Initialize the individual blocks, then call `build()` on the composite:

```{code-cell} python
alpha_block = opt.VDist(["alpha"], model).mvn_diag(scale_diag=0.5)
beta_block = opt.VDist(["beta"], model).mvn_diag(scale_diag=0.2)
blocked = opt.CompositeVDist(alpha_block, beta_block).build()
```

```{code-cell} python
blocked
```

Here `alpha` and `beta` are independent under the variational distribution, even
though the target posterior correlates them. Fitting can change their means and
scales but cannot introduce that missing dependence. This is the same family as
one diagonal Gaussian with separately learned means and scales.

For larger models, a useful compromise is to put related parameters together in
an `mvn_tril` block and keep other groups in separate blocks. Dense blocks can
learn dependence within their groups; the composite imposes independence between
groups. Grouping also limits covariance storage to the individual blocks.
Independence is a restriction on the family, not a promise to recover the target's
marginal variances: the negative ELBO determines the best-fitting member.

Cover every parameter whose posterior you want to approximate. Any target
parameters omitted from the family stay fixed at their current target values;
Liesel does not silently create extra blocks for them.

## Connect a family to the loss

Both helpers use the same {meth}`~liesel.optim.NegElboLoss.from_vdist` interface.
It takes the target model, variational model, and draw mapping from the built
helper:

```{code-cell} python
dense_loss = opt.NegElboLoss.from_vdist(dense, nsamples=16, scale=True)
blocked_loss = opt.NegElboLoss.from_vdist(blocked, nsamples=16, scale=True)
```

Pass either object as `loss` to `LieselVI(model, ...)`, following
{doc}`variational-inference`. After fitting, use
`loss.approximate_joint_posterior(result).sample(..., seed=key)` for draws at the
fitted parameters. Calling `dense.sample(...)` or `blocked.sample(...)` without
an `at_position` still uses that helper's current model values, which fitting
does not update.

For a custom distribution over one flattened block, use
{meth}`~liesel.optim.VDist.init` with an `lsl.Dist` and mark its free variational
parameters with `lsl.Var.new_param`. The API reference includes an example.
The sampled distribution must support fully reparameterized draws. When the
variational distribution itself needs conditional dependencies, build its graph
directly as below.

(vi-custom-graph)=

## Advanced: build a custom graph

The helpers cover independent blocks and their internal distributions. Building
an ordinary Liesel model gives you direct control over conditional dependencies
and shared variational parameters. This requires distinguishing the target's
random quantities from the free parameters that describe their variational
distribution, and checking the density and draw mapping yourself.

### Define conditional draws

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
q.plot(width=8, height=7, legend=False)
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

### Fit and sample

Here the observed names and shapes in `q` already match the target parameters,
so the identity mapping suffices. An explicit loss must belong to the same target
model object passed to {class}`~liesel.optim.LieselVI`.

```{code-cell} python
loss = opt.NegElboLoss(model, q, nsamples=16, scale=True)

result = opt.LieselVI(
    model,
    loss=loss,
    optimizers=optax.adam(0.01),
    loss_monitor=opt.EmaTrainLossMonitor(effective_window=20.0),
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

### Choose q-parameter penalties

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

### Map names and shapes

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
