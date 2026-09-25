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

# Initialize VI from Laplace

Use {meth}`~liesel.optim.VDist.mvn_tril_from_laplace` to reuse a fitted
{class}`~liesel.optim.LaplaceApproximation` as the starting distribution for
variational inference. One block gives a dense Gaussian; several blocks give
independent Gaussians with dense covariance within each block.

The example fits two latent effects with a shared population mean to one noisy
measurement of their sum. All parameters are on the real line. This Gaussian
model has an exact Gaussian posterior, so it also makes the difference between
a dense family and independent blocks visible.

## Fit the Laplace model

```{code-cell} python
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

z = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="z")
a_loc = lsl.Var.new_calc(lambda value: jnp.repeat(value, 2), z, name="a_loc")
a = lsl.Var.new_param(
    jnp.zeros(2),
    lsl.Dist(tfd.MultivariateNormalDiag, a_loc, jnp.ones(2)),
    name="a",
)
total = lsl.Var.new_calc(jnp.sum, a, name="total")
y = lsl.Var.new_obs(
    jnp.array([2.0]), lsl.Dist(tfd.Normal, total, 1.0), name="y"
)
model = lsl.Model(y)
```

```{code-cell} python
---
mystnb:
  image:
    alt: "Model graph with population mean z, two latent effects a, their sum, and observed y."
---
model.plot()
```

{class}`~liesel.optim.LaplaceLoss` integrates out `a` and optimizes `z`.
Its joint posterior approximation includes both the outer and latent parameters.
The same initialization API also accepts an approximation from
{meth}`~liesel.optim.NegLogProbLoss.approximate_joint_posterior`.

```{code-cell} python
split = opt.PositionSplit.from_model(model)
laplace_loss = opt.LaplaceLoss(model, split, latent=["a"])
laplace_result = opt.LieselOptim(
    model,
    loss=laplace_loss,
    optimizers="lbfgs",
    loss_monitor="train_full_data",
    stopper=opt.Stopper(epochs=20, patience=3),
    show_progress=False,
).fit()
approximation = laplace_loss.approximate_joint_posterior(laplace_result)
```

```{code-cell} python
pd.Series(
    {name: value.tolist() for name, value in approximation.mean.items()},
    name="Fitted mean",
).to_frame()
```

The fitted means are about 0.57 for `z` and 0.86 for each element of `a`.
The approximation validates curvature and stationarity; initialization rejects
an invalid approximation instead of repairing it with jitter or clipping.

## Use one dense block

Include every approximation parameter to reuse its full joint covariance.
The builder matches names and shapes, then reorders the mean and covariance
into the variational family's flattened order. Transformed parameter names must
match literally; the helper does not transform between parameter scales or
verify that two models use the same data and graph.

```{code-cell} python
dense = (
    opt.VDist(list(approximation.mean), model)
    .mvn_tril_from_laplace(approximation)
    .build()
)
```

```{code-cell} python
pd.DataFrame(
    dense.var.dist_node.init_dist().covariance(),
    index=["a[0]", "a[1]", "z"],
    columns=["a[0]", "a[1]", "z"],
).round(3)
```

The dense family retains the negative dependence between the two effects and
positive dependence between each effect and `z`. In this Gaussian example it
starts at the exact posterior. Sampling noise can still move an optimizer away
from that starting point.

## Use independent blocks

Put `a` and `z` into separate blocks to remove cross-block dependence. For each
selected block S, the initializer uses the fitted mean and covariance
`inverse(P_SS)`, where P is the approximation's joint precision. This is the
covariance conditional on **every omitted approximation parameter at its fitted
mean**, not the block's marginal covariance.

```{code-cell} python
a_block = opt.VDist(["a"], model).mvn_tril_from_laplace(approximation)
z_block = opt.VDist(["z"], model).mvn_tril_from_laplace(approximation)
blocked = opt.CompositeVDist(a_block, z_block).build()
```

```{code-cell} python
pd.DataFrame(
    a_block.var.dist_node.init_dist().covariance(),
    index=["a[0]", "a[1]"],
    columns=["a[0]", "a[1]"],
).round(3)
```

The conditional variance of each effect is 2/3, compared with the marginal
variance 5/7 in the dense family. Dependence inside `a` remains represented.
A block may contain several parameter names; their correlations are retained too.
For a proper subset, the helper works with selected precision rows without
constructing the full joint covariance.

For a Gaussian target, these independent blocks minimize reverse KL under the
ordinary negative ELBO. Independence alone only fixes cross-block covariance to
zero: the family could also represent the product of marginals. Which member is
optimal depends on the objective. For a non-Gaussian posterior, neither good
initialization nor faster convergence is guaranteed. Marginal initialization is
also a legitimate choice; use {meth}`~liesel.optim.VDist.mvn_tril` with an explicit
mean and covariance Cholesky factor when that is your intended starting point.

The helper leaves the target model and approximation unchanged. If omitted target
parameters stay fixed instead of belonging to another variational block, set them
to their fitted means explicitly when you intend the conditional interpretation.
It does not shift the selected means for different fixed values.

## Continue with VI

Pass either family to {meth}`~liesel.optim.NegElboLoss.from_vdist`. This short run
shows the fitting and sampling workflow; inspect loss histories and posterior
predictions before treating a fit as converged.

```{code-cell} python
vi_loss = opt.NegElboLoss.from_vdist(blocked, nsamples=32)
vi_result = opt.LieselVI(
    model,
    loss=vi_loss,
    optimizers=optax.adam(0.001),
    loss_monitor=opt.EmaTrainLossMonitor(effective_window=2.0),
    stopper=opt.Stopper(epochs=30, patience=30),
    seed=42,
    show_progress=False,
).fit()
posterior = vi_loss.approximate_joint_posterior(vi_result)
draws = posterior.sample(2_000, seed=jax.random.key(43))
```

```{code-cell} python
---
mystnb:
  image:
    alt: "VI loss history across 30 epochs, showing Monte Carlo fluctuations around the Gaussian optimum."
---
vi_result.plot_loss()
```

The loss fluctuates because VI estimates the expectation with random draws,
even when its starting family is optimal within the independent-block restriction.
The curve alone cannot establish posterior accuracy.

```{code-cell} python
pd.DataFrame(
    {
        "Mean": [draws["z"].mean(), *draws["a"].mean(axis=0)],
        "SD": [draws["z"].std(), *draws["a"].std(axis=0)],
    },
    index=["z", "a[0]", "a[1]"],
).astype(float).round(3)
```

These draws describe the fitted variational family. They do not recover the
cross-block dependence omitted by its definition. See
{doc}`variational-models` for custom families and {doc}`optimizer-monitoring`
for stopping and fit diagnostics.
