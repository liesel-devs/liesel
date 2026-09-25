---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
---

# Laplace and REML

{py:class}`liesel.optim.LaplaceLoss` integrates selected continuous parameters out
of the model's joint density, then fits the remaining parameters. Here we fit a
Poisson model's mean and random-effect scale while integrating eight group effects.

```{note}
**Relation to REML.** `LaplaceLoss` approximates integration over selected
parameters in smooth models, including non-Gaussian and nonlinear models.
The integrated parameters, priors, and Jacobians determine the target. Here
we integrate the group effects and retain parameter priors, yielding a
marginal posterior.

Classical restricted maximum likelihood (REML) is a special case: integrate
both random effects and fixed-effect coefficients in a Gaussian linear mixed
model, with flat priors on the latter and no additional prior or Jacobian terms
on the remaining parameters. Laplace integration is then exact; see
[Bates et al., Section 3.4](https://lme4.github.io/lme4/articles/lmer.pdf#page=16).
```

## Build the model

Use float64 for this example: enable it before creating arrays and pass
`to_float32=False` to the model.

```{code-cell} ipython3
import logging

import jax
import jax.numpy as jnp
import pandas as pd
import plotnine as p9
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

logging.getLogger("liesel").setLevel(logging.WARNING)

jax.config.update("jax_enable_x64", True)

counts = jnp.array(
    [
        [1.0, 2.0, 1.0, 0.0],
        [3.0, 2.0, 4.0, 3.0],
        [6.0, 5.0, 4.0, 5.0],
        [9.0, 7.0, 10.0, 8.0],
        [13.0, 16.0, 12.0, 15.0],
        [2.0, 4.0, 3.0, 2.0],
        [5.0, 8.0, 6.0, 7.0],
        [20.0, 23.0, 19.0, 21.0],
    ]
)
group = jnp.repeat(jnp.arange(8), 4)

mu = lsl.Var.new_param(1.0, lsl.Dist(tfd.Normal, 0.0, 2.0), name="mu")
tau = lsl.Var.new_param(
    0.5,
    lsl.Dist(tfd.LogNormal, jnp.log(0.5), 0.7),
    bijector=tfb.Exp(),
    name="tau",
)
b = lsl.Var.new_param(jnp.zeros(8), lsl.Dist(tfd.Normal, 0.0, tau), name="b")
log_rate = lsl.Var.new_calc(lambda mu, b: mu + b[group], mu, b, name="log_rate")
y = lsl.Var.new_obs(counts.ravel(), lsl.Dist(tfd.Poisson, log_rate=log_rate), name="y")

model = lsl.Model(y, to_float32=False)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: "Model graph connecting mu and group effects b to Poisson observations y, with tau controlling the group-effect scale."
---
model.plot(width=8, height=6)
```

The group effects `b` enter the log rate; `tau` controls their prior scale.

## Fit the model

```{code-cell} ipython3
loss = opt.LaplaceLoss(model, latent=["b"])
stopper = opt.Stopper(epochs=60, patience=10, rtol=1e-10)
result = opt.LieselOptim(
    model,
    loss=loss,
    optimizers="lbfgs",
    loss_monitor="train_full_data",
    stopper=stopper,
    show_progress=False,
).fit()
```

```{code-cell} ipython3
result.status
```

Use full-data batches and `"train_full_data"` monitoring. The outer parameters
are `mu` and `h(tau)`; `b` keeps its prior. The unscaled loss includes priors,
Jacobians, and normalization constants; see {doc}`optimizer-loss-scaling`.

For observed arrays with different lengths, pass an explicit `split` to
`LaplaceLoss`; {ref}`check the groups and opt in <optimizer-split-groups>` first.

## Inspect group effects

The fit retains the group effects at their conditional mode, together with the
outer parameters that produced them. Recover both from the best recorded fit:

```{code-cell} ipython3
outer = result.position_min_monitor
state = result.loss_state_min_monitor
b_mode = state.latent_position["b"]
```

```{code-cell} ipython3
pd.DataFrame(
    {"estimate": [float(outer["mu"]), float(outer["h(tau)"])]}, index=["mu", "log(tau)"]
).round(3)
```

```{code-cell} ipython3
pd.DataFrame({"conditional_mode": b_mode}, index=range(1, 9)).rename_axis(
    "group"
).round(3)
```

Each effect multiplies the baseline rate `exp(mu)` by `exp(b)`.
For the first group, the multiplier is about `exp(-1.272) = 0.28`;
for the last group, it is about `exp(1.328) = 3.77`.

For the final snapshot, pair `position_final` with `loss_state_final`.
See {py:class}`~liesel.optim.LaplaceState` for convergence diagnostics, parameter
ordering, and the saved curvature in `latent_precision_cholesky`.

## Estimate uncertainty

Turn the completed fit into a joint Gaussian approximation for `mu`, `h(tau)`,
and all eight group effects. It includes their correlations, so each draw is a
complete parameter set that can be passed directly to `Model.predict`:

```{code-cell} ipython3
posterior = loss.approximate_joint_posterior(result)
draws = posterior.sample(1000, seed=jax.random.key(42))
predicted = model.predict(draws, predict=["tau", "log_rate"])
tau_draws = predicted["tau"]
rate_draws = jnp.exp(predicted["log_rate"])
quantiles = jnp.array([0.05, 0.5, 0.95])
```

`Model.predict` transforms draws back to the positive `tau` scale and evaluates
the log rates. For example, summarize the between-group scale with its 5th, 50th,
and 95th percentiles:

```{code-cell} ipython3
pd.DataFrame(
    {"tau": jnp.quantile(tau_draws, quantiles)}, index=["5%", "50%", "95%"]
).round(3)
```

The approximate posterior median is 0.79, with a 90% credible interval of
0.52 to 1.19. The same draws give intervals for each group's expected count:

```{code-cell} ipython3
group_rates = rate_draws[:, ::4]  # Four observations share each group's rate.
lower, median, upper = jnp.quantile(group_rates, quantiles, axis=0)
rate_summary = pd.DataFrame(
    {
        "group": range(1, 9),
        "median": median,
        "lower": lower,
        "upper": upper,
        "observed": counts.mean(axis=1),
    }
)
rate_plot = (
    p9.ggplot(rate_summary, p9.aes(x="group", y="median"))
    + p9.geom_pointrange(p9.aes(ymin="lower", ymax="upper"), color="#1f77b4")
    + p9.geom_point(
        p9.aes(y="observed"),
        shape="x",
        color="black",
        size=2.5,
        stroke=0.8,
        position=p9.position_nudge(x=0.15),
    )
    + p9.scale_x_continuous(breaks=range(1, 9))
    + p9.labs(x="Group", y="Expected count")
    + p9.theme_minimal()
    + p9.theme(figure_size=(7, 3.5))
)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: "Posterior medians and 90 percent credible intervals for eight group rates, with observed means marked by black crosses slightly to the right."
---
rate_plot
```

Blue points and bars show posterior medians and 90% credible intervals;
black crosses mark observed means, shifted slightly to the right for clarity.
The intervals carry uncertainty in the mean, scale, and group effects through
to expected counts. New observations also vary according to the Poisson
distribution.

For matrix calculations, `posterior.covariance()` constructs the full covariance
on request. `posterior.names` and `posterior.shapes` describe its ordering.

To work with one parameter, request its block by name:

```{code-cell} ipython3
b_covariance = posterior.marginal_covariance_blocks(["b"])["b"]
```

```{code-cell} ipython3
pd.DataFrame(b_covariance, index=range(1, 9), columns=range(1, 9)).round(3)
```

This 8 × 8 matrix is the marginal covariance of `b`; it includes the
uncertainty that `mu` and `h(tau)` propagate to `b`.
Omit the name selection to get a dictionary of blocks for every parameter.
For a factor of each block's inverse, use
`posterior.marginal_precision_cholesky_blocks()`.
For precision conditional on all the other parameters, use
`posterior.conditional_precision_blocks()`; these are diagonal blocks of the
joint precision and generally differ from the marginal precisions.

The helper adds curvature work once per call and checks stationarity and positive
definiteness. It selects `at="min_monitor"` by default; use `at="final"` for the
final snapshot. Keep the model, data, and fixed parameters unchanged between fitting
and this call. See {py:meth}`~liesel.optim.LaplaceLoss.approximate_joint_posterior` for controls.

(optimizer-laplace-failure)=

## Inspect a failure

An insufficient inner budget makes a useful diagnostic example:

```{code-cell} ipython3
short_loss = opt.LaplaceLoss(model, latent=["b"], inner_max_iter=1)
failed = opt.LieselOptim(
    model,
    loss=short_loss,
    optimizers="lbfgs",
    loss_monitor="train_full_data",
    show_progress=False,
).fit()
```

```{code-cell} ipython3
failed.status
```

```{code-cell} ipython3
failed.failure_reason
```

`failed.failed_loss_state` holds the inner solver diagnostics. Failures
preserve the last valid snapshot; without one, the loss state is `None`.
Failed fits cannot resume; earlier saved checkpoints remain available.

The posterior helper raises on failure by default. `raise_on_failure=False` returns
an invalid diagnostic object with available gradients, curvature, and the failure
reason:

```{code-cell} ipython3
diagnostic = short_loss.approximate_joint_posterior(failed, raise_on_failure=False)
```

```{code-cell} ipython3
diagnostic.valid
```

Its `sample` and `covariance` methods raise.

## Control warm starts

By default, each inner solve starts from the latent mode committed at the end of
the previous epoch. Set `warm_start=False` to start from the model's latent values:

```{code-cell} ipython3
cold_loss = opt.LaplaceLoss(model, latent=["b"], warm_start=False)
```

Pass `cold_loss` to a new fit to compare. Resuming an interrupted fit is a separate
operation: {doc}`optimizer-checkpointing` explains how to resume with saved states.

## Tune the inner solve

Use `inner_max_iter` to change the iteration budget and `inner_tol` to
control convergence. See {py:class}`~liesel.optim.LaplaceLoss` for their definitions
and defaults.

The solver finds a local mode; warm and cold starts can find different modes.
For d scalar latent parameters, dense curvature uses O(d²) storage and O(d³)
linear algebra.
This version targets a few hundred latents with a modest outer dimension.
