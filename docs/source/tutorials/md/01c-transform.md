# Parameter transformations

This tutorial builds on the [linear regression
tutorial](01a-lin-reg.md#linear-regression). Here, we demonstrate how to
transform a positive-valued parameter so that it can be sampled with a
NUTS kernel on an unconstrained scale.

First, let’s set up the linear regression model again. The
data-generating process and the model structure are the same as in the
[linear regression tutorial](01a-lin-reg.md#linear-regression), but this
time we prepare the model for joint NUTS sampling of the regression
coefficients and the error variance.

``` python
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# We use distributions and bijectors from tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import liesel.goose as gs
import liesel.model as lsl

rng = np.random.default_rng(42)

# data-generating process
n = 500
true_beta = np.array([1.0, 2.0])
true_sigma = 1.0
x0 = rng.uniform(size=n)
X_mat = np.c_[np.ones(n), x0]
y_vec = X_mat @ true_beta + rng.normal(scale=true_sigma, size=n)

# Model
# Part 1: Model for the mean
beta_prior = lsl.Dist(tfd.Normal, loc=0.0, scale=100.0)
beta = lsl.Var.new_param(
    value=np.array([0.0, 0.0]),
    dist=beta_prior,
    name="beta",
    inference=gs.MCMCSpec(gs.NUTSKernel, kernel_group="1"),
)

X = lsl.Var.new_obs(X_mat, name="X")
mu = lsl.Var(lsl.Calc(jnp.dot, X, beta), name="mu")

# Part 2: Model for the standard deviation
a = lsl.Var(0.01, name="a")
b = lsl.Var(0.01, name="b")
sigma_sq_prior = lsl.Dist(tfd.InverseGamma, concentration=a, scale=b)
sigma_sq = lsl.Var.new_param(value=10.0, dist=sigma_sq_prior, name="sigma_sq")

sigma = lsl.Var(lsl.Calc(jnp.sqrt, sigma_sq), name="sigma")

# Observation model
y_dist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
y = lsl.Var(y_vec, dist=y_dist, name="y")
```

Now let’s try to sample the regression coefficients $\boldsymbol{\beta}$
and the variance $\sigma^2$ with a single NUTS kernel. NUTS operates on
unconstrained real-valued parameters, whereas $\sigma^2$ must remain
positive. We therefore biject `sigma_sq` with an exponential bijector.
This creates an unconstrained latent variable representing
$\log(\sigma^2)$ and keeps `sigma_sq` as the positive back-transformed
value. Both `beta` and the transformed variance receive NUTS inference
specifications with the same `kernel_group`, so Goose samples them
jointly in one NUTS block.

``` python
sigma_sq.biject(tfb.Exp(), inference=gs.MCMCSpec(gs.NUTSKernel, kernel_group="1"))

model = lsl.Model(y)
model.plot()
```

    liesel.model.model - WARNING - Inconsistent log prob decomposition: Model.log_prob=-1177.35 ≠ (Model.log_lik=0.00 + Model.log_prior=-15.72). 
    liesel.model.model - WARNING - Var(name="y") has a distribution but Var.parameter=False and Var.observed=False.

<img
src="01c-transform_files/figure-commonmark/graph-and-transformation-output-2.png"
id="graph-and-transformation" />

The response distribution still requires the standard deviation on the
original scale. The model graph shows that `sigma_sq` is now a
deterministic, positive-valued transformation of its unconstrained
latent variable. The standard deviation `sigma` is then computed as
`sqrt(sigma_sq)`, so the likelihood continues to receive a valid scale
parameter.

Now we can set up and run the MCMC algorithm directly from the
`MCMCSpec` objects stored in the model. We also include `sigma_sq` in
the stored positions, because the NUTS kernel itself samples the
transformed variable, while `sigma_sq` is the easier quantity to
interpret.

``` python
results = gs.LieselMCMC(model).run_for_epochs(
    seed=1,
    num_chains=4,
    adaptation=1000,
    posterior=1000,
    positions_included=["sigma_sq"],
)
```

    liesel.goose.builder - WARNING - No jitter functions provided for position keys 'h(sigma_sq)', 'beta'. The initial values for these keys won't be jittered
    liesel.goose.engine - INFO - Initializing kernels...
    liesel.goose.engine - INFO - Done
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 100 transitions, 25 jitted together
      0%|                                                  | 0/4 [00:00<?, ?chunk/s] 25%|██████████▌                               | 1/4 [00:01<00:04,  1.51s/chunk]100%|██████████████████████████████████████████| 4/4 [00:01<00:00,  2.63chunk/s]
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 3, 4 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
      0%|                                                  | 0/1 [00:00<?, ?chunk/s]100%|████████████████████████████████████████| 1/1 [00:00<00:00, 1216.45chunk/s]
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 1, 1 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
      0%|                                                  | 0/2 [00:00<?, ?chunk/s]100%|████████████████████████████████████████| 2/2 [00:00<00:00, 1808.28chunk/s]
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 3, 2 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
      0%|                                                  | 0/4 [00:00<?, ?chunk/s]100%|████████████████████████████████████████| 4/4 [00:00<00:00, 2760.32chunk/s]
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 4, 1, 3 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 525 transitions, 25 jitted together
      0%|                                                 | 0/21 [00:00<?, ?chunk/s]100%|███████████████████████████████████████| 21/21 [00:00<00:00, 293.89chunk/s]
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 6, 2, 4, 4 / 525 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 200 transitions, 25 jitted together
      0%|                                                  | 0/8 [00:00<?, ?chunk/s]100%|█████████████████████████████████████████| 8/8 [00:00<00:00, 995.62chunk/s]
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 4, 4, 4, 2 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
      0%|                                                 | 0/40 [00:00<?, ?chunk/s] 78%|██████████████████████████████▏        | 31/40 [00:00<00:00, 306.39chunk/s]100%|███████████████████████████████████████| 40/40 [00:00<00:00, 286.52chunk/s]
    liesel.goose.engine - INFO - Finished epoch

Judging from the trace plots, it seems that all chains have converged.

``` python
gs.plot_trace(results)
```

<img src="01c-transform_files/figure-commonmark/traceplots-output-1.png"
id="traceplots" />

We can also take a look at the summary table, which includes both the
original $\sigma^2$ and the transformed $\log(\sigma^2)$.

``` python
gs.Summary(results)
```

<p>
<strong>Parameter summary:</strong>
</p>
<table border="0" class="dataframe">
<thead>
<tr style="text-align: right;">
<th>
</th>
<th>
</th>
<th>
kernel
</th>
<th>
mean
</th>
<th>
sd
</th>
<th>
q_0.05
</th>
<th>
q_0.5
</th>
<th>
q_0.95
</th>
<th>
sample_size
</th>
<th>
ess_bulk
</th>
<th>
ess_tail
</th>
<th>
rhat
</th>
</tr>
<tr>
<th>
parameter
</th>
<th>
index
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="2" valign="top">
beta
</th>
<th>
(0,)
</th>
<td>
kernel_00
</td>
<td>
0.980
</td>
<td>
0.088
</td>
<td>
0.833
</td>
<td>
0.981
</td>
<td>
1.123
</td>
<td>
4000
</td>
<td>
928.464
</td>
<td>
1263.500
</td>
<td>
1.002
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_00
</td>
<td>
1.915
</td>
<td>
0.155
</td>
<td>
1.663
</td>
<td>
1.915
</td>
<td>
2.175
</td>
<td>
4000
</td>
<td>
906.710
</td>
<td>
1130.257
</td>
<td>
1.002
</td>
</tr>
<tr>
<th>
h(sigma_sq)
</th>
<th>
()
</th>
<td>
kernel_00
</td>
<td>
0.042
</td>
<td>
0.063
</td>
<td>
-0.060
</td>
<td>
0.040
</td>
<td>
0.149
</td>
<td>
4000
</td>
<td>
5655.171
</td>
<td>
2760.321
</td>
<td>
1.000
</td>
</tr>
<tr>
<th>
sigma_sq
</th>
<th>
()
</th>
<td>
\-
</td>
<td>
1.045
</td>
<td>
0.066
</td>
<td>
0.941
</td>
<td>
1.041
</td>
<td>
1.161
</td>
<td>
4000
</td>
<td>
5655.174
</td>
<td>
2760.321
</td>
<td>
1.000
</td>
</tr>
</tbody>
</table>
<p>
<strong>Acceptance probabilities:</strong>
</p>
<table border="0" class="dataframe">
<thead>
<tr style="text-align: right;">
<th>
</th>
<th>
</th>
<th>
</th>
<th>
acceptance_probability
</th>
<th>
position_moved
</th>
</tr>
<tr>
<th>
kernel
</th>
<th>
positions
</th>
<th>
phase
</th>
<th>
</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="2" valign="top">
kernel_00
</th>
<th rowspan="2" valign="top">
h(sigma_sq), beta
</th>
<th>
posterior
</th>
<td>
0.867
</td>
<td>
NaN
</td>
</tr>
<tr>
<th>
warmup
</th>
<td>
0.791
</td>
<td>
NaN
</td>
</tr>
</tbody>
</table>
<p>
<strong>Error summary:</strong>
</p>
<table border="0" class="dataframe">
<thead>
<tr style="text-align: right;">
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
count
</th>
<th>
sample_size
</th>
<th>
sample_size_total
</th>
<th>
relative
</th>
</tr>
<tr>
<th>
kernel
</th>
<th>
positions
</th>
<th>
error_code
</th>
<th>
error_msg
</th>
<th>
phase
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="2" valign="top">
kernel_00
</th>
<th rowspan="2" valign="top">
h(sigma_sq), beta
</th>
<th rowspan="2" valign="top">
1
</th>
<th rowspan="2" valign="top">
divergent transition
</th>
<th>
warmup
</th>
<td>
59
</td>
<td>
4000
</td>
<td>
4000
</td>
<td>
0.015
</td>
</tr>
<tr>
<th>
posterior
</th>
<td>
0
</td>
<td>
4000
</td>
<td>
4000
</td>
<td>
0.000
</td>
</tr>
</tbody>
</table>

Finally, let’s check the autocorrelation of the samples.

``` python
gs.plot_cor(results)
```

<img
src="01c-transform_files/figure-commonmark/correlation-plots-output-1.png"
id="correlation-plots" />
