
# Location-scale regression

This tutorial implements a Bayesian location-scale regression model
within the Liesel framework. In contrast to the standard linear model
with constant variance, the location-scale model allows for
heteroscedasticity such that both the mean of the response variable as
well as its variance depend on (possibly) different covariates.

This tutorial assumes a linear relationship between the expected value
of the response and the regressors, whereas a logarithmic link is chosen
for the standard deviation. More specifically, we choose the model

$$
\begin{aligned}
y_i \sim \mathcal{N}_{} \left( \mathbf{x}_i^T \boldsymbol{\beta}, \exp \left( \mathbf{ z}_i^T \boldsymbol{\gamma} \right)^2 \right)
\end{aligned}
$$ in which the single observation are conditionally independent.

From the equation we see that *location* covariates are collected in the
design matrix $\mathbf{X}$ and *scale* covariates are contained in the
design matrix $\mathbf{ Z}$. Both matrices can, but generally do not
have to, share common regressors. We refer to $\boldsymbol{\beta}$ as
location parameter and to $\boldsymbol{\gamma}$ as scale parameter.

In this notebook, both design matrices only contain one intercept and
one regressor column. However, the model design naturally generalizes to
any (reasonable) number of covariates.

``` python
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability.substrates.jax.distributions as tfd

sns.set_theme(style="whitegrid")
```

First lets generate the data according to the model

``` python
key = jax.random.PRNGKey(13)
```

    No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

``` python
n = 500

key, key_X, key_Z = jax.random.split(key, 3)

true_beta = jnp.array([1.0, 3.0])
true_gamma = jnp.array([0.0, 0.5])

X_mat = jnp.column_stack([jnp.ones(n), tfd.Uniform(low=0., high=5.).sample(n, seed=key_X)])
Z_mat = jnp.column_stack([jnp.ones(n), tfd.Normal(loc=2., scale=1.).sample(n, seed=key_Z)])

y_vec = jnp.zeros(n)
key_y = jax.random.split(key, n)

y_vec = jax.vmap(
    lambda x, beta, z, gamma, key: tfd.Normal(loc=x @ beta, scale=jnp.exp(z @ gamma)).sample(seed=key),
    (0, None, 0, None, 0))(X_mat, true_beta, Z_mat, true_gamma, key_y)
```

The simulated data displays a linear relationship between the response
$\mathbf{y}$ and the covariate $\mathbf{x}$. The slope of the estimated
regression line is close to the true $\beta_1 = 3$. The right plot shows
the relationship between $\mathbf{y}$ and the scale covariate vector
$\mathbf{z}$. Larger values of $\mathbf{ z}$ lead to a larger variance
of the response.

``` python
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
sns.regplot(
    x=X_mat[:, 1],
    y=y_vec,
    fit_reg=True,
    scatter_kws=dict(color="grey", s=20),
    line_kws=dict(color="blue"),
    ax=ax1,
).set(xlabel="x", ylabel="y", xlim=[-0.2, 5.2])

sns.scatterplot(
    x=Z_mat[:, 1],
    y=y_vec,
    color="grey",
    s=40,
    ax=ax2,
).set(xlabel="z", xlim=[-1, 5.2])

fig.suptitle("Location-Scale Regression Model with Heteroscedastic Error")
fig.tight_layout()
plt.show()
```

![](02-ls-reg_files/figure-commonmark/unnamed-chunk-5-1.png)

Since positivity of the variance is ensured by the exponential function,
the linear part $\mathbf{z}_i^T \boldsymbol{\gamma}$ is not restricted
to the positive real line. Hence, setting a normal prior distribution
for $\gamma$ is feasible, leading to an almost symmetric specification
of the location and scale parts of the model. The variables `beta` and
`gamma` are initialized with values far away from zero to support a
stable sampling process:

``` python
beta_loc = lsl.Var(0.0, name="beta_loc")
beta_scale = lsl.Var(100.0, name="beta_scale")

dist_beta = lsl.Dist(
    distribution=tfd.Normal, loc=beta_loc, scale=beta_scale
)
dist_beta = lsl.Dist(tfd.Normal, loc=beta_loc, scale=beta_scale)

beta = lsl.Param(
    value=jnp.array([10., 10.]), distribution=dist_beta, name="beta"
)
```

``` python
gamma_loc = lsl.Var(0.0, name="gamma_loc")
gamma_scale = lsl.Var(3.0, name="gamma_scale")

dist_gamma = lsl.Dist(
    distribution=tfd.Normal, loc=gamma_loc, scale=gamma_scale
)
gamma = lsl.Param(
    value=jnp.array([5.0, 5.0]), distribution=dist_gamma, name="gamma"
)
```

The additional complexity of the location-scale model compared to the
standard linear model is handled in the next step. Since `gamma` takes
values on the whole real line, but the response variable `y` expects a
positive scale input, we need to apply the exponential function to the
linear predictor to ensure positivity.

``` python
X = lsl.Obs(value=X_mat, name="X")
Z = lsl.Obs(value=Z_mat, name="Z")

mu = lsl.Var(lsl.Calc(lambda X, beta: X @ beta, X, beta), name="mu")
scale = lsl.Var(lsl.Calc(lambda Z, gamma: jnp.exp(Z @ gamma), Z, gamma), name="scale")

dist_y = lsl.Dist(distribution=tfd.Normal, loc=mu, scale=scale)
y = lsl.Obs(value=y_vec, distribution=dist_y, name="y")
```

We can now combine the nodes in a model and visualize it

``` python
sns.set_theme(style="white")

gb = lsl.GraphBuilder()
gb.add(y)
```

    GraphBuilder<0 nodes, 1 vars>

``` python
model = gb.build_model() # builds the model from the graph (PGMs)

lsl.plot_vars(model=model, width=12, height=8)
```

![](02-ls-reg_files/figure-commonmark/unnamed-chunk-9-3.png)

We choose the No U-Turn sampler for generating posterior samples.
Therefore the location and scale parameters can be drawn by separate
NUTS kernels, or, if all remaining inputs to the kernel coincide, by one
common kernel. The latter option might lead to better estimation results
but lacks the flexibility to e.g. choose different step sizes during the
sampling process.

However, we will just fuse everything into one kernel do not use any
specific arguments and hope that the default warmup scheme (similar to
the warmup used in STAN) will do the trick.

``` python
builder = gs.EngineBuilder(seed=73, num_chains=4)

# connects the engine with the model
builder.set_model(lsl.GooseModel(model))

# we use the same initial values for all chains
builder.set_initial_values(model.state)

# add the kernel
builder.add_kernel(gs.NUTSKernel(["beta", "gamma"]))

# set number of iterations in warmup and posterior
builder.set_duration(warmup_duration=1500, posterior_duration=1000, term_duration=500)

# create the engine
engine = builder.build()

# generate samples
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 4, 7, 6, 11 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 1, 2, 3 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 3, 3, 1 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 3, 3, 4 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 5, 2, 7, 4 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 550 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 5, 3, 6, 3 / 550 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 500 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 8, 3, 5, 3 / 500 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch

Now that we have 1000 posterior samples per chain, we can check the
results. Starting with the trace plots just using one chain.

``` python
results = engine.get_results()
g = gs.plot_trace(results, chain_indices=0, ncol=4)
```

![](02-ls-reg_files/figure-commonmark/unnamed-chunk-11-5.png)

Looks decent although we can see some correlation in the tracplots.
Let’s check at the combined summary:

``` python
gs.summary_m.Summary(results, per_chain=False)
```

<div class="cell-output-display">

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
0.876259
</td>
<td>
0.180972
</td>
<td>
0.575500
</td>
<td>
0.877072
</td>
<td>
1.176521
</td>
<td>
4000
</td>
<td>
1709.596175
</td>
<td>
1627.075607
</td>
<td>
1.001248
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
3.003814
</td>
<td>
0.063462
</td>
<td>
2.899862
</td>
<td>
3.004124
</td>
<td>
3.107582
</td>
<td>
4000
</td>
<td>
1803.239892
</td>
<td>
1826.049184
</td>
<td>
1.000803
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
gamma
</th>
<th>
(0,)
</th>
<td>
kernel_00
</td>
<td>
-0.044902
</td>
<td>
0.071244
</td>
<td>
-0.160150
</td>
<td>
-0.045564
</td>
<td>
0.072341
</td>
<td>
4000
</td>
<td>
1758.819174
</td>
<td>
1746.003713
</td>
<td>
1.000879
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
0.516198
</td>
<td>
0.031559
</td>
<td>
0.464533
</td>
<td>
0.516380
</td>
<td>
0.567629
</td>
<td>
4000
</td>
<td>
1799.915611
</td>
<td>
1919.756384
</td>
<td>
1.000471
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
count
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
</tr>
</thead>
<tbody>
<tr>
<th rowspan="2" valign="top">
kernel_00
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
113
</td>
<td>
0.018833
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
0.000000
</td>
</tr>
</tbody>
</table>

</div>

Maybe a longer warm-up would give us better samples.

``` python
builder = gs.EngineBuilder(seed=3, num_chains=4)

# connects the engine with the model
builder.set_model(lsl.GooseModel(model))

# we use the same initial values for all chains
builder.set_initial_values(model.state)

# add the kernel
builder.add_kernel(gs.NUTSKernel(["beta", "gamma"]))

# set number of iterations in warmup and posterior
builder.set_duration(warmup_duration=4000, posterior_duration=1000, term_duration=1000)

# create the engine
engine = builder.build()

# generate samples
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 7, 8, 8, 11 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 2, 2, 1 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 2, 1, 2 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 4, 2, 4 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 5, 5, 1, 3 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 4, 7, 6, 6 / 400 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 2150 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 5, 3, 3, 1 / 2150 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 1000 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 3, 5, 4 / 1000 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch

``` python
results = engine.get_results()
g = gs.plot_trace(results, chain_indices=0, ncol=4)
```

![](02-ls-reg_files/figure-commonmark/unnamed-chunk-14-7.png)

``` python
gs.summary_m.Summary(results, per_chain=False)
```

<div class="cell-output-display">

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
0.873678
</td>
<td>
0.172997
</td>
<td>
0.595108
</td>
<td>
0.872357
</td>
<td>
1.166812
</td>
<td>
4000
</td>
<td>
1905.381275
</td>
<td>
1609.181917
</td>
<td>
1.005724
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
3.003912
</td>
<td>
0.060999
</td>
<td>
2.902545
</td>
<td>
3.004837
</td>
<td>
3.100385
</td>
<td>
4000
</td>
<td>
1815.532371
</td>
<td>
2088.351006
</td>
<td>
1.004318
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
gamma
</th>
<th>
(0,)
</th>
<td>
kernel_00
</td>
<td>
-0.043378
</td>
<td>
0.070821
</td>
<td>
-0.159004
</td>
<td>
-0.042451
</td>
<td>
0.071857
</td>
<td>
4000
</td>
<td>
1881.848828
</td>
<td>
1865.732306
</td>
<td>
1.004667
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
0.515433
</td>
<td>
0.031271
</td>
<td>
0.464488
</td>
<td>
0.515818
</td>
<td>
0.567116
</td>
<td>
4000
</td>
<td>
1864.979460
</td>
<td>
2106.394241
</td>
<td>
1.004497
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
count
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
</tr>
</thead>
<tbody>
<tr>
<th rowspan="2" valign="top">
kernel_00
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
123
</td>
<td>
0.007688
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
0.000000
</td>
</tr>
</tbody>
</table>

</div>

The trace plots for $\boldsymbol{\gamma}$ improved but those for
$\boldsymbol{\beta}$ still show some corelation.