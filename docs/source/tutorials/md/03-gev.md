
# GEV responses

In this tutorial, we illustrate how to set up a distributional
regression model with the generalized extreme value distribution as a
response distribution. First, we simulate some data in R:

- The location parameter ($\mu$) is a function of an intercept and a
  non-linear covariate effect.
- The scale parameter ($\sigma$) is a function of an intercept and a
  linear effect and uses a log-link.
- The shape or concentration parameter ($\xi$) is a function of an
  intercept and a linear effect.

After simulating the data, we can configure the model with a single call
to the `rliesel::liesel()` function.

``` r
library(rliesel)
```

    Please set your Liesel venv, e.g. with use_liesel_venv()

``` r
library(VGAM)
```

    Loading required package: stats4

    Loading required package: splines

``` r
set.seed(1337)

n <- 1000

x0 <- runif(n)
x1 <- runif(n)
x2 <- runif(n)

y <- rgev(
  n,
  location = 0 + sin(2 * pi * x0),
  scale = exp(-3 + x1),
  shape = 0.1 + x2
)

plot(y)
```

![](03-gev_files/figure-commonmark/model-1.png)

``` r
model <- liesel(
  response = y,
  distribution = "GeneralizedExtremeValue",
  predictors = list(
    loc = predictor(~ s(x0)),
    scale = predictor(~ x1, inverse_link = "Exp"),
    concentration = predictor(~ x2)
  )
)
```

Now, we can continue in Python and use the `lsl.dist_reg_mcmc()`
function to set up a sampling algorithm with IWLS kernels for the
regression coefficients ($\boldsymbol{\beta}$) and a Gibbs kernel for
the smoothing parameter ($\tau^2$) of the spline. Note that we need to
set $\beta_0$ for $\xi$ to 0.1 manually, because $\xi = 0$ breaks the
sampler.

``` python
import liesel.model as lsl
import jax.numpy as jnp

model = r.model

# concentration == 0.0 seems to break the sampler
model.vars["concentration_p0_beta"].value = jnp.array([0.1, 0.0])

builder = lsl.dist_reg_mcmc(model, seed=42, num_chains=4)
builder.set_duration(warmup_duration=1000, posterior_duration=1000)

engine = builder.build()
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 8, 12, 10, 3 / 75 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 1, 0, 0, 1 / 75 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 0, 0, 1 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 0, 1, 0, 3 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 1, 1, 1, 0 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 1, 2, 1 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 2, 1, 1, 1 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 2, 0 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 0, 0, 1, 1 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 2, 0, 2, 1 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 1, 1, 0 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 3, 1, 2 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 1, 1, 1, 1 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 3, 2, 1 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 2, 2, 1 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 3, 1, 1 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 0, 1, 2, 1 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 1, 1, 2 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 2, 2, 1, 2 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 500 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 4, 1, 1 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 1, 3, 1, 1 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 1, 1, 2 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 2, 1, 2, 2 / 500 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 2, 0, 1 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 0, 1, 0, 0 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 1, 1, 1 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 3, 0, 0, 1 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 0, 1, 2 / 1000 transitions
    liesel.goose.engine - INFO - Finished epoch

Some tabular summary statistics of the posterior samples:

``` python
import liesel.goose as gs

results = engine.get_results()
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
concentration_p0_beta_value
</th>
<th>
(0,)
</th>
<td>
kernel_00
</td>
<td>
0.070828
</td>
<td>
0.050930
</td>
<td>
-0.011126
</td>
<td>
0.070087
</td>
<td>
0.157356
</td>
<td>
4000
</td>
<td>
272.012363
</td>
<td>
680.371746
</td>
<td>
1.018061
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
1.062463
</td>
<td>
0.100716
</td>
<td>
0.895923
</td>
<td>
1.064303
</td>
<td>
1.227816
</td>
<td>
4000
</td>
<td>
135.632390
</td>
<td>
503.175908
</td>
<td>
1.022766
</td>
</tr>
<tr>
<th rowspan="9" valign="top">
loc_np0_beta_value
</th>
<th>
(0,)
</th>
<td>
kernel_03
</td>
<td>
-0.618549
</td>
<td>
0.231815
</td>
<td>
-1.001004
</td>
<td>
-0.626966
</td>
<td>
-0.235181
</td>
<td>
4000
</td>
<td>
77.339280
</td>
<td>
125.899778
</td>
<td>
1.053870
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_03
</td>
<td>
0.308085
</td>
<td>
0.121566
</td>
<td>
0.120156
</td>
<td>
0.301655
</td>
<td>
0.510482
</td>
<td>
4000
</td>
<td>
124.202149
</td>
<td>
235.106801
</td>
<td>
1.040677
</td>
</tr>
<tr>
<th>
(2,)
</th>
<td>
kernel_03
</td>
<td>
-0.376483
</td>
<td>
0.127559
</td>
<td>
-0.625379
</td>
<td>
-0.365348
</td>
<td>
-0.183140
</td>
<td>
4000
</td>
<td>
81.750175
</td>
<td>
91.508421
</td>
<td>
1.052561
</td>
</tr>
<tr>
<th>
(3,)
</th>
<td>
kernel_03
</td>
<td>
0.357412
</td>
<td>
0.063043
</td>
<td>
0.247995
</td>
<td>
0.359298
</td>
<td>
0.461894
</td>
<td>
4000
</td>
<td>
77.374009
</td>
<td>
183.947778
</td>
<td>
1.053452
</td>
</tr>
<tr>
<th>
(4,)
</th>
<td>
kernel_03
</td>
<td>
-0.253001
</td>
<td>
0.079011
</td>
<td>
-0.386524
</td>
<td>
-0.251439
</td>
<td>
-0.127248
</td>
<td>
4000
</td>
<td>
30.062791
</td>
<td>
120.618003
</td>
<td>
1.121351
</td>
</tr>
<tr>
<th>
(5,)
</th>
<td>
kernel_03
</td>
<td>
0.175876
</td>
<td>
0.030534
</td>
<td>
0.127078
</td>
<td>
0.175883
</td>
<td>
0.223764
</td>
<td>
4000
</td>
<td>
78.551570
</td>
<td>
124.219526
</td>
<td>
1.073983
</td>
</tr>
<tr>
<th>
(6,)
</th>
<td>
kernel_03
</td>
<td>
6.026882
</td>
<td>
0.039856
</td>
<td>
5.963779
</td>
<td>
6.025156
</td>
<td>
6.095318
</td>
<td>
4000
</td>
<td>
78.746774
</td>
<td>
202.790551
</td>
<td>
1.027015
</td>
</tr>
<tr>
<th>
(7,)
</th>
<td>
kernel_03
</td>
<td>
0.531174
</td>
<td>
0.066872
</td>
<td>
0.423115
</td>
<td>
0.532653
</td>
<td>
0.644100
</td>
<td>
4000
</td>
<td>
30.309528
</td>
<td>
201.283824
</td>
<td>
1.116477
</td>
</tr>
<tr>
<th>
(8,)
</th>
<td>
kernel_03
</td>
<td>
1.700920
</td>
<td>
0.031162
</td>
<td>
1.651670
</td>
<td>
1.700472
</td>
<td>
1.753882
</td>
<td>
4000
</td>
<td>
80.955794
</td>
<td>
196.447041
</td>
<td>
1.030534
</td>
</tr>
<tr>
<th>
loc_np0_tau2_value
</th>
<th>
()
</th>
<td>
kernel_02
</td>
<td>
6.286887
</td>
<td>
5.884435
</td>
<td>
2.424950
</td>
<td>
5.091027
</td>
<td>
13.521162
</td>
<td>
4000
</td>
<td>
3630.239951
</td>
<td>
3753.971108
</td>
<td>
0.999621
</td>
</tr>
<tr>
<th>
loc_p0_beta_value
</th>
<th>
(0,)
</th>
<td>
kernel_04
</td>
<td>
0.027247
</td>
<td>
0.002491
</td>
<td>
0.023125
</td>
<td>
0.027234
</td>
<td>
0.031493
</td>
<td>
4000
</td>
<td>
92.138604
</td>
<td>
159.435880
</td>
<td>
1.032528
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
scale_p0_beta_value
</th>
<th>
(0,)
</th>
<td>
kernel_01
</td>
<td>
-3.065548
</td>
<td>
0.063301
</td>
<td>
-3.166213
</td>
<td>
-3.067286
</td>
<td>
-2.955889
</td>
<td>
4000
</td>
<td>
74.441443
</td>
<td>
123.947335
</td>
<td>
1.052915
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_01
</td>
<td>
1.042774
</td>
<td>
0.080379
</td>
<td>
0.908366
</td>
<td>
1.044098
</td>
<td>
1.175719
</td>
<td>
4000
</td>
<td>
104.251728
</td>
<td>
230.752675
</td>
<td>
1.030835
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
90
</th>
<th rowspan="2" valign="top">
nan acceptance prob
</th>
<th>
warmup
</th>
<td>
69
</td>
<td>
0.01725
</td>
</tr>
<tr>
<th>
posterior
</th>
<td>
4
</td>
<td>
0.00100
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
kernel_01
</th>
<th rowspan="2" valign="top">
90
</th>
<th rowspan="2" valign="top">
nan acceptance prob
</th>
<th>
warmup
</th>
<td>
22
</td>
<td>
0.00550
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
0.00000
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
kernel_03
</th>
<th rowspan="2" valign="top">
90
</th>
<th rowspan="2" valign="top">
nan acceptance prob
</th>
<th>
warmup
</th>
<td>
33
</td>
<td>
0.00825
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
0.00000
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
kernel_04
</th>
<th rowspan="2" valign="top">
90
</th>
<th rowspan="2" valign="top">
nan acceptance prob
</th>
<th>
warmup
</th>
<td>
32
</td>
<td>
0.00800
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
0.00000
</td>
</tr>
</tbody>
</table>

And the corresponding trace plots:

``` python
fig = gs.plot_trace(results, "loc_p0_beta_value")
```

![](03-gev_files/figure-commonmark/traces-1.png)

``` python
fig = gs.plot_trace(results, "loc_np0_tau2_value")
```

![](03-gev_files/figure-commonmark/traces-2.png)

``` python
fig = gs.plot_trace(results, "loc_np0_beta_value")
```

![](03-gev_files/figure-commonmark/traces-3.png)

``` python
fig = gs.plot_trace(results, "scale_p0_beta_value")
```

![](03-gev_files/figure-commonmark/traces-4.png)

``` python
fig = gs.plot_trace(results, "concentration_p0_beta_value")
```

![](03-gev_files/figure-commonmark/traces-5.png)

We need to reset the index of the summary data frame before we can
transfer it to R.

``` python
summary = gs.Summary(results).to_dataframe().reset_index()
```

After transferring the summary data frame to R, we can process it with
packages like dplyr and ggplot2. Here is a visualization of the
estimated spline vs. the true function:

``` r
library(dplyr)
```


    Attaching package: 'dplyr'

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

``` r
library(ggplot2)
library(reticulate)

summary <- py$summary

beta <- summary %>%
  filter(variable == "loc_np0_beta_value") %>%
  group_by(var_index) %>%
  summarize(mean = mean(mean)) %>%
  ungroup()

beta <- beta$mean
X <- py_to_r(model$vars["loc_np0_X"]$value)
estimate <- X %*% beta

true <- sin(2 * pi * x0)

ggplot(data.frame(x0 = x0, estimate = estimate, true = true)) +
  geom_line(aes(x0, estimate), color = palette()[2]) +
  geom_line(aes(x0, true), color = palette()[4]) +
  ggtitle("Estimated spline (red) vs. true function (blue)") +
  ylab("f") +
  theme_minimal()
```

![](03-gev_files/figure-commonmark/spline-11.png)
