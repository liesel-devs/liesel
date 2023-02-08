
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
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 0, 0 / 75 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 0, 0, 0, 1 / 75 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 5, 7, 8, 6 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 1, 1, 1 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 1, 1, 1 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 0, 1, 0 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 0, 1, 2, 0 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 3, 2 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 0 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 0, 1, 3, 1 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 1, 3, 2 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 0, 1, 1 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 1 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 2, 2, 0 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 1, 1, 3 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 0, 1 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 0 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 2, 1, 3, 2 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 2, 2, 1, 2 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 500 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 0, 1, 3, 0 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 2, 1 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 2, 2, 2, 2 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 4, 2, 2, 3 / 500 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 2, 1, 0 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 3, 1, 1 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 0, 0, 0 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 2, 1, 2, 2 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 0, 1, 2, 1 / 1000 transitions
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
kernel_04
</td>
<td>
0.070919
</td>
<td>
0.050065
</td>
<td>
-0.007002
</td>
<td>
0.068140
</td>
<td>
0.159025
</td>
<td>
4000
</td>
<td>
314.319488
</td>
<td>
730.590480
</td>
<td>
1.004567
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_04
</td>
<td>
1.066979
</td>
<td>
0.101746
</td>
<td>
0.896463
</td>
<td>
1.070067
</td>
<td>
1.231077
</td>
<td>
4000
</td>
<td>
156.182368
</td>
<td>
408.119717
</td>
<td>
1.014958
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
kernel_02
</td>
<td>
-0.619810
</td>
<td>
0.209004
</td>
<td>
-0.977139
</td>
<td>
-0.616525
</td>
<td>
-0.277129
</td>
<td>
4000
</td>
<td>
146.245158
</td>
<td>
236.632085
</td>
<td>
1.023291
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_02
</td>
<td>
0.293308
</td>
<td>
0.130491
</td>
<td>
0.080198
</td>
<td>
0.294015
</td>
<td>
0.501205
</td>
<td>
4000
</td>
<td>
112.816758
</td>
<td>
286.924758
</td>
<td>
1.025576
</td>
</tr>
<tr>
<th>
(2,)
</th>
<td>
kernel_02
</td>
<td>
-0.381735
</td>
<td>
0.113291
</td>
<td>
-0.561648
</td>
<td>
-0.383904
</td>
<td>
-0.194060
</td>
<td>
4000
</td>
<td>
205.090264
</td>
<td>
403.509320
</td>
<td>
1.007925
</td>
</tr>
<tr>
<th>
(3,)
</th>
<td>
kernel_02
</td>
<td>
0.356764
</td>
<td>
0.069085
</td>
<td>
0.243040
</td>
<td>
0.359056
</td>
<td>
0.473349
</td>
<td>
4000
</td>
<td>
76.970662
</td>
<td>
130.293533
</td>
<td>
1.050146
</td>
</tr>
<tr>
<th>
(4,)
</th>
<td>
kernel_02
</td>
<td>
-0.261368
</td>
<td>
0.081368
</td>
<td>
-0.400211
</td>
<td>
-0.256852
</td>
<td>
-0.132445
</td>
<td>
4000
</td>
<td>
63.506089
</td>
<td>
203.286637
</td>
<td>
1.069075
</td>
</tr>
<tr>
<th>
(5,)
</th>
<td>
kernel_02
</td>
<td>
0.182809
</td>
<td>
0.030587
</td>
<td>
0.134047
</td>
<td>
0.181415
</td>
<td>
0.235130
</td>
<td>
4000
</td>
<td>
98.030581
</td>
<td>
170.356336
</td>
<td>
1.034912
</td>
</tr>
<tr>
<th>
(6,)
</th>
<td>
kernel_02
</td>
<td>
6.029377
</td>
<td>
0.041879
</td>
<td>
5.961965
</td>
<td>
6.028557
</td>
<td>
6.098739
</td>
<td>
4000
</td>
<td>
72.755403
</td>
<td>
251.277811
</td>
<td>
1.056982
</td>
</tr>
<tr>
<th>
(7,)
</th>
<td>
kernel_02
</td>
<td>
0.516094
</td>
<td>
0.066582
</td>
<td>
0.397667
</td>
<td>
0.520247
</td>
<td>
0.623368
</td>
<td>
4000
</td>
<td>
105.102081
</td>
<td>
168.142696
</td>
<td>
1.037750
</td>
</tr>
<tr>
<th>
(8,)
</th>
<td>
kernel_02
</td>
<td>
1.701461
</td>
<td>
0.032325
</td>
<td>
1.650673
</td>
<td>
1.701423
</td>
<td>
1.755116
</td>
<td>
4000
</td>
<td>
73.178291
</td>
<td>
248.892315
</td>
<td>
1.053755
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
kernel_01
</td>
<td>
6.363233
</td>
<td>
4.313678
</td>
<td>
2.459297
</td>
<td>
5.147676
</td>
<td>
14.093001
</td>
<td>
4000
</td>
<td>
3626.276097
</td>
<td>
3887.143262
</td>
<td>
0.999741
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
kernel_03
</td>
<td>
0.026860
</td>
<td>
0.002540
</td>
<td>
0.022690
</td>
<td>
0.026908
</td>
<td>
0.030963
</td>
<td>
4000
</td>
<td>
134.189000
</td>
<td>
338.226615
</td>
<td>
1.041553
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
kernel_00
</td>
<td>
-3.077337
</td>
<td>
0.061324
</td>
<td>
-3.183016
</td>
<td>
-3.074161
</td>
<td>
-2.976252
</td>
<td>
4000
</td>
<td>
139.327922
</td>
<td>
231.278273
</td>
<td>
1.033659
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
1.056422
</td>
<td>
0.075294
</td>
<td>
0.934316
</td>
<td>
1.055524
</td>
<td>
1.181464
</td>
<td>
4000
</td>
<td>
182.070200
</td>
<td>
342.034422
</td>
<td>
1.019498
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
29
</td>
<td>
0.00725
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
kernel_02
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
24
</td>
<td>
0.00600
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
30
</td>
<td>
0.00750
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
67
</td>
<td>
0.01675
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
