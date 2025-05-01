
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
library(VGAM)
```

    Loading required package: stats4

    Loading required package: splines

``` r
set.seed(13)

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

    Did not find response 'y' in data. Using 'y' found in parent environment.

Now, we can continue in Python and use the `lsl.dist_reg_mcmc()`
function to set up a sampling algorithm with IWLS kernels for the
regression coefficients ($\boldsymbol{\beta}$) and a Gibbs kernel for
the smoothing parameter ($\tau^2$) of the spline.

The support of the GEV distribution changes with the parameter values
(compare
[Wikipedia](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution)).
To ensure that the initial parameters support the observed data we set
$xi = 0.1$ and disable jittering of the the variance and regression
parameters. For the latter, we supply user-defined jitter functions to
`lsl.dist_reg_mcmc` that are essentially the identity function w.r.t.
the parameter value.

``` python
import liesel.model as lsl
import jax.numpy as jnp

model = r.model

# concentration == 0.0 seems to break the sampler
model.vars["concentration_p0_beta"].value = jnp.array([0.1, 0.0])

builder = lsl.dist_reg_mcmc(model, seed=42, num_chains=4, tau2_jitter_fn=lambda key, val: val, beta_jitter_fn=lambda key, val: val)
builder.set_duration(warmup_duration=1000, posterior_duration=1000)

engine = builder.build()
engine.sample_all_epochs()
```


      0%|                                                  | 0/3 [00:00<?, ?chunk/s]
     33%|##############                            | 1/3 [00:04<00:09,  4.81s/chunk]
    100%|##########################################| 3/3 [00:04<00:00,  1.60s/chunk]

      0%|                                                  | 0/1 [00:00<?, ?chunk/s]
    100%|########################################| 1/1 [00:00<00:00, 1828.38chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|########################################| 2/2 [00:00<00:00, 1919.15chunk/s]

      0%|                                                  | 0/4 [00:00<?, ?chunk/s]
    100%|########################################| 4/4 [00:00<00:00, 2291.97chunk/s]

      0%|                                                  | 0/8 [00:00<?, ?chunk/s]
    100%|#########################################| 8/8 [00:00<00:00, 396.61chunk/s]

      0%|                                                 | 0/20 [00:00<?, ?chunk/s]
     65%|#########################3             | 13/20 [00:00<00:00, 114.97chunk/s]
    100%|########################################| 20/20 [00:00<00:00, 82.45chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|########################################| 2/2 [00:00<00:00, 2123.70chunk/s]

      0%|                                                 | 0/40 [00:00<?, ?chunk/s]
     32%|############6                          | 13/40 [00:00<00:00, 114.72chunk/s]
     62%|#########################               | 25/40 [00:00<00:00, 69.94chunk/s]
     82%|#################################       | 33/40 [00:00<00:00, 52.50chunk/s]
    100%|########################################| 40/40 [00:00<00:00, 64.56chunk/s]

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
concentration_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_00
</td>
<td>
0.104
</td>
<td>
0.054
</td>
<td>
0.016
</td>
<td>
0.103
</td>
<td>
0.193
</td>
<td>
4000
</td>
<td>
372.271
</td>
<td>
988.761
</td>
<td>
1.004
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
0.964
</td>
<td>
0.099
</td>
<td>
0.796
</td>
<td>
0.967
</td>
<td>
1.121
</td>
<td>
4000
</td>
<td>
207.805
</td>
<td>
645.642
</td>
<td>
1.010
</td>
</tr>
<tr>
<th rowspan="9" valign="top">
loc_np0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_03
</td>
<td>
0.469
</td>
<td>
0.207
</td>
<td>
0.121
</td>
<td>
0.469
</td>
<td>
0.807
</td>
<td>
4000
</td>
<td>
54.068
</td>
<td>
156.325
</td>
<td>
1.067
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
-0.147
</td>
<td>
0.129
</td>
<td>
-0.358
</td>
<td>
-0.149
</td>
<td>
0.067
</td>
<td>
4000
</td>
<td>
51.923
</td>
<td>
105.754
</td>
<td>
1.081
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
0.473
</td>
<td>
0.139
</td>
<td>
0.241
</td>
<td>
0.472
</td>
<td>
0.696
</td>
<td>
4000
</td>
<td>
85.108
</td>
<td>
129.521
</td>
<td>
1.037
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
-0.008
</td>
<td>
0.073
</td>
<td>
-0.132
</td>
<td>
-0.005
</td>
<td>
0.113
</td>
<td>
4000
</td>
<td>
61.796
</td>
<td>
168.336
</td>
<td>
1.093
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
0.472
</td>
<td>
0.070
</td>
<td>
0.362
</td>
<td>
0.470
</td>
<td>
0.589
</td>
<td>
4000
</td>
<td>
64.460
</td>
<td>
135.078
</td>
<td>
1.074
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
0.458
</td>
<td>
0.031
</td>
<td>
0.412
</td>
<td>
0.457
</td>
<td>
0.512
</td>
<td>
4000
</td>
<td>
87.095
</td>
<td>
127.444
</td>
<td>
1.023
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
-5.911
</td>
<td>
0.031
</td>
<td>
-5.964
</td>
<td>
-5.913
</td>
<td>
-5.862
</td>
<td>
4000
</td>
<td>
75.689
</td>
<td>
136.909
</td>
<td>
1.069
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
0.375
</td>
<td>
0.069
</td>
<td>
0.253
</td>
<td>
0.375
</td>
<td>
0.488
</td>
<td>
4000
</td>
<td>
87.037
</td>
<td>
169.840
</td>
<td>
1.040
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
-1.794
</td>
<td>
0.026
</td>
<td>
-1.837
</td>
<td>
-1.794
</td>
<td>
-1.753
</td>
<td>
4000
</td>
<td>
87.187
</td>
<td>
160.277
</td>
<td>
1.059
</td>
</tr>
<tr>
<th>
loc_np0_tau2
</th>
<th>
()
</th>
<td>
kernel_02
</td>
<td>
5.967
</td>
<td>
4.374
</td>
<td>
2.276
</td>
<td>
4.946
</td>
<td>
12.862
</td>
<td>
4000
</td>
<td>
3610.352
</td>
<td>
3848.888
</td>
<td>
1.001
</td>
</tr>
<tr>
<th>
loc_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_04
</td>
<td>
-0.027
</td>
<td>
0.002
</td>
<td>
-0.031
</td>
<td>
-0.027
</td>
<td>
-0.023
</td>
<td>
4000
</td>
<td>
90.065
</td>
<td>
390.131
</td>
<td>
1.050
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
scale_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_01
</td>
<td>
-3.093
</td>
<td>
0.059
</td>
<td>
-3.190
</td>
<td>
-3.090
</td>
<td>
-2.999
</td>
<td>
4000
</td>
<td>
151.457
</td>
<td>
348.065
</td>
<td>
1.057
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
1.197
</td>
<td>
0.081
</td>
<td>
1.067
</td>
<td>
1.196
</td>
<td>
1.332
</td>
<td>
4000
</td>
<td>
246.643
</td>
<td>
530.955
</td>
<td>
1.038
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
0.017
</td>
</tr>
<tr>
<th>
posterior
</th>
<td>
1
</td>
<td>
0.000
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
32
</td>
<td>
0.008
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
0.000
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
25
</td>
<td>
0.006
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
0.000
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
23
</td>
<td>
0.006
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
0.000
</td>
</tr>
</tbody>
</table>

And the corresponding trace plots:

``` python
fig = gs.plot_trace(results, "loc_p0_beta")
```

![](03-gev_files/figure-commonmark/traces-1.png)

``` python
fig = gs.plot_trace(results, "loc_np0_tau2")
```

![](03-gev_files/figure-commonmark/traces-2.png)

``` python
fig = gs.plot_trace(results, "loc_np0_beta")
```

![](03-gev_files/figure-commonmark/traces-3.png)

``` python
fig = gs.plot_trace(results, "scale_p0_beta")
```

![](03-gev_files/figure-commonmark/traces-4.png)

``` python
fig = gs.plot_trace(results, "concentration_p0_beta")
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
```

    Warning: package 'reticulate' was built under R version 4.4.1

``` r
summary <- py$summary

beta <- summary %>%
  filter(variable == "loc_np0_beta") %>%
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
