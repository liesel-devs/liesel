
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

    Please make sure you are using a virtual or conda environment with Liesel installed, e.g. using `reticulate::use_virtualenv()` or `reticulate::use_condaenv()`. See `vignette("versions", "reticulate")`.

    After setting the environment, check if the installed versions of RLiesel and Liesel are compatible with `check_liesel_version()`.

``` r
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

    Installed Liesel version 0.2.10-dev is compatible, continuing to set up model

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
0.102
</td>
<td>
0.053
</td>
<td>
0.017
</td>
<td>
0.101
</td>
<td>
0.193
</td>
<td>
4000
</td>
<td>
371.025
</td>
<td>
840.414
</td>
<td>
1.015
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
0.972
</td>
<td>
0.099
</td>
<td>
0.811
</td>
<td>
0.975
</td>
<td>
1.132
</td>
<td>
4000
</td>
<td>
161.693
</td>
<td>
449.305
</td>
<td>
1.038
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
0.486
</td>
<td>
0.214
</td>
<td>
0.140
</td>
<td>
0.492
</td>
<td>
0.844
</td>
<td>
4000
</td>
<td>
81.248
</td>
<td>
246.038
</td>
<td>
1.037
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
-0.168
</td>
<td>
0.106
</td>
<td>
-0.343
</td>
<td>
-0.168
</td>
<td>
0.003
</td>
<td>
4000
</td>
<td>
125.877
</td>
<td>
351.505
</td>
<td>
1.028
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
-0.462
</td>
<td>
0.143
</td>
<td>
-0.684
</td>
<td>
-0.469
</td>
<td>
-0.211
</td>
<td>
4000
</td>
<td>
107.938
</td>
<td>
201.522
</td>
<td>
1.019
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
-0.022
</td>
<td>
0.068
</td>
<td>
-0.132
</td>
<td>
-0.024
</td>
<td>
0.100
</td>
<td>
4000
</td>
<td>
131.218
</td>
<td>
231.624
</td>
<td>
1.026
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
0.477
</td>
<td>
0.062
</td>
<td>
0.374
</td>
<td>
0.478
</td>
<td>
0.575
</td>
<td>
4000
</td>
<td>
118.509
</td>
<td>
253.829
</td>
<td>
1.022
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
0.457
</td>
<td>
0.027
</td>
<td>
0.410
</td>
<td>
0.456
</td>
<td>
0.500
</td>
<td>
4000
</td>
<td>
122.648
</td>
<td>
230.152
</td>
<td>
1.016
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
-5.906
</td>
<td>
0.031
</td>
<td>
-5.956
</td>
<td>
-5.905
</td>
<td>
-5.856
</td>
<td>
4000
</td>
<td>
105.971
</td>
<td>
257.752
</td>
<td>
1.031
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
-0.384
</td>
<td>
0.062
</td>
<td>
-0.482
</td>
<td>
-0.384
</td>
<td>
-0.284
</td>
<td>
4000
</td>
<td>
149.321
</td>
<td>
292.330
</td>
<td>
1.005
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
-1.788
</td>
<td>
0.026
</td>
<td>
-1.831
</td>
<td>
-1.788
</td>
<td>
-1.746
</td>
<td>
4000
</td>
<td>
103.978
</td>
<td>
245.266
</td>
<td>
1.027
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
6.048
</td>
<td>
4.345
</td>
<td>
2.356
</td>
<td>
4.932
</td>
<td>
13.018
</td>
<td>
4000
</td>
<td>
3980.546
</td>
<td>
3781.071
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
0.003
</td>
<td>
-0.031
</td>
<td>
-0.027
</td>
<td>
-0.022
</td>
<td>
4000
</td>
<td>
21.449
</td>
<td>
188.444
</td>
<td>
1.157
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
-3.100
</td>
<td>
0.061
</td>
<td>
-3.200
</td>
<td>
-3.100
</td>
<td>
-2.997
</td>
<td>
4000
</td>
<td>
42.713
</td>
<td>
245.213
</td>
<td>
1.107
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
1.201
</td>
<td>
0.080
</td>
<td>
1.070
</td>
<td>
1.201
</td>
<td>
1.335
</td>
<td>
4000
</td>
<td>
158.839
</td>
<td>
438.542
</td>
<td>
1.045
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
88
</td>
<td>
0.022
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
45
</td>
<td>
0.011
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
27
</td>
<td>
0.007
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
22
</td>
<td>
0.005
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
