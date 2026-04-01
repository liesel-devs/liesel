

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

    Response 'y' found in environment, but not data, using environment.

Now, we can continue in Python and use the `lsl.dist_reg_mcmc()`
function to set up a sampling algorithm with IWLS kernels for the
regression coefficients ($\boldsymbol{\beta}$) and a Gibbs kernel for
the smoothing parameter ($\tau^2$) of the spline.

The support of the GEV distribution changes with the parameter values
(compare
[Wikipedia](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution)).
To ensure that the initial parameters support the observed data we set
$xi = 0.1$ and disable jittering of the variance and regression
parameters.

``` python
import liesel.model as lsl
import jax.numpy as jnp

model = r.model

# concentration == 0.0 seems to break the sampler
model.vars["concentration_p0_beta"].value = jnp.array([0.1, 0.0])

builder = lsl.dist_reg_mcmc(model, seed=42, num_chains=4, apply_jitter=False)
builder.set_duration(warmup_duration=1000, posterior_duration=1000)

engine = builder.build()
engine.sample_all_epochs()
```


      0%|                                                  | 0/3 [00:00<?, ?chunk/s]
     33%|##############                            | 1/3 [00:13<00:26, 13.19s/chunk]
    100%|##########################################| 3/3 [00:13<00:00,  4.40s/chunk]

      0%|                                                  | 0/1 [00:00<?, ?chunk/s]
    100%|#########################################| 1/1 [00:00<00:00, 691.33chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|#########################################| 2/2 [00:00<00:00, 754.85chunk/s]

      0%|                                                  | 0/4 [00:00<?, ?chunk/s]
    100%|#########################################| 4/4 [00:00<00:00, 792.84chunk/s]

      0%|                                                  | 0/8 [00:00<?, ?chunk/s]
    100%|#########################################| 8/8 [00:00<00:00, 248.59chunk/s]

      0%|                                                 | 0/20 [00:00<?, ?chunk/s]
     55%|######################                  | 11/20 [00:00<00:00, 93.99chunk/s]
    100%|########################################| 20/20 [00:00<00:00, 53.89chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|#########################################| 2/2 [00:00<00:00, 717.16chunk/s]

      0%|                                                 | 0/40 [00:00<?, ?chunk/s]
     28%|###########                             | 11/40 [00:00<00:00, 92.27chunk/s]
     52%|#####################                   | 21/40 [00:00<00:00, 48.04chunk/s]
     68%|###########################             | 27/40 [00:00<00:00, 42.91chunk/s]
     80%|################################        | 32/40 [00:00<00:00, 40.44chunk/s]
     92%|#####################################   | 37/40 [00:00<00:00, 38.79chunk/s]
    100%|########################################| 40/40 [00:00<00:00, 42.39chunk/s]

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

0.054
</td>

<td>

0.016
</td>

<td>

0.102
</td>

<td>

0.193
</td>

<td>

4000
</td>

<td>

310.636
</td>

<td>

609.038
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

0.966
</td>

<td>

0.098
</td>

<td>

0.800
</td>

<td>

0.969
</td>

<td>

1.124
</td>

<td>

4000
</td>

<td>

190.571
</td>

<td>

606.042
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

0.468
</td>

<td>

0.208
</td>

<td>

0.106
</td>

<td>

0.472
</td>

<td>

0.805
</td>

<td>

4000
</td>

<td>

40.894
</td>

<td>

134.991
</td>

<td>

1.084
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

-0.148
</td>

<td>

0.132
</td>

<td>

-0.367
</td>

<td>

-0.148
</td>

<td>

0.065
</td>

<td>

4000
</td>

<td>

50.483
</td>

<td>

120.105
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

0.476
</td>

<td>

0.140
</td>

<td>

0.245
</td>

<td>

0.477
</td>

<td>

0.698
</td>

<td>

4000
</td>

<td>

84.706
</td>

<td>

124.158
</td>

<td>

1.031
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

-0.007
</td>

<td>

0.074
</td>

<td>

-0.130
</td>

<td>

-0.005
</td>

<td>

0.114
</td>

<td>

4000
</td>

<td>

47.191
</td>

<td>

138.705
</td>

<td>

1.099
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

0.471
</td>

<td>

0.070
</td>

<td>

0.360
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

56.723
</td>

<td>

130.083
</td>

<td>

1.079
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

0.510
</td>

<td>

4000
</td>

<td>

83.115
</td>

<td>

128.557
</td>

<td>

1.021
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

-5.962
</td>

<td>

-5.912
</td>

<td>

-5.862
</td>

<td>

4000
</td>

<td>

61.691
</td>

<td>

192.405
</td>

<td>

1.076
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

0.374
</td>

<td>

0.068
</td>

<td>

0.253
</td>

<td>

0.376
</td>

<td>

0.481
</td>

<td>

4000
</td>

<td>

83.433
</td>

<td>

167.784
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

0.025
</td>

<td>

-1.836
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

79.933
</td>

<td>

177.089
</td>

<td>

1.065
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

5.966
</td>

<td>

4.374
</td>

<td>

2.278
</td>

<td>

4.940
</td>

<td>

12.879
</td>

<td>

4000
</td>

<td>

3614.251
</td>

<td>

3818.006
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

-0.030
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

106.968
</td>

<td>

412.177
</td>

<td>

1.046
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

-3.191
</td>

<td>

-3.088
</td>

<td>

-3.000
</td>

<td>

4000
</td>

<td>

166.721
</td>

<td>

406.233
</td>

<td>

1.049
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

1.198
</td>

<td>

0.081
</td>

<td>

1.069
</td>

<td>

1.195
</td>

<td>

1.331
</td>

<td>

4000
</td>

<td>

137.838
</td>

<td>

419.120
</td>

<td>

1.044
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

<th rowspan="6" valign="top">

kernel_00
</th>

<th rowspan="2" valign="top">

2
</th>

<th rowspan="2" valign="top">

indefinite information matrix (fallback to identity)
</th>

<th>

warmup
</th>

<td>

35
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.009
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

<tr>

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

49
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.012
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

<tr>

<th rowspan="2" valign="top">

92
</th>

<th rowspan="2" valign="top">

indefinite information matrix (fallback to identity) + nan acceptance
prob
</th>

<th>

warmup
</th>

<td>

8
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.002
</td>

</tr>

<tr>

<th>

posterior
</th>

<td>

3
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.001
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

50
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.013
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

<tr>

<th rowspan="6" valign="top">

kernel_03
</th>

<th rowspan="2" valign="top">

2
</th>

<th rowspan="2" valign="top">

indefinite information matrix (fallback to identity)
</th>

<th>

warmup
</th>

<td>

315
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.079
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

<tr>

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

4000
</td>

<td>

4000
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

4000
</td>

<td>

4000
</td>

<td>

0.000
</td>

</tr>

<tr>

<th rowspan="2" valign="top">

92
</th>

<th rowspan="2" valign="top">

indefinite information matrix (fallback to identity) + nan acceptance
prob
</th>

<th>

warmup
</th>

<td>

10
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.002
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

21
</td>

<td>

4000
</td>

<td>

4000
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
