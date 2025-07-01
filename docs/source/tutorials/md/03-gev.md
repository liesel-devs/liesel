

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
     33%|##############                            | 1/3 [00:12<00:25, 12.59s/chunk]
    100%|##########################################| 3/3 [00:12<00:00,  4.20s/chunk]

      0%|                                                  | 0/1 [00:00<?, ?chunk/s]
    100%|#########################################| 1/1 [00:00<00:00, 864.09chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|#########################################| 2/2 [00:00<00:00, 998.88chunk/s]

      0%|                                                  | 0/4 [00:00<?, ?chunk/s]
    100%|#########################################| 4/4 [00:00<00:00, 901.52chunk/s]

      0%|                                                  | 0/8 [00:00<?, ?chunk/s]
    100%|#########################################| 8/8 [00:00<00:00, 238.42chunk/s]

      0%|                                                 | 0/20 [00:00<?, ?chunk/s]
     55%|######################                  | 11/20 [00:00<00:00, 86.11chunk/s]
    100%|########################################| 20/20 [00:00<00:00, 43.99chunk/s]
    100%|########################################| 20/20 [00:00<00:00, 47.83chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|#########################################| 2/2 [00:00<00:00, 892.22chunk/s]

      0%|                                                 | 0/40 [00:00<?, ?chunk/s]
     28%|###########                             | 11/40 [00:00<00:00, 85.61chunk/s]
     50%|####################                    | 20/40 [00:00<00:00, 44.96chunk/s]
     65%|##########################              | 26/40 [00:00<00:00, 39.55chunk/s]
     78%|###############################         | 31/40 [00:00<00:00, 37.04chunk/s]
     88%|###################################     | 35/40 [00:00<00:00, 35.55chunk/s]
     98%|####################################### | 39/40 [00:01<00:00, 34.54chunk/s]
    100%|########################################| 40/40 [00:01<00:00, 38.60chunk/s]

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

0.103
</td>

<td>

0.054
</td>

<td>

0.017
</td>

<td>

0.102
</td>

<td>

0.192
</td>

<td>

4000
</td>

<td>

350.585
</td>

<td>

960.411
</td>

<td>

1.003
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

0.965
</td>

<td>

0.099
</td>

<td>

0.796
</td>

<td>

0.968
</td>

<td>

1.125
</td>

<td>

4000
</td>

<td>

211.096
</td>

<td>

649.547
</td>

<td>

1.012
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

0.458
</td>

<td>

0.214
</td>

<td>

0.076
</td>

<td>

0.463
</td>

<td>

0.804
</td>

<td>

4000
</td>

<td>

37.229
</td>

<td>

150.088
</td>

<td>

1.088
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

-0.360
</td>

<td>

-0.150
</td>

<td>

0.065
</td>

<td>

4000
</td>

<td>

47.046
</td>

<td>

111.479
</td>

<td>

1.090
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

0.470
</td>

<td>

0.138
</td>

<td>

0.240
</td>

<td>

0.472
</td>

<td>

0.691
</td>

<td>

4000
</td>

<td>

82.064
</td>

<td>

123.233
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

-0.010
</td>

<td>

0.073
</td>

<td>

-0.133
</td>

<td>

-0.007
</td>

<td>

0.110
</td>

<td>

4000
</td>

<td>

45.732
</td>

<td>

121.734
</td>

<td>

1.103
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

0.071
</td>

<td>

0.359
</td>

<td>

0.470
</td>

<td>

0.590
</td>

<td>

4000
</td>

<td>

49.928
</td>

<td>

107.821
</td>

<td>

1.087
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

0.031
</td>

<td>

0.410
</td>

<td>

0.456
</td>

<td>

0.510
</td>

<td>

4000
</td>

<td>

81.156
</td>

<td>

130.776
</td>

<td>

1.025
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

-5.912
</td>

<td>

0.031
</td>

<td>

-5.965
</td>

<td>

-5.913
</td>

<td>

-5.863
</td>

<td>

4000
</td>

<td>

59.453
</td>

<td>

144.919
</td>

<td>

1.078
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

0.378
</td>

<td>

0.069
</td>

<td>

0.254
</td>

<td>

0.380
</td>

<td>

0.487
</td>

<td>

4000
</td>

<td>

83.115
</td>

<td>

180.751
</td>

<td>

1.042
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

-1.795
</td>

<td>

0.026
</td>

<td>

-1.838
</td>

<td>

-1.795
</td>

<td>

-1.753
</td>

<td>

4000
</td>

<td>

80.385
</td>

<td>

146.879
</td>

<td>

1.071
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

4.375
</td>

<td>

2.277
</td>

<td>

4.944
</td>

<td>

12.875
</td>

<td>

4000
</td>

<td>

3612.737
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

94.393
</td>

<td>

456.776
</td>

<td>

1.051
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

-3.092
</td>

<td>

0.059
</td>

<td>

-3.189
</td>

<td>

-3.089
</td>

<td>

-2.998
</td>

<td>

4000
</td>

<td>

146.759
</td>

<td>

374.322
</td>

<td>

1.063
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

1.196
</td>

<td>

0.082
</td>

<td>

1.066
</td>

<td>

1.194
</td>

<td>

1.332
</td>

<td>

4000
</td>

<td>

87.618
</td>

<td>

509.374
</td>

<td>

1.049
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

70
</td>

<td>

0.018
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

33
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
