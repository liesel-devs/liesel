

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
     33%|##############                            | 1/3 [00:10<00:20, 10.44s/chunk]
    100%|##########################################| 3/3 [00:10<00:00,  3.48s/chunk]

      0%|                                                  | 0/1 [00:00<?, ?chunk/s]
    100%|#########################################| 1/1 [00:00<00:00, 865.52chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|#########################################| 2/2 [00:00<00:00, 998.52chunk/s]

      0%|                                                  | 0/4 [00:00<?, ?chunk/s]
    100%|########################################| 4/4 [00:00<00:00, 1081.01chunk/s]

      0%|                                                  | 0/8 [00:00<?, ?chunk/s]
    100%|#########################################| 8/8 [00:00<00:00, 233.79chunk/s]

      0%|                                                 | 0/20 [00:00<?, ?chunk/s]
     55%|######################                  | 11/20 [00:00<00:00, 85.09chunk/s]
    100%|########################################| 20/20 [00:00<00:00, 44.53chunk/s]
    100%|########################################| 20/20 [00:00<00:00, 48.31chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|########################################| 2/2 [00:00<00:00, 1023.75chunk/s]

      0%|                                                 | 0/40 [00:00<?, ?chunk/s]
     28%|###########                             | 11/40 [00:00<00:00, 84.91chunk/s]
     50%|####################                    | 20/40 [00:00<00:00, 44.48chunk/s]
     65%|##########################              | 26/40 [00:00<00:00, 39.08chunk/s]
     78%|###############################         | 31/40 [00:00<00:00, 36.60chunk/s]
     88%|###################################     | 35/40 [00:00<00:00, 35.22chunk/s]
     98%|####################################### | 39/40 [00:01<00:00, 34.24chunk/s]
    100%|########################################| 40/40 [00:01<00:00, 38.22chunk/s]

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

0.063
</td>

<td>

0.082
</td>

<td>

-0.053
</td>

<td>

0.077
</td>

<td>

0.185
</td>

<td>

4000
</td>

<td>

7.290
</td>

<td>

16.977
</td>

<td>

1.525
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

1.047
</td>

<td>

0.163
</td>

<td>

0.814
</td>

<td>

1.009
</td>

<td>

1.289
</td>

<td>

4000
</td>

<td>

7.310
</td>

<td>

520.799
</td>

<td>

1.524
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

0.461
</td>

<td>

0.200
</td>

<td>

0.113
</td>

<td>

0.467
</td>

<td>

0.774
</td>

<td>

4000
</td>

<td>

75.945
</td>

<td>

166.602
</td>

<td>

1.054
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

-0.149
</td>

<td>

0.125
</td>

<td>

-0.353
</td>

<td>

-0.141
</td>

<td>

0.042
</td>

<td>

4000
</td>

<td>

51.456
</td>

<td>

111.198
</td>

<td>

1.066
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

0.493
</td>

<td>

0.142
</td>

<td>

0.251
</td>

<td>

0.491
</td>

<td>

0.715
</td>

<td>

4000
</td>

<td>

68.266
</td>

<td>

96.098
</td>

<td>

1.038
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

0.001
</td>

<td>

0.075
</td>

<td>

-0.127
</td>

<td>

0.003
</td>

<td>

0.129
</td>

<td>

4000
</td>

<td>

21.950
</td>

<td>

75.684
</td>

<td>

1.173
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

0.479
</td>

<td>

0.069
</td>

<td>

0.364
</td>

<td>

0.478
</td>

<td>

0.598
</td>

<td>

4000
</td>

<td>

37.181
</td>

<td>

77.586
</td>

<td>

1.093
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

0.030
</td>

<td>

0.409
</td>

<td>

0.457
</td>

<td>

0.505
</td>

<td>

4000
</td>

<td>

78.472
</td>

<td>

139.800
</td>

<td>

1.031
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

-5.910
</td>

<td>

0.030
</td>

<td>

-5.961
</td>

<td>

-5.912
</td>

<td>

-5.861
</td>

<td>

4000
</td>

<td>

81.145
</td>

<td>

87.996
</td>

<td>

1.056
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

0.368
</td>

<td>

0.068
</td>

<td>

0.247
</td>

<td>

0.369
</td>

<td>

0.478
</td>

<td>

4000
</td>

<td>

50.969
</td>

<td>

162.685
</td>

<td>

1.077
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

-1.795
</td>

<td>

-1.753
</td>

<td>

4000
</td>

<td>

86.516
</td>

<td>

92.065
</td>

<td>

1.080
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

4.371
</td>

<td>

2.277
</td>

<td>

4.944
</td>

<td>

12.884
</td>

<td>

4000
</td>

<td>

3614.685
</td>

<td>

3832.700
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

55.744
</td>

<td>

356.177
</td>

<td>

1.068
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

0.057
</td>

<td>

-3.186
</td>

<td>

-3.090
</td>

<td>

-3.001
</td>

<td>

4000
</td>

<td>

167.940
</td>

<td>

431.583
</td>

<td>

1.045
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

1.211
</td>

<td>

0.083
</td>

<td>

1.075
</td>

<td>

1.211
</td>

<td>

1.346
</td>

<td>

4000
</td>

<td>

34.447
</td>

<td>

369.714
</td>

<td>

1.104
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

423
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.106
</td>

</tr>

<tr>

<th>

posterior
</th>

<td>

347
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.087
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

42
</td>

<td>

4000
</td>

<td>

4000
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

5
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

42
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.010
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

359
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.090
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

16
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.004
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
