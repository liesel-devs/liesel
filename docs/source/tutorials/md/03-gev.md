
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

use_liesel_venv()
```

    [1] "liesel"

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

    Warning in poetry_config(required_module): This project appears to use Poetry for Python dependency management.
    However, the 'poetry' command line tool is not available.
    reticulate will be unable to activate this project.
    Please ensure that 'poetry' is available on the PATH.

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
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 0, 1, 0 / 75 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 5, 8, 8, 6 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 1, 3, 2 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 1, 1, 1 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 0, 1, 0 / 25 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 3, 1, 1, 1 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 2, 3 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 2, 1, 1, 0 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 0, 1, 2, 3 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 0, 1, 0 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 1, 1, 2 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 2 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 1, 1, 1 / 100 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 1, 1, 1 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 2, 2, 1 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 0 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 0, 0, 1 / 200 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 4, 1, 1, 2 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 500 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 3, 3, 0 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 2 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 0, 1, 1, 1 / 500 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 3, 2, 2, 1 / 500 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 2, 0, 0 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 2, 1, 1 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 0, 0, 0 / 50 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 1, 1, 3 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 0, 3, 0, 10 / 1000 transitions
    liesel.goose.engine - INFO - Finished epoch

Some tabular summary statistics of the posterior samples:

``` python
import liesel.goose as gs

results = engine.get_results()
gs.Summary(results)
```

**Parameter summary:**

                                          kernel      mean  ...     ess_tail      rhat
    parameter                   index                       ...                       
    concentration_p0_beta_value (0,)   kernel_04  0.071163  ...   621.257847  1.005525
                                (1,)   kernel_04  1.063034  ...   327.458905  1.018020
    loc_np0_beta_value          (0,)   kernel_02 -0.628609  ...   244.957340  1.020311
                                (1,)   kernel_02  0.294310  ...   234.118677  1.022517
                                (2,)   kernel_02 -0.381878  ...   391.440468  1.014608
                                (3,)   kernel_02  0.357636  ...   154.609905  1.055907
                                (4,)   kernel_02 -0.264448  ...   196.603094  1.073938
                                (5,)   kernel_02  0.183906  ...   150.374472  1.036915
                                (6,)   kernel_02  6.028223  ...   237.016711  1.058829
                                (7,)   kernel_02  0.514359  ...   134.607186  1.037403
                                (8,)   kernel_02  1.700216  ...   233.830268  1.053565
    loc_np0_tau2_value          ()     kernel_01  6.363313  ...  3887.143262  0.999758
    loc_p0_beta_value           (0,)   kernel_03  0.026949  ...   225.241539  1.055979
    scale_p0_beta_value         (0,)   kernel_00 -3.076459  ...   184.429870  1.043563
                                (1,)   kernel_00  1.055013  ...   291.023287  1.020358

    [15 rows x 10 columns]

**Error summary:**

                                                       count  relative
    kernel    error_code error_msg           phase                    
    kernel_00 90         nan acceptance prob warmup       40   0.01000
                                             posterior     0   0.00000
    kernel_02 90         nan acceptance prob warmup       26   0.00650
                                             posterior     0   0.00000
    kernel_03 90         nan acceptance prob warmup       18   0.00450
                                             posterior     0   0.00000
    kernel_04 90         nan acceptance prob warmup       61   0.01525
                                             posterior    13   0.00325

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
