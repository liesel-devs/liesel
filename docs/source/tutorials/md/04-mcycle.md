
# Comparing samplers

In this tutorial, we are comparing two different sampling schemes on the
`mcycle` dataset with a Gaussian location-scale regression model and two
splines for the mean and the standard deviation. The `mcycle` dataset is
a “data frame giving a series of measurements of head acceleration in a
simulated motorcycle accident, used to test crash helmets” (from the
help page). It contains the following two variables:

- `times`: in milliseconds after impact
- `accel`: in g

We start off in R by loading the dataset and setting up the model with
the `rliesel::liesel()` function.

``` r
library(MASS)
library(rliesel)
```

    Please set your Liesel venv, e.g. with use_liesel_venv()

``` r
data(mcycle)
with(mcycle, plot(times, accel))
```

![](04-mcycle_files/figure-commonmark/model-1.png)

``` r
model <- liesel(
  response = mcycle$accel,
  distribution = "Normal",
  predictors = list(
    loc = predictor(~ s(times)),
    scale = predictor(~ s(times), inverse_link = "Exp")
  ),
  data = mcycle
)
```

## Metropolis-in-Gibbs

First, we try a Metropolis-in-Gibbs sampling scheme with IWLS kernels
for the regression coefficients ($\boldsymbol{\beta}$) and Gibbs kernels
for the smoothing parameters ($\tau^2$) of the splines.

``` python
import liesel.model as lsl

model = r.model

builder = lsl.dist_reg_mcmc(model, seed=42, num_chains=4)
builder.set_duration(warmup_duration=5000, posterior_duration=1000)

engine = builder.build()
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 0, 0, 0, 1 / 75 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_05: 1, 1, 1, 1 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 1, 1, 0 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 0, 1, 0, 0 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 800 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 3300 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 0, 1, 0, 1 / 3300 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_04: 1, 0, 0, 1 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch

Clearly, the performance of the sampler could be better, especially for
the intercept of the mean. The corresponding chain exhibits a very
strong autocorrelation.

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
<th rowspan="9" valign="top">
loc_np0_beta_value
</th>
<th>
(0,)
</th>
<td>
kernel_01
</td>
<td>
-71.749649
</td>
<td>
243.404678
</td>
<td>
-471.154236
</td>
<td>
-78.070900
</td>
<td>
3.265136e+02
</td>
<td>
4000
</td>
<td>
107.242884
</td>
<td>
480.206111
</td>
<td>
1.030491
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
-1435.168457
</td>
<td>
234.020386
</td>
<td>
-1813.984619
</td>
<td>
-1439.191589
</td>
<td>
-1.045109e+03
</td>
<td>
4000
</td>
<td>
300.685537
</td>
<td>
799.823367
</td>
<td>
1.009126
</td>
</tr>
<tr>
<th>
(2,)
</th>
<td>
kernel_01
</td>
<td>
-693.983948
</td>
<td>
170.614014
</td>
<td>
-966.296927
</td>
<td>
-691.828979
</td>
<td>
-4.237409e+02
</td>
<td>
4000
</td>
<td>
422.634179
</td>
<td>
908.951817
</td>
<td>
1.012569
</td>
</tr>
<tr>
<th>
(3,)
</th>
<td>
kernel_01
</td>
<td>
-560.734131
</td>
<td>
112.971436
</td>
<td>
-742.432364
</td>
<td>
-564.023438
</td>
<td>
-3.731574e+02
</td>
<td>
4000
</td>
<td>
223.983433
</td>
<td>
551.776886
</td>
<td>
1.021564
</td>
</tr>
<tr>
<th>
(4,)
</th>
<td>
kernel_01
</td>
<td>
1126.935669
</td>
<td>
94.510315
</td>
<td>
971.606635
</td>
<td>
1125.880127
</td>
<td>
1.284334e+03
</td>
<td>
4000
</td>
<td>
314.077898
</td>
<td>
704.643443
</td>
<td>
1.009145
</td>
</tr>
<tr>
<th>
(5,)
</th>
<td>
kernel_01
</td>
<td>
-66.792961
</td>
<td>
32.871391
</td>
<td>
-123.013205
</td>
<td>
-66.058674
</td>
<td>
-1.292954e+01
</td>
<td>
4000
</td>
<td>
164.500902
</td>
<td>
632.450287
</td>
<td>
1.025022
</td>
</tr>
<tr>
<th>
(6,)
</th>
<td>
kernel_01
</td>
<td>
-213.597229
</td>
<td>
20.918447
</td>
<td>
-247.094501
</td>
<td>
-214.438644
</td>
<td>
-1.780851e+02
</td>
<td>
4000
</td>
<td>
344.216499
</td>
<td>
645.171342
</td>
<td>
1.014935
</td>
</tr>
<tr>
<th>
(7,)
</th>
<td>
kernel_01
</td>
<td>
111.644020
</td>
<td>
67.200516
</td>
<td>
10.435681
</td>
<td>
104.948242
</td>
<td>
2.337109e+02
</td>
<td>
4000
</td>
<td>
244.085149
</td>
<td>
475.721276
</td>
<td>
1.017609
</td>
</tr>
<tr>
<th>
(8,)
</th>
<td>
kernel_01
</td>
<td>
28.792318
</td>
<td>
17.524246
</td>
<td>
2.202152
</td>
<td>
27.455837
</td>
<td>
6.069218e+01
</td>
<td>
4000
</td>
<td>
228.925737
</td>
<td>
385.856747
</td>
<td>
1.020899
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
kernel_00
</td>
<td>
720253.937500
</td>
<td>
486326.312500
</td>
<td>
257366.581250
</td>
<td>
591308.312500
</td>
<td>
1.646430e+06
</td>
<td>
4000
</td>
<td>
1890.629863
</td>
<td>
2472.727683
</td>
<td>
1.001899
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
kernel_02
</td>
<td>
-24.568300
</td>
<td>
1.798792
</td>
<td>
-27.581183
</td>
<td>
-24.467516
</td>
<td>
-2.187579e+01
</td>
<td>
4000
</td>
<td>
16.928578
</td>
<td>
33.934520
</td>
<td>
1.176516
</td>
</tr>
<tr>
<th rowspan="9" valign="top">
scale_np0_beta_value
</th>
<th>
(0,)
</th>
<td>
kernel_04
</td>
<td>
6.429193
</td>
<td>
7.917597
</td>
<td>
-3.949329
</td>
<td>
4.958287
</td>
<td>
2.210393e+01
</td>
<td>
4000
</td>
<td>
62.329368
</td>
<td>
62.625436
</td>
<td>
1.047392
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
-0.844249
</td>
<td>
6.420584
</td>
<td>
-10.796793
</td>
<td>
-0.719076
</td>
<td>
8.953830e+00
</td>
<td>
4000
</td>
<td>
60.485515
</td>
<td>
89.620435
</td>
<td>
1.055760
</td>
</tr>
<tr>
<th>
(2,)
</th>
<td>
kernel_04
</td>
<td>
-14.543651
</td>
<td>
8.604739
</td>
<td>
-31.966818
</td>
<td>
-12.813149
</td>
<td>
-3.221321e+00
</td>
<td>
4000
</td>
<td>
44.754935
</td>
<td>
46.606683
</td>
<td>
1.059186
</td>
</tr>
<tr>
<th>
(3,)
</th>
<td>
kernel_04
</td>
<td>
8.757330
</td>
<td>
4.395264
</td>
<td>
2.127014
</td>
<td>
8.510785
</td>
<td>
1.613344e+01
</td>
<td>
4000
</td>
<td>
43.713822
</td>
<td>
207.391494
</td>
<td>
1.083672
</td>
</tr>
<tr>
<th>
(4,)
</th>
<td>
kernel_04
</td>
<td>
-1.344296
</td>
<td>
4.209912
</td>
<td>
-9.322977
</td>
<td>
-1.015017
</td>
<td>
5.067147e+00
</td>
<td>
4000
</td>
<td>
38.432665
</td>
<td>
27.255761
</td>
<td>
1.078597
</td>
</tr>
<tr>
<th>
(5,)
</th>
<td>
kernel_04
</td>
<td>
3.709730
</td>
<td>
1.829208
</td>
<td>
0.818840
</td>
<td>
3.691401
</td>
<td>
6.898891e+00
</td>
<td>
4000
</td>
<td>
68.721534
</td>
<td>
116.002352
</td>
<td>
1.027493
</td>
</tr>
<tr>
<th>
(6,)
</th>
<td>
kernel_04
</td>
<td>
0.522969
</td>
<td>
2.617249
</td>
<td>
-4.709338
</td>
<td>
1.073140
</td>
<td>
3.875310e+00
</td>
<td>
4000
</td>
<td>
35.732689
</td>
<td>
69.548887
</td>
<td>
1.088659
</td>
</tr>
<tr>
<th>
(7,)
</th>
<td>
kernel_04
</td>
<td>
-1.137452
</td>
<td>
3.631919
</td>
<td>
-7.263342
</td>
<td>
-1.241295
</td>
<td>
5.061523e+00
</td>
<td>
4000
</td>
<td>
65.055335
</td>
<td>
95.181598
</td>
<td>
1.054235
</td>
</tr>
<tr>
<th>
(8,)
</th>
<td>
kernel_04
</td>
<td>
-0.619457
</td>
<td>
1.702731
</td>
<td>
-3.915074
</td>
<td>
-0.276503
</td>
<td>
1.537570e+00
</td>
<td>
4000
</td>
<td>
35.525902
</td>
<td>
59.684818
</td>
<td>
1.067420
</td>
</tr>
<tr>
<th>
scale_np0_tau2_value
</th>
<th>
()
</th>
<td>
kernel_03
</td>
<td>
95.551109
</td>
<td>
127.029015
</td>
<td>
10.594745
</td>
<td>
55.283794
</td>
<td>
3.109806e+02
</td>
<td>
4000
</td>
<td>
50.197601
</td>
<td>
142.255749
</td>
<td>
1.028855
</td>
</tr>
<tr>
<th>
scale_p0_beta_value
</th>
<th>
(0,)
</th>
<td>
kernel_05
</td>
<td>
2.773644
</td>
<td>
0.068992
</td>
<td>
2.662522
</td>
<td>
2.772745
</td>
<td>
2.890415e+00
</td>
<td>
4000
</td>
<td>
408.181179
</td>
<td>
1309.091828
</td>
<td>
1.003256
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
9
</td>
<td>
0.00045
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
kernel_05
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
4
</td>
<td>
0.00020
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

``` python
fig = gs.plot_trace(results, "loc_p0_beta_value")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-1.png)

``` python
fig = gs.plot_trace(results, "loc_np0_tau2_value")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-2.png)

``` python
fig = gs.plot_trace(results, "loc_np0_beta_value")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-3.png)

``` python
fig = gs.plot_trace(results, "scale_p0_beta_value")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-4.png)

``` python
fig = gs.plot_trace(results, "scale_np0_tau2_value")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-5.png)

``` python
fig = gs.plot_trace(results, "scale_np0_beta_value")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-6.png)

To confirm that the chains have converged to reasonable values, here is
a plot of the estimated mean function:

``` python
summary = gs.Summary(results).to_dataframe().reset_index()
```

``` r
library(dplyr)
```


    Attaching package: 'dplyr'

    The following object is masked from 'package:MASS':

        select

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
f <- X %*% beta

beta0 <- summary %>%
  filter(variable == "loc_p0_beta_value") %>%
  group_by(var_index) %>%
  summarize(mean = mean(mean)) %>%
  ungroup()

beta0 <- beta0$mean

ggplot(data.frame(times = mcycle$times, mean = beta0 + f)) +
  geom_line(aes(times, mean), color = palette()[2], linewidth = 1) +
  geom_point(aes(times, accel), data = mcycle) +
  ggtitle("Estimated mean function") +
  theme_minimal()
```

![](04-mcycle_files/figure-commonmark/iwls-spline-13.png)

## NUTS sampler

As an alternative, we try a NUTS kernel which samples all model
parameters (regression coefficients and smoothing parameters) in one
block. To do so, we first need to log-transform the smoothing
parameters. This is the model graph before the transformation:

``` python
lsl.plot_vars(model)
```

![](04-mcycle_files/figure-commonmark/untransformed-graph-1.png)

Before transforming the smoothing parameters with the
`lsl.transform_parameter()` function, we first need to copy all model
nodes. Once this is done, we need to update the output nodes of the
smoothing parameters and rebuild the model. There are two additional
nodes in the new model graph.

``` python
import tensorflow_probability.substrates.jax.bijectors as tfb

nodes, _vars = model.pop_nodes_and_vars()

gb = lsl.GraphBuilder()
gb.add(_vars["response"])
```

    GraphBuilder<0 nodes, 1 vars>

``` python
_ = gb.transform(_vars["loc_np0_tau2"], tfb.Exp)
_ = gb.transform(_vars["scale_np0_tau2"], tfb.Exp)
model = gb.build_model()
lsl.plot_vars(model)
```

![](04-mcycle_files/figure-commonmark/transformed-graph-3.png)

Now we can set up the NUTS sampler, which is straightforward because we
are using only one kernel.

``` python
parameters = [name for name, var in model.vars.items() if var.parameter]

builder = gs.EngineBuilder(seed=42, num_chains=4)

builder.set_model(lsl.GooseModel(model))
builder.add_kernel(gs.NUTSKernel(parameters))
builder.set_initial_values(model.state)

builder.set_duration(warmup_duration=5000, posterior_duration=1000)

engine = builder.build()
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 41, 42, 50, 61 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 24, 20, 8, 16 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 49, 43, 18, 47 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 94, 92, 71, 96 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 191, 193, 142, 189 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 379, 374, 365, 367 / 400 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 800 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 735, 751, 750, 745 / 800 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 3300 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3098, 3084, 3034, 3072 / 3300 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 48, 42, 45, 47 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 978, 992, 968, 964 / 1000 transitions
    liesel.goose.engine - INFO - Finished epoch

The results are mixed. On the one hand, the NUTS sampler performs much
better on the intercepts (for both the mean and the standard deviation),
but on the other hand, the Metropolis-in-Gibbs sampler with the IWLS
kernels seems to work better for the spline coefficients.

``` python
results = engine.get_results()
gs.Summary(results)
```

    Parameter summary:

                                         kernel         mean  ...     ess_tail      rhat
    parameter                  index                          ...                       
    loc_np0_beta               (0,)   kernel_00   184.036636  ...    14.946803  3.236748
                               (1,)   kernel_00  -273.360443  ...    12.325592  3.363239
                               (2,)   kernel_00  -740.798828  ...    42.893821  1.411848
                               (3,)   kernel_00  -252.073593  ...   203.741898  1.190143
                               (4,)   kernel_00  1085.564331  ...   198.281495  1.050383
                               (5,)   kernel_00  -110.461342  ...  2275.842903  1.009886
                               (6,)   kernel_00  -209.517624  ...  1305.586163  1.011247
                               (7,)   kernel_00   -42.542221  ...   218.267383  1.113177
                               (8,)   kernel_00    -4.597168  ...   656.371539  1.076956
    loc_np0_tau2_transformed   ()     kernel_00    12.579103  ...  1279.937733  1.030615
    loc_p0_beta                (0,)   kernel_00   -21.830856  ...    57.630030  1.164192
    scale_np0_beta             (0,)   kernel_00     5.307924  ...   481.078983  1.024117
                               (1,)   kernel_00    -1.240266  ...  2265.030459  1.009408
                               (2,)   kernel_00   -16.626715  ...  2193.251674  1.017538
                               (3,)   kernel_00    15.601765  ...   143.618801  1.078908
                               (4,)   kernel_00    -3.942277  ...  2616.551185  1.034236
                               (5,)   kernel_00     5.939505  ...  1400.506537  1.043917
                               (6,)   kernel_00    -0.735045  ...  1296.869985  1.020721
                               (7,)   kernel_00     2.927418  ...  1495.753775  1.029497
                               (8,)   kernel_00    -0.888935  ...   962.829040  1.015447
    scale_np0_tau2_transformed ()     kernel_00     4.630955  ...  1703.860741  1.018547
    scale_p0_beta              (0,)   kernel_00     2.874944  ...    92.260779  1.139997

    [22 rows x 10 columns]

    Error summary:

                                                                              count  relative
    kernel    error_code error_msg                                 phase                     
    kernel_00 1          divergent transition                      warmup      1893   0.09465
                                                                   posterior      0   0.00000
              2          maximum tree depth                        warmup     15910   0.79550
                                                                   posterior   3900   0.97500
              3          divergent transition + maximum tree depth warmup       620   0.03100
                                                                   posterior      2   0.00050

``` python
fig = gs.plot_trace(results, "loc_p0_beta")
```

![](04-mcycle_files/figure-commonmark/nuts-traces-5.png)

``` python
fig = gs.plot_trace(results, "loc_np0_tau2_transformed")
```

![](04-mcycle_files/figure-commonmark/nuts-traces-6.png)

``` python
fig = gs.plot_trace(results, "loc_np0_beta")
```

![](04-mcycle_files/figure-commonmark/nuts-traces-7.png)

``` python
fig = gs.plot_trace(results, "scale_p0_beta")
```

![](04-mcycle_files/figure-commonmark/nuts-traces-8.png)

``` python
fig = gs.plot_trace(results, "scale_np0_tau2_transformed")
```

![](04-mcycle_files/figure-commonmark/nuts-traces-9.png)

``` python
fig = gs.plot_trace(results, "scale_np0_beta")
```

![](04-mcycle_files/figure-commonmark/nuts-traces-10.png)

Again, here is a plot of the estimated mean function:

``` python
summary = gs.Summary(results).to_dataframe().reset_index()
```

``` r
library(dplyr)
library(ggplot2)
library(reticulate)

summary <- py$summary
model <- py$model

beta <- summary %>%
  filter(variable == "loc_np0_beta") %>%
  group_by(var_index) %>%
  summarize(mean = mean(mean)) %>%
  ungroup()

beta <- beta$mean
X <- model$vars["loc_np0_X"]$value
f <- X %*% beta

beta0 <- summary %>%
  filter(variable == "loc_p0_beta") %>%
  group_by(var_index) %>%
  summarize(mean = mean(mean)) %>%
  ungroup()

beta0 <- beta0$mean

ggplot(data.frame(times = mcycle$times, mean = beta0 + f)) +
  geom_line(aes(times, mean), color = palette()[2], linewidth = 1) +
  geom_point(aes(times, accel), data = mcycle) +
  ggtitle("Estimated mean function") +
  theme_minimal()
```

![](04-mcycle_files/figure-commonmark/nuts-spline-17.png)
