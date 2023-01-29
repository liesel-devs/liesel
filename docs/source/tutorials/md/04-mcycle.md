
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
    liesel.goose.engine - WARNING - Errors per chain for kernel_03: 1, 1, 1, 1 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 0 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 1, 0, 0 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 0, 0 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 0, 1, 0 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 0, 1 / 400 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 800 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 3300 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 1, 0, 1 / 3300 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 0 / 50 transitions
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
kernel_05
</td>
<td>
-56.897034
</td>
<td>
247.331650
</td>
<td>
-454.135793
</td>
<td>
-59.564489
</td>
<td>
3.560394e+02
</td>
<td>
4000
</td>
<td>
86.925144
</td>
<td>
344.533460
</td>
<td>
1.020005
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_05
</td>
<td>
-1432.163330
</td>
<td>
243.634781
</td>
<td>
-1827.563367
</td>
<td>
-1433.617493
</td>
<td>
-1.021385e+03
</td>
<td>
4000
</td>
<td>
264.138120
</td>
<td>
587.754383
</td>
<td>
1.026829
</td>
</tr>
<tr>
<th>
(2,)
</th>
<td>
kernel_05
</td>
<td>
-677.383667
</td>
<td>
172.859512
</td>
<td>
-964.781274
</td>
<td>
-678.492889
</td>
<td>
-3.924719e+02
</td>
<td>
4000
</td>
<td>
330.538765
</td>
<td>
713.160888
</td>
<td>
1.006775
</td>
</tr>
<tr>
<th>
(3,)
</th>
<td>
kernel_05
</td>
<td>
-562.555054
</td>
<td>
107.869606
</td>
<td>
-736.960809
</td>
<td>
-561.708557
</td>
<td>
-3.818610e+02
</td>
<td>
4000
</td>
<td>
380.408678
</td>
<td>
840.415182
</td>
<td>
1.013114
</td>
</tr>
<tr>
<th>
(4,)
</th>
<td>
kernel_05
</td>
<td>
1122.058594
</td>
<td>
95.760429
</td>
<td>
965.669815
</td>
<td>
1122.442627
</td>
<td>
1.274909e+03
</td>
<td>
4000
</td>
<td>
348.533748
</td>
<td>
919.526419
</td>
<td>
1.006122
</td>
</tr>
<tr>
<th>
(5,)
</th>
<td>
kernel_05
</td>
<td>
-69.086502
</td>
<td>
35.034733
</td>
<td>
-127.649308
</td>
<td>
-68.869061
</td>
<td>
-1.212329e+01
</td>
<td>
4000
</td>
<td>
39.768700
</td>
<td>
185.049338
</td>
<td>
1.069330
</td>
</tr>
<tr>
<th>
(6,)
</th>
<td>
kernel_05
</td>
<td>
-211.887573
</td>
<td>
21.014408
</td>
<td>
-245.214204
</td>
<td>
-212.456818
</td>
<td>
-1.762564e+02
</td>
<td>
4000
</td>
<td>
265.241672
</td>
<td>
729.909122
</td>
<td>
1.011962
</td>
</tr>
<tr>
<th>
(7,)
</th>
<td>
kernel_05
</td>
<td>
111.441582
</td>
<td>
65.166245
</td>
<td>
11.178269
</td>
<td>
107.475796
</td>
<td>
2.308414e+02
</td>
<td>
4000
</td>
<td>
279.820418
</td>
<td>
419.138988
</td>
<td>
1.009782
</td>
</tr>
<tr>
<th>
(8,)
</th>
<td>
kernel_05
</td>
<td>
29.027969
</td>
<td>
17.128319
</td>
<td>
2.301867
</td>
<td>
28.202175
</td>
<td>
5.981050e+01
</td>
<td>
4000
</td>
<td>
177.594034
</td>
<td>
395.567662
</td>
<td>
1.009572
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
kernel_04
</td>
<td>
716900.687500
</td>
<td>
516016.562500
</td>
<td>
251151.125781
</td>
<td>
578716.750000
</td>
<td>
1.596758e+06
</td>
<td>
4000
</td>
<td>
1571.804489
</td>
<td>
2831.664780
</td>
<td>
1.002272
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
kernel_00
</td>
<td>
-25.349672
</td>
<td>
2.383100
</td>
<td>
-28.855255
</td>
<td>
-25.531158
</td>
<td>
-2.050030e+01
</td>
<td>
4000
</td>
<td>
7.199367
</td>
<td>
14.639932
</td>
<td>
1.531432
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
kernel_02
</td>
<td>
7.611047
</td>
<td>
9.393407
</td>
<td>
-4.532119
</td>
<td>
5.480760
</td>
<td>
2.653834e+01
</td>
<td>
4000
</td>
<td>
25.548312
</td>
<td>
143.008311
</td>
<td>
1.115716
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
-1.288361
</td>
<td>
6.576174
</td>
<td>
-12.128021
</td>
<td>
-0.989132
</td>
<td>
8.657103e+00
</td>
<td>
4000
</td>
<td>
24.533371
</td>
<td>
98.649428
</td>
<td>
1.128774
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
-14.753112
</td>
<td>
8.492762
</td>
<td>
-30.583091
</td>
<td>
-13.807506
</td>
<td>
-1.904776e+00
</td>
<td>
4000
</td>
<td>
44.008790
</td>
<td>
127.630838
</td>
<td>
1.111598
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
9.821408
</td>
<td>
4.455523
</td>
<td>
2.542726
</td>
<td>
9.521981
</td>
<td>
1.737749e+01
</td>
<td>
4000
</td>
<td>
21.175103
</td>
<td>
76.559590
</td>
<td>
1.128899
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
-2.118350
</td>
<td>
3.496424
</td>
<td>
-8.042471
</td>
<td>
-1.970381
</td>
<td>
3.903238e+00
</td>
<td>
4000
</td>
<td>
59.201777
</td>
<td>
104.012029
</td>
<td>
1.055930
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
4.000916
</td>
<td>
1.633981
</td>
<td>
1.518691
</td>
<td>
3.890211
</td>
<td>
6.698378e+00
</td>
<td>
4000
</td>
<td>
87.647512
</td>
<td>
179.785352
</td>
<td>
1.025957
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
0.100253
</td>
<td>
2.363394
</td>
<td>
-4.247794
</td>
<td>
0.535460
</td>
<td>
3.399006e+00
</td>
<td>
4000
</td>
<td>
31.448264
</td>
<td>
221.802043
</td>
<td>
1.124099
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
-0.453792
</td>
<td>
3.561935
</td>
<td>
-6.129934
</td>
<td>
-0.467416
</td>
<td>
5.298316e+00
</td>
<td>
4000
</td>
<td>
36.215595
</td>
<td>
119.368682
</td>
<td>
1.099000
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
-0.807665
</td>
<td>
1.569547
</td>
<td>
-3.861715
</td>
<td>
-0.535160
</td>
<td>
1.364098e+00
</td>
<td>
4000
</td>
<td>
25.103260
</td>
<td>
88.916374
</td>
<td>
1.148943
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
kernel_01
</td>
<td>
106.743095
</td>
<td>
125.970360
</td>
<td>
12.369365
</td>
<td>
64.301758
</td>
<td>
3.389188e+02
</td>
<td>
4000
</td>
<td>
23.438165
</td>
<td>
103.573026
</td>
<td>
1.126911
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
kernel_03
</td>
<td>
2.770894
</td>
<td>
0.072172
</td>
<td>
2.657961
</td>
<td>
2.768370
</td>
<td>
2.892691e+00
</td>
<td>
4000
</td>
<td>
207.094498
</td>
<td>
1054.961240
</td>
<td>
1.018275
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
15
</td>
<td>
0.00075
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
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 45, 46, 50, 61 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 22, 21, 9, 15 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 49, 43, 47, 32 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 91, 88, 91, 85 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 188, 185, 182, 184 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 380, 367, 368, 369 / 400 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 800 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 762, 746, 738, 730 / 800 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 3300 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3098, 3110, 3071, 3095 / 3300 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 46, 46, 47, 49 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 973, 993, 975, 987 / 1000 transitions
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
    loc_np0_beta               (0,)   kernel_00    60.347748  ...    17.366253  2.323560
                               (1,)   kernel_00 -1408.437256  ...   367.645453  1.035360
                               (2,)   kernel_00  -644.952148  ...  1405.174707  1.031278
                               (3,)   kernel_00  -572.595886  ...  1779.571668  1.001492
                               (4,)   kernel_00  1112.969727  ...  2584.603426  1.003619
                               (5,)   kernel_00   -71.841721  ...   986.436761  1.004968
                               (6,)   kernel_00  -214.100571  ...  1477.050275  1.007300
                               (7,)   kernel_00    98.718842  ...   281.350488  1.030520
                               (8,)   kernel_00    24.159561  ...   280.948179  1.039667
    loc_np0_tau2_transformed   ()     kernel_00    13.267749  ...  1871.173308  1.004598
    loc_p0_beta                (0,)   kernel_00   -25.210264  ...   351.970698  1.010474
    scale_np0_beta             (0,)   kernel_00     1.955048  ...    11.504319  2.324224
                               (1,)   kernel_00    -0.750614  ...   581.175568  1.007741
                               (2,)   kernel_00   -10.714042  ...   984.229950  1.015709
                               (3,)   kernel_00     7.743821  ...  1134.242240  1.008727
                               (4,)   kernel_00    -0.474424  ...  1380.880565  1.004669
                               (5,)   kernel_00     3.188846  ...  1241.694249  1.006054
                               (6,)   kernel_00     1.574633  ...  1101.072198  1.026402
                               (7,)   kernel_00    -1.180878  ...   976.510960  1.004117
                               (8,)   kernel_00     0.092508  ...   594.551919  1.040337
    scale_np0_tau2_transformed ()     kernel_00     3.540871  ...  1598.675741  1.018582
    scale_p0_beta              (0,)   kernel_00     2.783941  ...  2480.583803  1.003165

    [22 rows x 10 columns]

    Error summary:

                                                                              count  relative
    kernel    error_code error_msg                                 phase                     
    kernel_00 1          divergent transition                      warmup      1484    0.0742
                                                                   posterior     12    0.0030
              2          maximum tree depth                        warmup     16506    0.8253
                                                                   posterior   3910    0.9775
              3          divergent transition + maximum tree depth warmup       566    0.0283
                                                                   posterior      6    0.0015

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
