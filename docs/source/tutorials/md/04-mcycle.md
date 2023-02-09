
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
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 1 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 0, 1, 0, 1 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 0, 0, 1, 0 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 1, 0, 0, 1 / 400 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 800 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 1, 1, 0, 0 / 800 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 3300 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 1, 0, 1, 0 / 3300 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 1, 1, 0, 0 / 50 transitions
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
kernel_04
</td>
<td>
-108.944023
</td>
<td>
261.579132
</td>
<td>
-534.661642
</td>
<td>
-112.349243
</td>
<td>
3.214457e+02
</td>
<td>
4000
</td>
<td>
23.188321
</td>
<td>
289.586868
</td>
<td>
1.130036
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
-1435.197754
</td>
<td>
236.186478
</td>
<td>
-1825.811743
</td>
<td>
-1431.409546
</td>
<td>
-1.062530e+03
</td>
<td>
4000
</td>
<td>
177.889227
</td>
<td>
277.960100
</td>
<td>
1.017449
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
-714.273010
</td>
<td>
179.665131
</td>
<td>
-1012.843674
</td>
<td>
-712.025330
</td>
<td>
-4.222988e+02
</td>
<td>
4000
</td>
<td>
133.030770
</td>
<td>
566.805593
</td>
<td>
1.034158
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
-564.613586
</td>
<td>
105.870987
</td>
<td>
-738.670807
</td>
<td>
-563.240082
</td>
<td>
-3.907725e+02
</td>
<td>
4000
</td>
<td>
325.079382
</td>
<td>
609.108075
</td>
<td>
1.023588
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
1131.977051
</td>
<td>
91.643028
</td>
<td>
984.666318
</td>
<td>
1127.913696
</td>
<td>
1.285110e+03
</td>
<td>
4000
</td>
<td>
199.313114
</td>
<td>
395.645071
</td>
<td>
1.012527
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
-59.648293
</td>
<td>
32.147057
</td>
<td>
-111.867696
</td>
<td>
-60.775717
</td>
<td>
-4.936346e+00
</td>
<td>
4000
</td>
<td>
141.453249
</td>
<td>
397.598892
</td>
<td>
1.023719
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
-213.432419
</td>
<td>
19.873352
</td>
<td>
-246.126113
</td>
<td>
-213.572990
</td>
<td>
-1.812692e+02
</td>
<td>
4000
</td>
<td>
180.098521
</td>
<td>
719.412407
</td>
<td>
1.039936
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
114.653244
</td>
<td>
68.110146
</td>
<td>
15.722488
</td>
<td>
109.121391
</td>
<td>
2.334605e+02
</td>
<td>
4000
</td>
<td>
144.603555
</td>
<td>
259.777208
</td>
<td>
1.033663
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
29.835463
</td>
<td>
17.647947
</td>
<td>
3.961023
</td>
<td>
28.399349
</td>
<td>
6.104354e+01
</td>
<td>
4000
</td>
<td>
111.550958
</td>
<td>
254.811488
</td>
<td>
1.048766
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
kernel_03
</td>
<td>
732711.062500
</td>
<td>
561087.750000
</td>
<td>
252670.012500
</td>
<td>
588705.218750
</td>
<td>
1.676414e+06
</td>
<td>
4000
</td>
<td>
1207.734937
</td>
<td>
2832.780357
</td>
<td>
1.003479
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
kernel_05
</td>
<td>
-23.971762
</td>
<td>
1.802968
</td>
<td>
-27.230812
</td>
<td>
-23.868549
</td>
<td>
-2.093058e+01
</td>
<td>
4000
</td>
<td>
7.874187
</td>
<td>
19.373105
</td>
<td>
1.475758
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
kernel_01
</td>
<td>
7.073123
</td>
<td>
11.288192
</td>
<td>
-7.711460
</td>
<td>
5.240811
</td>
<td>
3.042840e+01
</td>
<td>
4000
</td>
<td>
18.591045
</td>
<td>
84.850686
</td>
<td>
1.190827
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
-1.583312
</td>
<td>
5.854820
</td>
<td>
-11.653714
</td>
<td>
-1.196289
</td>
<td>
8.011723e+00
</td>
<td>
4000
</td>
<td>
49.972196
</td>
<td>
221.776302
</td>
<td>
1.059100
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
-16.763771
</td>
<td>
10.055695
</td>
<td>
-34.984083
</td>
<td>
-15.423145
</td>
<td>
-2.512168e+00
</td>
<td>
4000
</td>
<td>
13.108076
</td>
<td>
77.963177
</td>
<td>
1.238042
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
10.695534
</td>
<td>
5.185567
</td>
<td>
2.713498
</td>
<td>
10.655252
</td>
<td>
1.887960e+01
</td>
<td>
4000
</td>
<td>
20.185052
</td>
<td>
231.770226
</td>
<td>
1.148924
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
-2.437805
</td>
<td>
4.297496
</td>
<td>
-9.761899
</td>
<td>
-1.990282
</td>
<td>
4.129804e+00
</td>
<td>
4000
</td>
<td>
14.861994
</td>
<td>
118.113220
</td>
<td>
1.213543
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
4.017594
</td>
<td>
1.891968
</td>
<td>
0.862828
</td>
<td>
3.958074
</td>
<td>
7.188039e+00
</td>
<td>
4000
</td>
<td>
14.654025
</td>
<td>
106.364459
</td>
<td>
1.191113
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
-0.316734
</td>
<td>
3.251244
</td>
<td>
-6.013668
</td>
<td>
0.202090
</td>
<td>
4.053662e+00
</td>
<td>
4000
</td>
<td>
11.430888
</td>
<td>
83.437761
</td>
<td>
1.277837
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
-0.233323
</td>
<td>
3.420532
</td>
<td>
-5.495466
</td>
<td>
-0.533906
</td>
<td>
5.871558e+00
</td>
<td>
4000
</td>
<td>
52.673577
</td>
<td>
193.124140
</td>
<td>
1.050747
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
-1.017423
</td>
<td>
1.968827
</td>
<td>
-4.636996
</td>
<td>
-0.683810
</td>
<td>
1.634275e+00
</td>
<td>
4000
</td>
<td>
11.507294
</td>
<td>
91.549336
</td>
<td>
1.275885
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
kernel_00
</td>
<td>
134.179840
</td>
<td>
169.470062
</td>
<td>
10.974957
</td>
<td>
74.841915
</td>
<td>
4.529928e+02
</td>
<td>
4000
</td>
<td>
13.729983
</td>
<td>
126.142356
</td>
<td>
1.214373
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
kernel_02
</td>
<td>
2.763654
</td>
<td>
0.069620
</td>
<td>
2.651497
</td>
<td>
2.763374
</td>
<td>
2.880432e+00
</td>
<td>
4000
</td>
<td>
84.591252
</td>
<td>
556.745930
</td>
<td>
1.040928
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
11
</td>
<td>
0.00055
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
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 43, 44, 50, 61 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 24, 21, 4, 11 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 46, 49, 33, 35 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 94, 88, 93, 99 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 197, 187, 185, 191 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 373, 379, 378, 374 / 400 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 800 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 776, 767, 755, 761 / 800 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 3300 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3159, 3208, 3179, 3223 / 3300 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 49, 48, 50, 49 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 997, 999, 999, 999 / 1000 transitions
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

                                         kernel        mean  ...     ess_tail      rhat
    parameter                  index                         ...
    loc_np0_beta               (0,)   kernel_00  112.808548  ...    11.496063  4.004832
                               (1,)   kernel_00 -101.381714  ...    11.450084  3.614009
                               (2,)   kernel_00  -78.493156  ...    25.473711  2.189577
                               (3,)   kernel_00  -82.997673  ...    11.447161  2.875846
                               (4,)   kernel_00  520.973450  ...    17.907814  2.575485
                               (5,)   kernel_00  -60.874229  ...    44.119444  1.230434
                               (6,)   kernel_00 -107.799263  ...    15.161721  2.693133
                               (7,)   kernel_00   22.326500  ...    12.790733  2.171271
                               (8,)   kernel_00   15.648965  ...    30.195664  2.269841
    loc_np0_tau2_transformed   ()     kernel_00   10.494014  ...    18.550497  1.728875
    loc_p0_beta                (0,)   kernel_00  -15.573112  ...    23.851092  1.647181
    scale_np0_beta             (0,)   kernel_00    2.479368  ...    15.636459  2.596379
                               (1,)   kernel_00    7.729941  ...    99.882154  1.142289
                               (2,)   kernel_00  -16.636984  ...    42.946903  1.580868
                               (3,)   kernel_00   15.459176  ...    37.680018  1.372760
                               (4,)   kernel_00   -8.674792  ...    44.500339  1.562794
                               (5,)   kernel_00    6.616967  ...    36.139064  1.530659
                               (6,)   kernel_00    0.018463  ...    53.694721  1.430559
                               (7,)   kernel_00    0.993386  ...  1166.093659  1.080550
                               (8,)   kernel_00   -1.298947  ...    52.449575  1.487767
    scale_np0_tau2_transformed ()     kernel_00    4.588464  ...    36.309556  1.468392
    scale_p0_beta              (0,)   kernel_00    3.028978  ...    48.338872  1.406440

    [22 rows x 10 columns]

    Error summary:

                                                                              count  relative
    kernel    error_code error_msg                                 phase
    kernel_00 1          divergent transition                      warmup      2523   0.12615
                                                                   posterior      0   0.00000
              2          maximum tree depth                        warmup     15765   0.78825
                                                                   posterior   3994   0.99850
              3          divergent transition + maximum tree depth warmup       795   0.03975
                                                                   posterior      0   0.00000

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
