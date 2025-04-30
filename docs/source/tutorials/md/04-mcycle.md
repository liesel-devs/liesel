
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
builder.show_progress = False

engine = builder.build()
engine.sample_all_epochs()
```

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
loc_np0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_04
</td>
<td>
-89.783
</td>
<td>
243.714
</td>
<td>
-483.961
</td>
<td>
-93.170
</td>
<td>
317.212
</td>
<td>
4000
</td>
<td>
35.650
</td>
<td>
419.354
</td>
<td>
1.079
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
-1450.583
</td>
<td>
251.260
</td>
<td>
-1876.271
</td>
<td>
-1440.334
</td>
<td>
-1060.724
</td>
<td>
4000
</td>
<td>
292.621
</td>
<td>
649.406
</td>
<td>
1.018
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
-685.581
</td>
<td>
170.859
</td>
<td>
-963.474
</td>
<td>
-682.322
</td>
<td>
-406.089
</td>
<td>
4000
</td>
<td>
232.319
</td>
<td>
413.269
</td>
<td>
1.017
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
-566.567
</td>
<td>
113.858
</td>
<td>
-748.643
</td>
<td>
-569.450
</td>
<td>
-375.760
</td>
<td>
4000
</td>
<td>
200.408
</td>
<td>
587.108
</td>
<td>
1.019
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
1117.895
</td>
<td>
94.023
</td>
<td>
963.678
</td>
<td>
1115.170
</td>
<td>
1269.938
</td>
<td>
4000
</td>
<td>
294.993
</td>
<td>
818.162
</td>
<td>
1.014
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
-64.185
</td>
<td>
34.569
</td>
<td>
-121.355
</td>
<td>
-62.629
</td>
<td>
-8.844
</td>
<td>
4000
</td>
<td>
95.479
</td>
<td>
288.695
</td>
<td>
1.043
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
-211.062
</td>
<td>
21.122
</td>
<td>
-246.101
</td>
<td>
-211.329
</td>
<td>
-176.134
</td>
<td>
4000
</td>
<td>
108.539
</td>
<td>
444.519
</td>
<td>
1.039
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
116.051
</td>
<td>
69.206
</td>
<td>
17.010
</td>
<td>
108.444
</td>
<td>
240.738
</td>
<td>
4000
</td>
<td>
164.199
</td>
<td>
356.351
</td>
<td>
1.010
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
30.198
</td>
<td>
17.908
</td>
<td>
4.629
</td>
<td>
27.936
</td>
<td>
62.939
</td>
<td>
4000
</td>
<td>
143.481
</td>
<td>
301.179
</td>
<td>
1.017
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
kernel_03
</td>
<td>
736267.500
</td>
<td>
576384.875
</td>
<td>
254046.856
</td>
<td>
586045.219
</td>
<td>
1712081.537
</td>
<td>
4000
</td>
<td>
1716.899
</td>
<td>
2508.099
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
kernel_05
</td>
<td>
-24.772
</td>
<td>
2.514
</td>
<td>
-28.690
</td>
<td>
-24.817
</td>
<td>
-20.496
</td>
<td>
4000
</td>
<td>
9.321
</td>
<td>
19.502
</td>
<td>
1.380
</td>
</tr>
<tr>
<th rowspan="9" valign="top">
scale_np0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_01
</td>
<td>
6.953
</td>
<td>
9.373
</td>
<td>
-5.137
</td>
<td>
5.156
</td>
<td>
24.396
</td>
<td>
4000
</td>
<td>
19.041
</td>
<td>
128.454
</td>
<td>
1.145
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
-2.119
</td>
<td>
6.417
</td>
<td>
-13.762
</td>
<td>
-1.500
</td>
<td>
7.521
</td>
<td>
4000
</td>
<td>
24.990
</td>
<td>
63.433
</td>
<td>
1.122
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
-16.403
</td>
<td>
9.940
</td>
<td>
-33.269
</td>
<td>
-16.295
</td>
<td>
-1.356
</td>
<td>
4000
</td>
<td>
15.048
</td>
<td>
38.506
</td>
<td>
1.220
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
9.526
</td>
<td>
4.653
</td>
<td>
2.741
</td>
<td>
9.110
</td>
<td>
17.955
</td>
<td>
4000
</td>
<td>
17.114
</td>
<td>
127.404
</td>
<td>
1.192
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
-2.031
</td>
<td>
3.875
</td>
<td>
-8.943
</td>
<td>
-1.594
</td>
<td>
3.610
</td>
<td>
4000
</td>
<td>
34.877
</td>
<td>
97.160
</td>
<td>
1.084
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
3.733
</td>
<td>
1.977
</td>
<td>
0.723
</td>
<td>
3.572
</td>
<td>
6.974
</td>
<td>
4000
</td>
<td>
22.106
</td>
<td>
43.125
</td>
<td>
1.144
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
-0.032
</td>
<td>
2.937
</td>
<td>
-5.476
</td>
<td>
0.459
</td>
<td>
4.034
</td>
<td>
4000
</td>
<td>
13.837
</td>
<td>
41.847
</td>
<td>
1.203
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
-0.828
</td>
<td>
3.341
</td>
<td>
-5.853
</td>
<td>
-1.107
</td>
<td>
4.954
</td>
<td>
4000
</td>
<td>
56.561
</td>
<td>
173.938
</td>
<td>
1.065
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
-0.929
</td>
<td>
1.772
</td>
<td>
-4.211
</td>
<td>
-0.641
</td>
<td>
1.491
</td>
<td>
4000
</td>
<td>
16.101
</td>
<td>
87.743
</td>
<td>
1.178
</td>
</tr>
<tr>
<th>
scale_np0_tau2
</th>
<th>
()
</th>
<td>
kernel_00
</td>
<td>
121.644
</td>
<td>
171.814
</td>
<td>
7.737
</td>
<td>
72.676
</td>
<td>
393.397
</td>
<td>
4000
</td>
<td>
16.500
</td>
<td>
109.850
</td>
<td>
1.187
</td>
</tr>
<tr>
<th>
scale_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_02
</td>
<td>
2.773
</td>
<td>
0.069
</td>
<td>
2.661
</td>
<td>
2.773
</td>
<td>
2.888
</td>
<td>
4000
</td>
<td>
258.903
</td>
<td>
639.815
</td>
<td>
1.033
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
1
</td>
<td>
0.000
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

``` python
fig = gs.plot_trace(results, "loc_p0_beta")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-1.png)

``` python
fig = gs.plot_trace(results, "loc_np0_tau2")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-2.png)

``` python
fig = gs.plot_trace(results, "loc_np0_beta")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-3.png)

``` python
fig = gs.plot_trace(results, "scale_p0_beta")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-4.png)

``` python
fig = gs.plot_trace(results, "scale_np0_tau2")
```

![](04-mcycle_files/figure-commonmark/iwls-traces-5.png)

``` python
fig = gs.plot_trace(results, "scale_np0_beta")
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
f <- X %*% beta

beta0 <- summary %>%
  filter(variable == "loc_p0_beta") %>%
  group_by(var_index) %>%
  summarize(mean = mean(mean)) %>%
  ungroup()

beta0 <- beta0$mean

ggplot(data.frame(times = mcycle$times, mean = beta0 + f)) +
  geom_line(aes(times, mean), color = palette()[2], size = 1) +
  geom_point(aes(times, accel), data = mcycle) +
  ggtitle("Estimated mean function") +
  theme_minimal()
```

    Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
    ℹ Please use `linewidth` instead.

![](04-mcycle_files/figure-commonmark/iwls-spline-13.png)

## NUTS sampler

As an alternative, we try using NUTS kernels for all parameters. To do
so, we first need to log-transform the smoothing parameters. This is the
model graph before the transformation:

``` python
lsl.plot_vars(model)
```

![](04-mcycle_files/figure-commonmark/untransformed-graph-1.png)

To transform the smoothing parameters with the method
{meth}`.Var.transform`, we need to retrieve the nodes and vars form the
model. This is necessary, because while they are part of a model, the
inputs and outputs of nodes and vars cannot be changed. We retrieve the
nodes and vars using {meth}`.Model.pop_nodes_and_vars`, which renders
the model empty.

After transformation, there are two additional nodes in the new model
graph.

``` python
import tensorflow_probability.substrates.jax.bijectors as tfb

nodes, _vars = model.pop_nodes_and_vars()

_vars["loc_np0_tau2"].transform(tfb.Exp())
```

    Var(name="loc_np0_tau2_transformed")

``` python
_vars["scale_np0_tau2"].transform(tfb.Exp())
```

    Var(name="scale_np0_tau2_transformed")

``` python
gb = lsl.GraphBuilder().add(_vars["response"])

model = gb.build_model()
lsl.plot_vars(model)
```

![](04-mcycle_files/figure-commonmark/transformed-graph-3.png)

Now we can set up the NUTS sampler. In complex models like this one it
can be very beneficial to use individual NUTS samplers for blocks of
parameters. This is pretty much the same strategy that we apply to the
IWLS sampler, too.

``` python
builder = gs.EngineBuilder(seed=42, num_chains=4)

builder.set_model(gs.LieselInterface(model))

# add NUTS kernels
parameters = [name for name, var in model.vars.items() if var.parameter]

for parameter in parameters:
  builder.add_kernel(gs.NUTSKernel([parameter]))


builder.set_initial_values(model.state)

builder.set_epochs(
  gs.stan_epochs(warmup_duration=5000, posterior_duration=1000, init_duration=750, term_duration=500)
)

builder.show_progress = False

engine = builder.build()
engine.sample_all_epochs()
```

The NUTS sampler overall seems to do a good job - and even yields higher
effective sample sizes than the IWLS sampler, especially for the spline
coefficients of the scale model.

``` python
results = engine.get_results()
gs.Summary(results)
```

<div class="cell-output-display">

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
loc_np0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_03
</td>
<td>
-88.104
</td>
<td>
252.529
</td>
<td>
-498.737
</td>
<td>
-94.419
</td>
<td>
337.766
</td>
<td>
4000
</td>
<td>
474.180
</td>
<td>
1377.730
</td>
<td>
1.008
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
-1453.895
</td>
<td>
245.675
</td>
<td>
-1843.600
</td>
<td>
-1453.718
</td>
<td>
-1053.058
</td>
<td>
4000
</td>
<td>
765.684
</td>
<td>
1813.545
</td>
<td>
1.006
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
-681.106
</td>
<td>
182.549
</td>
<td>
-985.511
</td>
<td>
-682.477
</td>
<td>
-377.593
</td>
<td>
4000
</td>
<td>
1104.403
</td>
<td>
1789.338
</td>
<td>
1.001
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
-554.689
</td>
<td>
113.558
</td>
<td>
-735.240
</td>
<td>
-553.886
</td>
<td>
-365.407
</td>
<td>
4000
</td>
<td>
1009.067
</td>
<td>
1669.084
</td>
<td>
1.005
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
1126.253
</td>
<td>
98.042
</td>
<td>
968.061
</td>
<td>
1126.222
</td>
<td>
1288.899
</td>
<td>
4000
</td>
<td>
1400.525
</td>
<td>
1695.291
</td>
<td>
1.003
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
-65.458
</td>
<td>
34.845
</td>
<td>
-121.377
</td>
<td>
-65.759
</td>
<td>
-6.803
</td>
<td>
4000
</td>
<td>
365.669
</td>
<td>
798.602
</td>
<td>
1.010
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
-210.146
</td>
<td>
23.005
</td>
<td>
-245.842
</td>
<td>
-210.666
</td>
<td>
-172.913
</td>
<td>
4000
</td>
<td>
1112.555
</td>
<td>
1175.961
</td>
<td>
1.003
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
122.930
</td>
<td>
80.350
</td>
<td>
12.235
</td>
<td>
113.462
</td>
<td>
269.568
</td>
<td>
4000
</td>
<td>
602.146
</td>
<td>
586.165
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
31.984
</td>
<td>
21.059
</td>
<td>
3.020
</td>
<td>
29.346
</td>
<td>
70.037
</td>
<td>
4000
</td>
<td>
556.233
</td>
<td>
544.773
</td>
<td>
1.005
</td>
</tr>
<tr>
<th>
loc_np0_tau2_transformed
</th>
<th>
()
</th>
<td>
kernel_04
</td>
<td>
13.329
</td>
<td>
0.571
</td>
<td>
12.460
</td>
<td>
13.294
</td>
<td>
14.329
</td>
<td>
4000
</td>
<td>
1090.885
</td>
<td>
1274.243
</td>
<td>
1.003
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
kernel_05
</td>
<td>
-24.860
</td>
<td>
1.858
</td>
<td>
-27.923
</td>
<td>
-24.778
</td>
<td>
-21.861
</td>
<td>
4000
</td>
<td>
54.740
</td>
<td>
105.458
</td>
<td>
1.069
</td>
</tr>
<tr>
<th rowspan="9" valign="top">
scale_np0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_00
</td>
<td>
6.745
</td>
<td>
9.369
</td>
<td>
-6.052
</td>
<td>
5.247
</td>
<td>
24.883
</td>
<td>
4000
</td>
<td>
474.570
</td>
<td>
946.737
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
-1.733
</td>
<td>
6.367
</td>
<td>
-13.032
</td>
<td>
-1.304
</td>
<td>
8.003
</td>
<td>
4000
</td>
<td>
746.909
</td>
<td>
894.980
</td>
<td>
1.005
</td>
</tr>
<tr>
<th>
(2,)
</th>
<td>
kernel_00
</td>
<td>
-15.173
</td>
<td>
8.991
</td>
<td>
-31.192
</td>
<td>
-14.309
</td>
<td>
-2.102
</td>
<td>
4000
</td>
<td>
305.664
</td>
<td>
1123.548
</td>
<td>
1.008
</td>
</tr>
<tr>
<th>
(3,)
</th>
<td>
kernel_00
</td>
<td>
10.141
</td>
<td>
5.134
</td>
<td>
2.598
</td>
<td>
9.723
</td>
<td>
19.043
</td>
<td>
4000
</td>
<td>
303.817
</td>
<td>
718.883
</td>
<td>
1.009
</td>
</tr>
<tr>
<th>
(4,)
</th>
<td>
kernel_00
</td>
<td>
-1.745
</td>
<td>
4.021
</td>
<td>
-8.910
</td>
<td>
-1.490
</td>
<td>
4.313
</td>
<td>
4000
</td>
<td>
523.670
</td>
<td>
1183.290
</td>
<td>
1.003
</td>
</tr>
<tr>
<th>
(5,)
</th>
<td>
kernel_00
</td>
<td>
4.106
</td>
<td>
2.021
</td>
<td>
1.005
</td>
<td>
4.023
</td>
<td>
7.558
</td>
<td>
4000
</td>
<td>
385.347
</td>
<td>
639.357
</td>
<td>
1.009
</td>
</tr>
<tr>
<th>
(6,)
</th>
<td>
kernel_00
</td>
<td>
0.222
</td>
<td>
2.756
</td>
<td>
-4.833
</td>
<td>
0.566
</td>
<td>
4.138
</td>
<td>
4000
</td>
<td>
302.514
</td>
<td>
943.799
</td>
<td>
1.005
</td>
</tr>
<tr>
<th>
(7,)
</th>
<td>
kernel_00
</td>
<td>
0.062
</td>
<td>
4.119
</td>
<td>
-6.222
</td>
<td>
-0.243
</td>
<td>
7.083
</td>
<td>
4000
</td>
<td>
493.763
</td>
<td>
595.323
</td>
<td>
1.006
</td>
</tr>
<tr>
<th>
(8,)
</th>
<td>
kernel_00
</td>
<td>
-0.678
</td>
<td>
1.709
</td>
<td>
-3.859
</td>
<td>
-0.432
</td>
<td>
1.655
</td>
<td>
4000
</td>
<td>
345.563
</td>
<td>
728.305
</td>
<td>
1.004
</td>
</tr>
<tr>
<th>
scale_np0_tau2_transformed
</th>
<th>
()
</th>
<td>
kernel_01
</td>
<td>
4.160
</td>
<td>
1.090
</td>
<td>
2.293
</td>
<td>
4.222
</td>
<td>
5.830
</td>
<td>
4000
</td>
<td>
226.354
</td>
<td>
481.191
</td>
<td>
1.012
</td>
</tr>
<tr>
<th>
scale_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_02
</td>
<td>
2.772
</td>
<td>
0.068
</td>
<td>
2.664
</td>
<td>
2.770
</td>
<td>
2.886
</td>
<td>
4000
</td>
<td>
806.194
</td>
<td>
2050.150
</td>
<td>
1.001
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
<th rowspan="4" valign="top">
kernel_00
</th>
<th rowspan="2" valign="top">
1
</th>
<th rowspan="2" valign="top">
divergent transition
</th>
<th>
warmup
</th>
<td>
1764
</td>
<td>
0.088
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
2
</th>
<th rowspan="2" valign="top">
maximum tree depth
</th>
<th>
warmup
</th>
<td>
75
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
0.000
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
kernel_01
</th>
<th rowspan="2" valign="top">
1
</th>
<th rowspan="2" valign="top">
divergent transition
</th>
<th>
warmup
</th>
<td>
107
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
<tr>
<th rowspan="2" valign="top">
kernel_02
</th>
<th rowspan="2" valign="top">
1
</th>
<th rowspan="2" valign="top">
divergent transition
</th>
<th>
warmup
</th>
<td>
63
</td>
<td>
0.003
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
<th rowspan="4" valign="top">
kernel_03
</th>
<th rowspan="2" valign="top">
1
</th>
<th rowspan="2" valign="top">
divergent transition
</th>
<th>
warmup
</th>
<td>
1150
</td>
<td>
0.057
</td>
</tr>
<tr>
<th>
posterior
</th>
<td>
66
</td>
<td>
0.016
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
2
</th>
<th rowspan="2" valign="top">
maximum tree depth
</th>
<th>
warmup
</th>
<td>
3441
</td>
<td>
0.172
</td>
</tr>
<tr>
<th>
posterior
</th>
<td>
119
</td>
<td>
0.030
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
kernel_04
</th>
<th rowspan="2" valign="top">
1
</th>
<th rowspan="2" valign="top">
divergent transition
</th>
<th>
warmup
</th>
<td>
117
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
kernel_05
</th>
<th rowspan="2" valign="top">
1
</th>
<th rowspan="2" valign="top">
divergent transition
</th>
<th>
warmup
</th>
<td>
145
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
1
</td>
<td>
0.000
</td>
</tr>
</tbody>
</table>

</div>

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
  geom_line(aes(times, mean), color = palette()[2], size = 1) +
  geom_point(aes(times, accel), data = mcycle) +
  ggtitle("Estimated mean function") +
  theme_minimal()
```

![](04-mcycle_files/figure-commonmark/nuts-spline-17.png)
