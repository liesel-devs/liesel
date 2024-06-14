
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

    Please make sure you are using a virtual or conda environment with Liesel installed, e.g. using `reticulate::use_virtualenv()` or `reticulate::use_condaenv()`. See `vignette("versions", "reticulate")`.

    After setting the environment, check if the installed versions of RLiesel and Liesel are compatible with `check_liesel_version()`.

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

    Installed Liesel version 0.2.10-dev is compatible, continuing to set up model

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
-22.073
</td>
<td>
242.590
</td>
<td>
-420.845
</td>
<td>
-22.455
</td>
<td>
391.772
</td>
<td>
4000
</td>
<td>
51.359
</td>
<td>
328.125
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
kernel_04
</td>
<td>
-1470.396
</td>
<td>
227.801
</td>
<td>
-1853.595
</td>
<td>
-1466.905
</td>
<td>
-1101.078
</td>
<td>
4000
</td>
<td>
227.695
</td>
<td>
567.559
</td>
<td>
1.016
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
-672.804
</td>
<td>
170.204
</td>
<td>
-954.507
</td>
<td>
-672.363
</td>
<td>
-394.065
</td>
<td>
4000
</td>
<td>
163.062
</td>
<td>
574.742
</td>
<td>
1.020
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
-578.864
</td>
<td>
111.028
</td>
<td>
-759.696
</td>
<td>
-578.867
</td>
<td>
-400.150
</td>
<td>
4000
</td>
<td>
230.468
</td>
<td>
629.516
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
kernel_04
</td>
<td>
1130.755
</td>
<td>
86.333
</td>
<td>
986.248
</td>
<td>
1132.358
</td>
<td>
1273.147
</td>
<td>
4000
</td>
<td>
372.603
</td>
<td>
727.101
</td>
<td>
1.004
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
-68.810
</td>
<td>
34.394
</td>
<td>
-126.301
</td>
<td>
-68.735
</td>
<td>
-11.545
</td>
<td>
4000
</td>
<td>
83.891
</td>
<td>
237.951
</td>
<td>
1.051
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
-213.497
</td>
<td>
20.291
</td>
<td>
-246.551
</td>
<td>
-213.732
</td>
<td>
-180.546
</td>
<td>
4000
</td>
<td>
271.528
</td>
<td>
468.387
</td>
<td>
1.021
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
114.058
</td>
<td>
65.370
</td>
<td>
16.265
</td>
<td>
110.296
</td>
<td>
228.006
</td>
<td>
4000
</td>
<td>
181.350
</td>
<td>
257.754
</td>
<td>
1.028
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
28.875
</td>
<td>
17.138
</td>
<td>
3.737
</td>
<td>
27.905
</td>
<td>
58.993
</td>
<td>
4000
</td>
<td>
172.313
</td>
<td>
276.003
</td>
<td>
1.024
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
747036.375
</td>
<td>
528680.062
</td>
<td>
270306.500
</td>
<td>
608441.594
</td>
<td>
1687256.681
</td>
<td>
4000
</td>
<td>
1748.916
</td>
<td>
2601.569
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
-25.543
</td>
<td>
1.652
</td>
<td>
-28.474
</td>
<td>
-25.445
</td>
<td>
-22.769
</td>
<td>
4000
</td>
<td>
10.236
</td>
<td>
38.237
</td>
<td>
1.322
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
6.445
</td>
<td>
8.440
</td>
<td>
-3.077
</td>
<td>
3.844
</td>
<td>
23.779
</td>
<td>
4000
</td>
<td>
18.906
</td>
<td>
150.106
</td>
<td>
1.159
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
-0.798
</td>
<td>
5.463
</td>
<td>
-9.721
</td>
<td>
-0.742
</td>
<td>
7.866
</td>
<td>
4000
</td>
<td>
51.066
</td>
<td>
85.230
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
kernel_01
</td>
<td>
-11.898
</td>
<td>
9.353
</td>
<td>
-30.580
</td>
<td>
-10.475
</td>
<td>
-0.123
</td>
<td>
4000
</td>
<td>
13.333
</td>
<td>
25.570
</td>
<td>
1.270
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
8.310
</td>
<td>
4.650
</td>
<td>
1.507
</td>
<td>
7.647
</td>
<td>
16.789
</td>
<td>
4000
</td>
<td>
26.348
</td>
<td>
120.082
</td>
<td>
1.144
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
-1.273
</td>
<td>
3.804
</td>
<td>
-8.567
</td>
<td>
-0.712
</td>
<td>
4.079
</td>
<td>
4000
</td>
<td>
24.836
</td>
<td>
70.130
</td>
<td>
1.128
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
3.199
</td>
<td>
1.715
</td>
<td>
0.500
</td>
<td>
3.162
</td>
<td>
6.098
</td>
<td>
4000
</td>
<td>
46.873
</td>
<td>
59.511
</td>
<td>
1.071
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
0.893
</td>
<td>
2.923
</td>
<td>
-4.867
</td>
<td>
1.280
</td>
<td>
4.816
</td>
<td>
4000
</td>
<td>
14.427
</td>
<td>
59.989
</td>
<td>
1.223
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
-1.347
</td>
<td>
3.302
</td>
<td>
-6.438
</td>
<td>
-1.574
</td>
<td>
4.416
</td>
<td>
4000
</td>
<td>
75.501
</td>
<td>
128.461
</td>
<td>
1.046
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
-0.414
</td>
<td>
1.831
</td>
<td>
-3.999
</td>
<td>
0.034
</td>
<td>
1.880
</td>
<td>
4000
</td>
<td>
12.663
</td>
<td>
39.420
</td>
<td>
1.263
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
86.308
</td>
<td>
135.919
</td>
<td>
6.482
</td>
<td>
41.332
</td>
<td>
308.744
</td>
<td>
4000
</td>
<td>
13.655
</td>
<td>
90.453
</td>
<td>
1.239
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
2.779
</td>
<td>
0.072
</td>
<td>
2.662
</td>
<td>
2.777
</td>
<td>
2.900
</td>
<td>
4000
</td>
<td>
44.596
</td>
<td>
608.667
</td>
<td>
1.065
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
15
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
2
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
18
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

    GraphBuilder(0 nodes, 1 vars)

``` python
_ = gb.transform(_vars["loc_np0_tau2"], tfb.Exp)
_ = gb.transform(_vars["scale_np0_tau2"], tfb.Exp)
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

builder.set_duration(warmup_duration=5000, posterior_duration=1000)

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
-58.968
</td>
<td>
244.904
</td>
<td>
-446.256
</td>
<td>
-63.482
</td>
<td>
350.972
</td>
<td>
4000
</td>
<td>
303.105
</td>
<td>
1106.189
</td>
<td>
1.010
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
-1452.305
</td>
<td>
240.587
</td>
<td>
-1854.228
</td>
<td>
-1445.285
</td>
<td>
-1062.321
</td>
<td>
4000
</td>
<td>
1237.965
</td>
<td>
1670.364
</td>
<td>
1.001
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
-679.038
</td>
<td>
174.487
</td>
<td>
-969.307
</td>
<td>
-678.779
</td>
<td>
-395.784
</td>
<td>
4000
</td>
<td>
880.418
</td>
<td>
1521.646
</td>
<td>
1.007
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
-565.712
</td>
<td>
111.768
</td>
<td>
-750.466
</td>
<td>
-567.981
</td>
<td>
-377.534
</td>
<td>
4000
</td>
<td>
1313.834
</td>
<td>
1494.371
</td>
<td>
1.003
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
1122.383
</td>
<td>
95.030
</td>
<td>
969.316
</td>
<td>
1120.986
</td>
<td>
1276.253
</td>
<td>
4000
</td>
<td>
1599.451
</td>
<td>
1671.529
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
kernel_04
</td>
<td>
-69.899
</td>
<td>
34.855
</td>
<td>
-124.631
</td>
<td>
-70.025
</td>
<td>
-13.924
</td>
<td>
4000
</td>
<td>
223.843
</td>
<td>
541.307
</td>
<td>
1.005
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
-210.811
</td>
<td>
22.367
</td>
<td>
-246.532
</td>
<td>
-211.490
</td>
<td>
-172.968
</td>
<td>
4000
</td>
<td>
1119.211
</td>
<td>
1752.065
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
kernel_04
</td>
<td>
113.849
</td>
<td>
72.297
</td>
<td>
9.920
</td>
<td>
106.721
</td>
<td>
244.043
</td>
<td>
4000
</td>
<td>
1017.327
</td>
<td>
849.913
</td>
<td>
1.003
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
29.644
</td>
<td>
19.091
</td>
<td>
2.267
</td>
<td>
27.771
</td>
<td>
63.949
</td>
<td>
4000
</td>
<td>
940.276
</td>
<td>
875.634
</td>
<td>
1.003
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
kernel_01
</td>
<td>
13.317
</td>
<td>
0.559
</td>
<td>
12.454
</td>
<td>
13.283
</td>
<td>
14.302
</td>
<td>
4000
</td>
<td>
1157.609
</td>
<td>
1434.248
</td>
<td>
1.005
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
-25.334
</td>
<td>
2.172
</td>
<td>
-28.720
</td>
<td>
-25.286
</td>
<td>
-21.895
</td>
<td>
4000
</td>
<td>
45.604
</td>
<td>
57.632
</td>
<td>
1.130
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
kernel_02
</td>
<td>
6.210
</td>
<td>
9.305
</td>
<td>
-5.335
</td>
<td>
4.210
</td>
<td>
24.434
</td>
<td>
4000
</td>
<td>
327.892
</td>
<td>
702.445
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
kernel_02
</td>
<td>
-1.433
</td>
<td>
6.164
</td>
<td>
-12.271
</td>
<td>
-1.055
</td>
<td>
7.977
</td>
<td>
4000
</td>
<td>
1002.500
</td>
<td>
938.664
</td>
<td>
1.003
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
-14.282
</td>
<td>
8.983
</td>
<td>
-30.375
</td>
<td>
-13.225
</td>
<td>
-1.864
</td>
<td>
4000
</td>
<td>
175.802
</td>
<td>
621.941
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
kernel_02
</td>
<td>
9.383
</td>
<td>
5.158
</td>
<td>
1.775
</td>
<td>
8.788
</td>
<td>
18.633
</td>
<td>
4000
</td>
<td>
260.234
</td>
<td>
1068.942
</td>
<td>
1.006
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
-1.532
</td>
<td>
3.911
</td>
<td>
-8.335
</td>
<td>
-1.210
</td>
<td>
4.324
</td>
<td>
4000
</td>
<td>
420.107
</td>
<td>
809.669
</td>
<td>
1.007
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
3.789
</td>
<td>
2.003
</td>
<td>
0.679
</td>
<td>
3.612
</td>
<td>
7.370
</td>
<td>
4000
</td>
<td>
311.801
</td>
<td>
1155.484
</td>
<td>
1.004
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
0.432
</td>
<td>
2.745
</td>
<td>
-4.628
</td>
<td>
0.886
</td>
<td>
4.188
</td>
<td>
4000
</td>
<td>
203.742
</td>
<td>
615.753
</td>
<td>
1.008
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
-0.531
</td>
<td>
3.899
</td>
<td>
-6.398
</td>
<td>
-0.800
</td>
<td>
6.485
</td>
<td>
4000
</td>
<td>
765.810
</td>
<td>
1120.062
</td>
<td>
1.004
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
-0.596
</td>
<td>
1.679
</td>
<td>
-3.793
</td>
<td>
-0.300
</td>
<td>
1.621
</td>
<td>
4000
</td>
<td>
221.605
</td>
<td>
614.913
</td>
<td>
1.009
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
kernel_00
</td>
<td>
4.016
</td>
<td>
1.157
</td>
<td>
2.158
</td>
<td>
4.046
</td>
<td>
5.854
</td>
<td>
4000
</td>
<td>
147.315
</td>
<td>
425.801
</td>
<td>
1.009
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
kernel_03
</td>
<td>
2.774
</td>
<td>
0.069
</td>
<td>
2.660
</td>
<td>
2.774
</td>
<td>
2.884
</td>
<td>
4000
</td>
<td>
475.142
</td>
<td>
1499.446
</td>
<td>
1.006
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
1
</th>
<th rowspan="2" valign="top">
divergent transition
</th>
<th>
warmup
</th>
<td>
83
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
<th rowspan="4" valign="top">
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
1441
</td>
<td>
0.072
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
2
</th>
<th rowspan="2" valign="top">
maximum tree depth
</th>
<th>
warmup
</th>
<td>
47
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
0.000
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
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
60
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
1036
</td>
<td>
0.052
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
1783
</td>
<td>
0.089
</td>
</tr>
<tr>
<th>
posterior
</th>
<td>
458
</td>
<td>
0.115
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
139
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
</tbody>
</table>

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
