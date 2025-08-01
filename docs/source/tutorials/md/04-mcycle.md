

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

-94.349
</td>

<td>

243.847
</td>

<td>

-488.775
</td>

<td>

-95.656
</td>

<td>

311.129
</td>

<td>

4000
</td>

<td>

30.923
</td>

<td>

282.879
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

kernel_04
</td>

<td>

-1453.466
</td>

<td>

252.842
</td>

<td>

-1878.822
</td>

<td>

-1446.134
</td>

<td>

-1053.006
</td>

<td>

4000
</td>

<td>

246.578
</td>

<td>

510.540
</td>

<td>

1.023
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

-686.443
</td>

<td>

169.833
</td>

<td>

-960.594
</td>

<td>

-684.660
</td>

<td>

-405.767
</td>

<td>

4000
</td>

<td>

223.260
</td>

<td>

481.583
</td>

<td>

1.018
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

-567.134
</td>

<td>

115.355
</td>

<td>

-749.654
</td>

<td>

-571.310
</td>

<td>

-374.490
</td>

<td>

4000
</td>

<td>

167.944
</td>

<td>

470.736
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

kernel_04
</td>

<td>

1116.798
</td>

<td>

95.052
</td>

<td>

962.106
</td>

<td>

1114.293
</td>

<td>

1270.780
</td>

<td>

4000
</td>

<td>

271.148
</td>

<td>

693.212
</td>

<td>

1.016
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

-63.816
</td>

<td>

34.940
</td>

<td>

-122.269
</td>

<td>

-62.284
</td>

<td>

-7.426
</td>

<td>

4000
</td>

<td>

77.993
</td>

<td>

269.120
</td>

<td>

1.052
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

-210.732
</td>

<td>

21.238
</td>

<td>

-245.768
</td>

<td>

-210.927
</td>

<td>

-174.976
</td>

<td>

4000
</td>

<td>

103.363
</td>

<td>

358.025
</td>

<td>

1.043
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

116.662
</td>

<td>

70.087
</td>

<td>

16.241
</td>

<td>

109.447
</td>

<td>

243.632
</td>

<td>

4000
</td>

<td>

147.127
</td>

<td>

252.830
</td>

<td>

1.009
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

30.363
</td>

<td>

18.146
</td>

<td>

4.304
</td>

<td>

28.137
</td>

<td>

62.882
</td>

<td>

4000
</td>

<td>

127.388
</td>

<td>

251.364
</td>

<td>

1.018
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

737823.625
</td>

<td>

578259.562
</td>

<td>

253337.921
</td>

<td>

588002.312
</td>

<td>

1716808.669
</td>

<td>

4000
</td>

<td>

1587.651
</td>

<td>

2275.345
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

-24.765
</td>

<td>

2.553
</td>

<td>

-28.572
</td>

<td>

-24.883
</td>

<td>

-20.010
</td>

<td>

4000
</td>

<td>

7.843
</td>

<td>

14.328
</td>

<td>

1.489
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

6.961
</td>

<td>

9.337
</td>

<td>

-5.408
</td>

<td>

5.269
</td>

<td>

23.987
</td>

<td>

4000
</td>

<td>

17.877
</td>

<td>

120.385
</td>

<td>

1.157
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

-2.036
</td>

<td>

6.480
</td>

<td>

-13.667
</td>

<td>

-1.482
</td>

<td>

8.049
</td>

<td>

4000
</td>

<td>

22.954
</td>

<td>

51.514
</td>

<td>

1.130
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

-16.658
</td>

<td>

9.988
</td>

<td>

-33.657
</td>

<td>

-16.499
</td>

<td>

-1.449
</td>

<td>

4000
</td>

<td>

14.119
</td>

<td>

42.907
</td>

<td>

1.229
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

9.618
</td>

<td>

4.658
</td>

<td>

2.838
</td>

<td>

9.276
</td>

<td>

18.055
</td>

<td>

4000
</td>

<td>

14.480
</td>

<td>

135.546
</td>

<td>

1.226
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

-2.146
</td>

<td>

3.917
</td>

<td>

-9.329
</td>

<td>

-1.678
</td>

<td>

3.532
</td>

<td>

4000
</td>

<td>

29.007
</td>

<td>

60.175
</td>

<td>

1.100
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

3.789
</td>

<td>

2.042
</td>

<td>

0.597
</td>

<td>

3.632
</td>

<td>

7.119
</td>

<td>

4000
</td>

<td>

19.686
</td>

<td>

38.758
</td>

<td>

1.165
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

-0.128
</td>

<td>

2.958
</td>

<td>

-5.473
</td>

<td>

0.398
</td>

<td>

4.005
</td>

<td>

4000
</td>

<td>

13.462
</td>

<td>

45.945
</td>

<td>

1.210
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

-0.780
</td>

<td>

3.411
</td>

<td>

-5.942
</td>

<td>

-1.042
</td>

<td>

5.018
</td>

<td>

4000
</td>

<td>

49.047
</td>

<td>

118.980
</td>

<td>

1.076
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

-0.976
</td>

<td>

1.772
</td>

<td>

-4.167
</td>

<td>

-0.686
</td>

<td>

1.475
</td>

<td>

4000
</td>

<td>

15.308
</td>

<td>

85.463
</td>

<td>

1.185
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

123.742
</td>

<td>

170.641
</td>

<td>

8.036
</td>

<td>

74.600
</td>

<td>

391.613
</td>

<td>

4000
</td>

<td>

15.272
</td>

<td>

97.255
</td>

<td>

1.199
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

2.663
</td>

<td>

2.772
</td>

<td>

2.891
</td>

<td>

4000
</td>

<td>

277.278
</td>

<td>

673.429
</td>

<td>

1.032
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

10
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

Note, RLiesel automatically populates the parameter variables with
intormation how to conduct the MCMC inference. To trasform the
variables, we can drop it. We setup the MCMC sampler in the next chunk.

After transformation, there are two additional nodes in the new model
graph.

``` python
import tensorflow_probability.substrates.jax.bijectors as tfb

nodes, _vars = model.pop_nodes_and_vars()

_vars["loc_np0_tau2"].transform(tfb.Exp(), inference='drop')
```

    Var(name="loc_np0_tau2_transformed")

``` python
_vars["scale_np0_tau2"].transform(tfb.Exp(), inference='drop')
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

-78.504
</td>

<td>

250.892
</td>

<td>

-492.913
</td>

<td>

-87.434
</td>

<td>

351.543
</td>

<td>

4000
</td>

<td>

484.665
</td>

<td>

1063.631
</td>

<td>

1.005
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

-1455.964
</td>

<td>

249.587
</td>

<td>

-1875.602
</td>

<td>

-1452.794
</td>

<td>

-1049.933
</td>

<td>

4000
</td>

<td>

1061.125
</td>

<td>

1909.782
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

kernel_03
</td>

<td>

-679.539
</td>

<td>

178.596
</td>

<td>

-973.216
</td>

<td>

-673.895
</td>

<td>

-391.914
</td>

<td>

4000
</td>

<td>

889.420
</td>

<td>

1856.702
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

kernel_03
</td>

<td>

-559.658
</td>

<td>

112.455
</td>

<td>

-745.106
</td>

<td>

-561.955
</td>

<td>

-380.485
</td>

<td>

4000
</td>

<td>

1117.456
</td>

<td>

2049.015
</td>

<td>

1.001
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

1125.364
</td>

<td>

95.436
</td>

<td>

976.197
</td>

<td>

1123.721
</td>

<td>

1289.182
</td>

<td>

4000
</td>

<td>

1107.656
</td>

<td>

1756.276
</td>

<td>

1.005
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

-68.012
</td>

<td>

34.946
</td>

<td>

-124.186
</td>

<td>

-68.340
</td>

<td>

-9.189
</td>

<td>

4000
</td>

<td>

136.228
</td>

<td>

1150.207
</td>

<td>

1.030
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

-210.810
</td>

<td>

22.757
</td>

<td>

-246.558
</td>

<td>

-211.805
</td>

<td>

-172.949
</td>

<td>

4000
</td>

<td>

926.620
</td>

<td>

1525.961
</td>

<td>

1.007
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

119.202
</td>

<td>

76.890
</td>

<td>

11.201
</td>

<td>

111.466
</td>

<td>

254.942
</td>

<td>

4000
</td>

<td>

865.692
</td>

<td>

755.492
</td>

<td>

1.007
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

31.107
</td>

<td>

20.130
</td>

<td>

2.605
</td>

<td>

28.867
</td>

<td>

65.898
</td>

<td>

4000
</td>

<td>

801.140
</td>

<td>

743.522
</td>

<td>

1.008
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

13.319
</td>

<td>

0.568
</td>

<td>

12.446
</td>

<td>

13.276
</td>

<td>

14.321
</td>

<td>

4000
</td>

<td>

1206.455
</td>

<td>

1563.516
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

-25.163
</td>

<td>

1.971
</td>

<td>

-28.495
</td>

<td>

-25.061
</td>

<td>

-22.075
</td>

<td>

4000
</td>

<td>

12.187
</td>

<td>

55.473
</td>

<td>

1.264
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

6.750
</td>

<td>

9.518
</td>

<td>

-5.564
</td>

<td>

4.897
</td>

<td>

25.307
</td>

<td>

4000
</td>

<td>

495.323
</td>

<td>

745.611
</td>

<td>

1.005
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

-1.649
</td>

<td>

6.359
</td>

<td>

-12.758
</td>

<td>

-1.107
</td>

<td>

8.198
</td>

<td>

4000
</td>

<td>

953.466
</td>

<td>

1354.547
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

kernel_00
</td>

<td>

-14.995
</td>

<td>

8.873
</td>

<td>

-30.431
</td>

<td>

-14.222
</td>

<td>

-2.026
</td>

<td>

4000
</td>

<td>

283.189
</td>

<td>

1119.078
</td>

<td>

1.011
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

9.985
</td>

<td>

5.052
</td>

<td>

2.358
</td>

<td>

9.681
</td>

<td>

18.773
</td>

<td>

4000
</td>

<td>

361.367
</td>

<td>

1121.055
</td>

<td>

1.011
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

-1.727
</td>

<td>

3.986
</td>

<td>

-8.544
</td>

<td>

-1.443
</td>

<td>

4.405
</td>

<td>

4000
</td>

<td>

476.417
</td>

<td>

1216.115
</td>

<td>

1.008
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

3.996
</td>

<td>

1.973
</td>

<td>

1.034
</td>

<td>

3.844
</td>

<td>

7.469
</td>

<td>

4000
</td>

<td>

427.897
</td>

<td>

1012.064
</td>

<td>

1.008
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

0.230
</td>

<td>

2.758
</td>

<td>

-4.770
</td>

<td>

0.634
</td>

<td>

4.115
</td>

<td>

4000
</td>

<td>

292.947
</td>

<td>

1104.930
</td>

<td>

1.011
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

-0.147
</td>

<td>

3.990
</td>

<td>

-6.172
</td>

<td>

-0.500
</td>

<td>

6.775
</td>

<td>

4000
</td>

<td>

633.632
</td>

<td>

964.743
</td>

<td>

1.007
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

-0.687
</td>

<td>

1.695
</td>

<td>

-3.771
</td>

<td>

-0.423
</td>

<td>

1.658
</td>

<td>

4000
</td>

<td>

316.789
</td>

<td>

945.565
</td>

<td>

1.008
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

4.133
</td>

<td>

1.074
</td>

<td>

2.297
</td>

<td>

4.175
</td>

<td>

5.815
</td>

<td>

4000
</td>

<td>

224.795
</td>

<td>

518.098
</td>

<td>

1.013
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

2.771
</td>

<td>

0.068
</td>

<td>

2.663
</td>

<td>

2.768
</td>

<td>

2.887
</td>

<td>

4000
</td>

<td>

802.467
</td>

<td>

1824.805
</td>

<td>

1.002
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

1686
</td>

<td>

0.084
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

0.001
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

80
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

77
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

1184
</td>

<td>

0.059
</td>

</tr>

<tr>

<th>

posterior
</th>

<td>

118
</td>

<td>

0.030
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

3450
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

99
</td>

<td>

0.025
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

106
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

134
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
