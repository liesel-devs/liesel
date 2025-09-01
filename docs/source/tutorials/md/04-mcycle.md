

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

-91.390
</td>

<td>

247.682
</td>

<td>

-488.397
</td>

<td>

-90.347
</td>

<td>

315.251
</td>

<td>

4000
</td>

<td>

28.217
</td>

<td>

264.417
</td>

<td>

1.104
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

-1455.262
</td>

<td>

253.239
</td>

<td>

-1894.081
</td>

<td>

-1441.528
</td>

<td>

-1056.263
</td>

<td>

4000
</td>

<td>

241.989
</td>

<td>

521.924
</td>

<td>

1.021
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

-683.301
</td>

<td>

169.958
</td>

<td>

-958.177
</td>

<td>

-682.715
</td>

<td>

-402.217
</td>

<td>

4000
</td>

<td>

192.407
</td>

<td>

468.798
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

-566.658
</td>

<td>

114.645
</td>

<td>

-749.965
</td>

<td>

-568.552
</td>

<td>

-378.175
</td>

<td>

4000
</td>

<td>

189.440
</td>

<td>

484.421
</td>

<td>

1.020
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

1116.650
</td>

<td>

95.214
</td>

<td>

963.945
</td>

<td>

1115.443
</td>

<td>

1271.991
</td>

<td>

4000
</td>

<td>

271.909
</td>

<td>

667.411
</td>

<td>

1.017
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

-65.228
</td>

<td>

34.067
</td>

<td>

-121.626
</td>

<td>

-63.814
</td>

<td>

-10.395
</td>

<td>

4000
</td>

<td>

114.652
</td>

<td>

261.656
</td>

<td>

1.038
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

-210.149
</td>

<td>

21.589
</td>

<td>

-245.430
</td>

<td>

-210.597
</td>

<td>

-173.556
</td>

<td>

4000
</td>

<td>

100.835
</td>

<td>

387.063
</td>

<td>

1.042
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

116.922
</td>

<td>

70.639
</td>

<td>

15.437
</td>

<td>

108.853
</td>

<td>

241.107
</td>

<td>

4000
</td>

<td>

139.555
</td>

<td>

291.663
</td>

<td>

1.014
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

30.563
</td>

<td>

18.338
</td>

<td>

4.360
</td>

<td>

28.062
</td>

<td>

62.986
</td>

<td>

4000
</td>

<td>

123.637
</td>

<td>

221.055
</td>

<td>

1.020
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

737984.250
</td>

<td>

577874.438
</td>

<td>

253233.323
</td>

<td>

589114.375
</td>

<td>

1722400.306
</td>

<td>

4000
</td>

<td>

1499.894
</td>

<td>

2375.246
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

-24.969
</td>

<td>

2.376
</td>

<td>

-28.642
</td>

<td>

-25.026
</td>

<td>

-20.779
</td>

<td>

4000
</td>

<td>

8.821
</td>

<td>

15.300
</td>

<td>

1.410
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

7.025
</td>

<td>

9.321
</td>

<td>

-5.289
</td>

<td>

5.091
</td>

<td>

24.275
</td>

<td>

4000
</td>

<td>

15.646
</td>

<td>

118.432
</td>

<td>

1.188
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

-1.923
</td>

<td>

6.277
</td>

<td>

-13.178
</td>

<td>

-1.462
</td>

<td>

7.818
</td>

<td>

4000
</td>

<td>

27.622
</td>

<td>

47.323
</td>

<td>

1.110
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

-16.511
</td>

<td>

10.157
</td>

<td>

-33.974
</td>

<td>

-15.995
</td>

<td>

-1.650
</td>

<td>

4000
</td>

<td>

11.677
</td>

<td>

54.664
</td>

<td>

1.292
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

9.611
</td>

<td>

4.704
</td>

<td>

2.787
</td>

<td>

9.231
</td>

<td>

18.011
</td>

<td>

4000
</td>

<td>

14.597
</td>

<td>

135.661
</td>

<td>

1.240
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

-2.125
</td>

<td>

4.119
</td>

<td>

-9.945
</td>

<td>

-1.643
</td>

<td>

3.839
</td>

<td>

4000
</td>

<td>

20.483
</td>

<td>

45.254
</td>

<td>

1.139
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

3.791
</td>

<td>

2.055
</td>

<td>

0.612
</td>

<td>

3.641
</td>

<td>

7.185
</td>

<td>

4000
</td>

<td>

21.548
</td>

<td>

43.031
</td>

<td>

1.152
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

3.070
</td>

<td>

-5.696
</td>

<td>

0.368
</td>

<td>

3.991
</td>

<td>

4000
</td>

<td>

10.707
</td>

<td>

43.260
</td>

<td>

1.284
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

-0.781
</td>

<td>

3.535
</td>

<td>

-6.027
</td>

<td>

-0.995
</td>

<td>

5.362
</td>

<td>

4000
</td>

<td>

65.318
</td>

<td>

114.175
</td>

<td>

1.059
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

-0.969
</td>

<td>

1.866
</td>

<td>

-4.425
</td>

<td>

-0.607
</td>

<td>

1.492
</td>

<td>

4000
</td>

<td>

11.778
</td>

<td>

56.939
</td>

<td>

1.262
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

123.806
</td>

<td>

175.328
</td>

<td>

8.288
</td>

<td>

71.644
</td>

<td>

401.147
</td>

<td>

4000
</td>

<td>

12.703
</td>

<td>

89.230
</td>

<td>

1.252
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

300.938
</td>

<td>

662.908
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

-63.229
</td>

<td>

243.324
</td>

<td>

-462.330
</td>

<td>

-68.467
</td>

<td>

347.418
</td>

<td>

4000
</td>

<td>

572.255
</td>

<td>

1508.810
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

kernel_03
</td>

<td>

-1459.893
</td>

<td>

249.440
</td>

<td>

-1880.648
</td>

<td>

-1461.151
</td>

<td>

-1042.745
</td>

<td>

4000
</td>

<td>

930.528
</td>

<td>

1700.929
</td>

<td>

1.002
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

-670.265
</td>

<td>

173.929
</td>

<td>

-948.845
</td>

<td>

-668.166
</td>

<td>

-389.307
</td>

<td>

4000
</td>

<td>

888.598
</td>

<td>

2170.708
</td>

<td>

1.002
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

-563.162
</td>

<td>

112.328
</td>

<td>

-742.858
</td>

<td>

-563.505
</td>

<td>

-381.376
</td>

<td>

4000
</td>

<td>

1341.029
</td>

<td>

2065.442
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

1120.906
</td>

<td>

93.639
</td>

<td>

969.670
</td>

<td>

1121.878
</td>

<td>

1274.684
</td>

<td>

4000
</td>

<td>

1415.871
</td>

<td>

1956.833
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

-71.256
</td>

<td>

33.912
</td>

<td>

-126.670
</td>

<td>

-71.497
</td>

<td>

-15.347
</td>

<td>

4000
</td>

<td>

300.396
</td>

<td>

936.001
</td>

<td>

1.012
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

-210.217
</td>

<td>

21.841
</td>

<td>

-244.542
</td>

<td>

-210.651
</td>

<td>

-173.495
</td>

<td>

4000
</td>

<td>

1002.059
</td>

<td>

1178.915
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

kernel_03
</td>

<td>

116.723
</td>

<td>

72.734
</td>

<td>

13.922
</td>

<td>

108.308
</td>

<td>

244.842
</td>

<td>

4000
</td>

<td>

998.296
</td>

<td>

992.638
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

kernel_03
</td>

<td>

30.491
</td>

<td>

18.964
</td>

<td>

3.143
</td>

<td>

28.584
</td>

<td>

63.034
</td>

<td>

4000
</td>

<td>

922.966
</td>

<td>

911.823
</td>

<td>

1.006
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

13.315
</td>

<td>

0.563
</td>

<td>

12.438
</td>

<td>

13.286
</td>

<td>

14.263
</td>

<td>

4000
</td>

<td>

1249.361
</td>

<td>

1825.657
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

-25.699
</td>

<td>

2.164
</td>

<td>

-29.375
</td>

<td>

-25.693
</td>

<td>

-22.333
</td>

<td>

4000
</td>

<td>

45.077
</td>

<td>

52.164
</td>

<td>

1.092
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

6.927
</td>

<td>

9.646
</td>

<td>

-5.290
</td>

<td>

4.681
</td>

<td>

26.189
</td>

<td>

4000
</td>

<td>

371.523
</td>

<td>

717.938
</td>

<td>

1.007
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

-2.126
</td>

<td>

6.575
</td>

<td>

-14.176
</td>

<td>

-1.464
</td>

<td>

7.550
</td>

<td>

4000
</td>

<td>

570.862
</td>

<td>

890.810
</td>

<td>

1.009
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

-14.821
</td>

<td>

9.122
</td>

<td>

-31.409
</td>

<td>

-13.890
</td>

<td>

-1.727
</td>

<td>

4000
</td>

<td>

207.251
</td>

<td>

866.569
</td>

<td>

1.014
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

9.645
</td>

<td>

5.051
</td>

<td>

2.141
</td>

<td>

9.249
</td>

<td>

18.771
</td>

<td>

4000
</td>

<td>

293.530
</td>

<td>

1210.692
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

-1.575
</td>

<td>

3.789
</td>

<td>

-8.151
</td>

<td>

-1.384
</td>

<td>

4.208
</td>

<td>

4000
</td>

<td>

444.559
</td>

<td>

1282.576
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

kernel_00
</td>

<td>

3.915
</td>

<td>

1.961
</td>

<td>

0.907
</td>

<td>

3.788
</td>

<td>

7.302
</td>

<td>

4000
</td>

<td>

392.881
</td>

<td>

1129.720
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

kernel_00
</td>

<td>

0.244
</td>

<td>

2.758
</td>

<td>

-4.679
</td>

<td>

0.623
</td>

<td>

4.009
</td>

<td>

4000
</td>

<td>

215.670
</td>

<td>

803.357
</td>

<td>

1.012
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

-0.262
</td>

<td>

3.858
</td>

<td>

-6.208
</td>

<td>

-0.585
</td>

<td>

6.385
</td>

<td>

4000
</td>

<td>

773.996
</td>

<td>

1264.366
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

kernel_00
</td>

<td>

-0.676
</td>

<td>

1.671
</td>

<td>

-3.842
</td>

<td>

-0.396
</td>

<td>

1.544
</td>

<td>

4000
</td>

<td>

222.014
</td>

<td>

642.370
</td>

<td>

1.012
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

4.084
</td>

<td>

1.146
</td>

<td>

2.136
</td>

<td>

4.115
</td>

<td>

5.928
</td>

<td>

4000
</td>

<td>

171.779
</td>

<td>

546.093
</td>

<td>

1.018
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

2.774
</td>

<td>

0.070
</td>

<td>

2.663
</td>

<td>

2.773
</td>

<td>

2.892
</td>

<td>

4000
</td>

<td>

651.244
</td>

<td>

1699.064
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

1690
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

2
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

74
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

122
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

1207
</td>

<td>

0.060
</td>

</tr>

<tr>

<th>

posterior
</th>

<td>

86
</td>

<td>

0.021
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

3210
</td>

<td>

0.161
</td>

</tr>

<tr>

<th>

posterior
</th>

<td>

104
</td>

<td>

0.026
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

110
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

141
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
