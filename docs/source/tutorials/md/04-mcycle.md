
# Comparing samplers

In this tutorial, we are comparing two different sampling schemes on the
`mcycle` dataset with a Gaussian location-scale regression model and two
splines for the mean and the standard deviation. The `mcycle` dataset is
a “data frame giving a series of measurements of head acceleration in a
simulated motorcycle accident, used to test crash helmets” (from the
help page). It contains the following two variables:

-   `times`: in milliseconds after impact
-   `accel`: in g

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

![](04-mcycle_files/figure-gfm/model-1.png)

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
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 1, 1 / 75 transitions
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 0, 1, 1 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 0, 1, 0 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 1, 0, 0 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 0, 0, 1 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 0, 0 / 400 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 800 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 0, 1, 1, 0 / 800 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 3300 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 1, 1, 1, 1 / 50 transitions
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

<p><strong>Parameter summary:</strong></p>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>kernel</th>
      <th>mean</th>
      <th>sd</th>
      <th>q_0.05</th>
      <th>q_0.5</th>
      <th>q_0.95</th>
      <th>sample_size</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>rhat</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9" valign="top">loc_np0_beta_value</th>
      <th>(0,)</th>
      <td>kernel_05</td>
      <td>-63.629112</td>
      <td>252.206650</td>
      <td>-463.528070</td>
      <td>-66.371796</td>
      <td>3.548199e+02</td>
      <td>4000</td>
      <td>55.554466</td>
      <td>369.315973</td>
      <td>1.059267</td>
    </tr>
    <tr>
      <th>(1,)</th>
      <td>kernel_05</td>
      <td>-1427.637939</td>
      <td>241.812134</td>
      <td>-1818.575171</td>
      <td>-1431.717224</td>
      <td>-1.020957e+03</td>
      <td>4000</td>
      <td>247.110114</td>
      <td>459.375393</td>
      <td>1.032153</td>
    </tr>
    <tr>
      <th>(2,)</th>
      <td>kernel_05</td>
      <td>-678.088440</td>
      <td>171.642838</td>
      <td>-970.513010</td>
      <td>-677.299347</td>
      <td>-4.031863e+02</td>
      <td>4000</td>
      <td>329.391650</td>
      <td>591.602447</td>
      <td>1.020950</td>
    </tr>
    <tr>
      <th>(3,)</th>
      <td>kernel_05</td>
      <td>-558.946472</td>
      <td>112.474327</td>
      <td>-739.689679</td>
      <td>-561.785309</td>
      <td>-3.653995e+02</td>
      <td>4000</td>
      <td>284.318221</td>
      <td>816.784478</td>
      <td>1.024861</td>
    </tr>
    <tr>
      <th>(4,)</th>
      <td>kernel_05</td>
      <td>1120.222656</td>
      <td>96.278214</td>
      <td>959.109668</td>
      <td>1120.252625</td>
      <td>1.276266e+03</td>
      <td>4000</td>
      <td>354.719744</td>
      <td>786.903091</td>
      <td>1.002698</td>
    </tr>
    <tr>
      <th>(5,)</th>
      <td>kernel_05</td>
      <td>68.624733</td>
      <td>32.499458</td>
      <td>14.422557</td>
      <td>69.044384</td>
      <td>1.210155e+02</td>
      <td>4000</td>
      <td>35.079228</td>
      <td>211.084067</td>
      <td>1.096223</td>
    </tr>
    <tr>
      <th>(6,)</th>
      <td>kernel_05</td>
      <td>-212.407516</td>
      <td>21.061493</td>
      <td>-245.020116</td>
      <td>-213.184097</td>
      <td>-1.761234e+02</td>
      <td>4000</td>
      <td>224.904444</td>
      <td>637.413924</td>
      <td>1.018782</td>
    </tr>
    <tr>
      <th>(7,)</th>
      <td>kernel_05</td>
      <td>111.004074</td>
      <td>64.579948</td>
      <td>14.019436</td>
      <td>106.151539</td>
      <td>2.258563e+02</td>
      <td>4000</td>
      <td>283.805424</td>
      <td>417.992671</td>
      <td>1.010584</td>
    </tr>
    <tr>
      <th>(8,)</th>
      <td>kernel_05</td>
      <td>28.898365</td>
      <td>16.867601</td>
      <td>3.067752</td>
      <td>27.748809</td>
      <td>5.868952e+01</td>
      <td>4000</td>
      <td>171.205745</td>
      <td>400.405662</td>
      <td>1.013773</td>
    </tr>
    <tr>
      <th>loc_np0_tau2_value</th>
      <th>()</th>
      <td>kernel_04</td>
      <td>715131.875000</td>
      <td>516161.156250</td>
      <td>250225.231250</td>
      <td>579500.468750</td>
      <td>1.622960e+06</td>
      <td>4000</td>
      <td>1479.821028</td>
      <td>2708.382344</td>
      <td>1.002613</td>
    </tr>
    <tr>
      <th>loc_p0_beta_value</th>
      <th>(0,)</th>
      <td>kernel_03</td>
      <td>-25.058702</td>
      <td>2.586398</td>
      <td>-30.043613</td>
      <td>-24.729219</td>
      <td>-2.117909e+01</td>
      <td>4000</td>
      <td>5.853757</td>
      <td>20.201773</td>
      <td>1.902743</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">scale_np0_beta_value</th>
      <th>(0,)</th>
      <td>kernel_02</td>
      <td>7.274026</td>
      <td>9.129668</td>
      <td>-5.353710</td>
      <td>5.675588</td>
      <td>2.416816e+01</td>
      <td>4000</td>
      <td>47.053076</td>
      <td>95.497479</td>
      <td>1.090194</td>
    </tr>
    <tr>
      <th>(1,)</th>
      <td>kernel_02</td>
      <td>-1.605662</td>
      <td>5.956748</td>
      <td>-11.499222</td>
      <td>-1.535112</td>
      <td>7.905879e+00</td>
      <td>4000</td>
      <td>122.922426</td>
      <td>174.535745</td>
      <td>1.044571</td>
    </tr>
    <tr>
      <th>(2,)</th>
      <td>kernel_02</td>
      <td>-14.581491</td>
      <td>8.203374</td>
      <td>-29.592062</td>
      <td>-13.785058</td>
      <td>-2.526876e+00</td>
      <td>4000</td>
      <td>51.205890</td>
      <td>134.677775</td>
      <td>1.068350</td>
    </tr>
    <tr>
      <th>(3,)</th>
      <td>kernel_02</td>
      <td>10.022041</td>
      <td>4.357362</td>
      <td>3.260167</td>
      <td>9.896984</td>
      <td>1.764593e+01</td>
      <td>4000</td>
      <td>18.047480</td>
      <td>125.964990</td>
      <td>1.153383</td>
    </tr>
    <tr>
      <th>(4,)</th>
      <td>kernel_02</td>
      <td>-2.235376</td>
      <td>3.612199</td>
      <td>-8.282276</td>
      <td>-2.104898</td>
      <td>4.009915e+00</td>
      <td>4000</td>
      <td>45.109928</td>
      <td>84.760860</td>
      <td>1.064670</td>
    </tr>
    <tr>
      <th>(5,)</th>
      <td>kernel_02</td>
      <td>-3.808048</td>
      <td>1.786785</td>
      <td>-6.885901</td>
      <td>-3.753090</td>
      <td>-9.967382e-01</td>
      <td>4000</td>
      <td>15.521945</td>
      <td>106.268926</td>
      <td>1.180884</td>
    </tr>
    <tr>
      <th>(6,)</th>
      <td>kernel_02</td>
      <td>0.174443</td>
      <td>2.208008</td>
      <td>-3.785905</td>
      <td>0.423982</td>
      <td>3.505149e+00</td>
      <td>4000</td>
      <td>39.387396</td>
      <td>204.297989</td>
      <td>1.089678</td>
    </tr>
    <tr>
      <th>(7,)</th>
      <td>kernel_02</td>
      <td>-0.529909</td>
      <td>3.395279</td>
      <td>-5.863642</td>
      <td>-0.602151</td>
      <td>5.151755e+00</td>
      <td>4000</td>
      <td>21.186439</td>
      <td>78.199690</td>
      <td>1.137757</td>
    </tr>
    <tr>
      <th>(8,)</th>
      <td>kernel_02</td>
      <td>-0.783153</td>
      <td>1.460789</td>
      <td>-3.551109</td>
      <td>-0.587341</td>
      <td>1.255919e+00</td>
      <td>4000</td>
      <td>41.102703</td>
      <td>111.783221</td>
      <td>1.086979</td>
    </tr>
    <tr>
      <th>scale_np0_tau2_value</th>
      <th>()</th>
      <td>kernel_01</td>
      <td>102.527267</td>
      <td>115.529694</td>
      <td>12.401674</td>
      <td>64.901726</td>
      <td>3.057285e+02</td>
      <td>4000</td>
      <td>37.918925</td>
      <td>169.881796</td>
      <td>1.094041</td>
    </tr>
    <tr>
      <th>scale_p0_beta_value</th>
      <th>(0,)</th>
      <td>kernel_00</td>
      <td>2.773070</td>
      <td>0.068457</td>
      <td>2.660991</td>
      <td>2.771538</td>
      <td>2.886909e+00</td>
      <td>4000</td>
      <td>202.928262</td>
      <td>955.105286</td>
      <td>1.018778</td>
    </tr>
  </tbody>
</table>
<p><strong>Error summary:</strong></p>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>count</th>
      <th>relative</th>
    </tr>
    <tr>
      <th>kernel</th>
      <th>error_code</th>
      <th>error_msg</th>
      <th>phase</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">kernel_00</th>
      <th rowspan="2" valign="top">90</th>
      <th rowspan="2" valign="top">nan acceptance prob</th>
      <th>warmup</th>
      <td>4</td>
      <td>0.0002</td>
    </tr>
    <tr>
      <th>posterior</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">kernel_02</th>
      <th rowspan="2" valign="top">90</th>
      <th rowspan="2" valign="top">nan acceptance prob</th>
      <th>warmup</th>
      <td>14</td>
      <td>0.0007</td>
    </tr>
    <tr>
      <th>posterior</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>

``` python
fig = gs.plot_trace(results, "loc_p0_beta_value")
```

<img src="04-mcycle_files/figure-gfm/iwls-traces-1.png" width="384" />

``` python
fig = gs.plot_trace(results, "loc_np0_tau2_value")
```

<img src="04-mcycle_files/figure-gfm/iwls-traces-2.png" width="384" />

``` python
fig = gs.plot_trace(results, "loc_np0_beta_value")
```

<img src="04-mcycle_files/figure-gfm/iwls-traces-3.png" width="960" />

``` python
fig = gs.plot_trace(results, "scale_p0_beta_value")
```

<img src="04-mcycle_files/figure-gfm/iwls-traces-4.png" width="384" />

``` python
fig = gs.plot_trace(results, "scale_np0_tau2_value")
```

<img src="04-mcycle_files/figure-gfm/iwls-traces-5.png" width="384" />

``` python
fig = gs.plot_trace(results, "scale_np0_beta_value")
```

<img src="04-mcycle_files/figure-gfm/iwls-traces-6.png" width="960" />

To confirm that the chains have converged to reasonable values, here is
a plot of the estimated mean function:

``` python
summary = gs.Summary(results).to_dataframe().reset_index()
```

``` r
library(dplyr)
```


    Attache Paket: 'dplyr'

    Das folgende Objekt ist maskiert 'package:MASS':

        select

    Die folgenden Objekte sind maskiert von 'package:stats':

        filter, lag

    Die folgenden Objekte sind maskiert von 'package:base':

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

    Warning: Ignoring unknown parameters: linewidth

![](04-mcycle_files/figure-gfm/iwls-spline-13.png)

## NUTS sampler

As an alternative, we try a NUTS kernel which samples all model
parameters (regression coefficients and smoothing parameters) in one
block. To do so, we first need to log-transform the smoothing
parameters. This is the model graph before the transformation:

``` python
lsl.plot_vars(model)
```

<img src="04-mcycle_files/figure-gfm/untransformed-graph-1.png"
width="1344" />

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

<img src="04-mcycle_files/figure-gfm/transformed-graph-3.png"
width="1344" />

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
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 42, 59, 50, 61 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 16, 13, 14, 8 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 34, 23, 45, 7 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 92, 87, 91, 86 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 183, 189, 175, 162 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 400 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 367, 352, 352, 360 / 400 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 800 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 692, 708, 704, 686 / 800 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 3300 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2951, 2981, 2978, 2946 / 3300 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 46, 47, 40, 48 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 987, 981, 939, 984 / 1000 transitions
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
    loc_np0_beta               (0,)   kernel_00     0.828418  ...    10.890932  2.167341
                               (1,)   kernel_00 -1396.033447  ...   293.549909  1.027742
                               (2,)   kernel_00  -653.058105  ...  1012.348261  1.036174
                               (3,)   kernel_00  -562.518433  ...  1331.902456  1.002152
                               (4,)   kernel_00  1110.601685  ...  2635.858942  1.003348
                               (5,)   kernel_00    71.067184  ...  1334.942299  1.008585
                               (6,)   kernel_00  -213.209106  ...  1994.046348  1.003797
                               (7,)   kernel_00   101.714714  ...   324.197245  1.024877
                               (8,)   kernel_00    26.070953  ...   356.516000  1.031495
    loc_np0_tau2_transformed   ()     kernel_00    13.274561  ...  1397.175101  1.004933
    loc_p0_beta                (0,)   kernel_00   -25.242102  ...   318.067798  1.020105
    scale_np0_beta             (0,)   kernel_00     6.426808  ...   970.593206  1.005669
                               (1,)   kernel_00    -1.524924  ...  1126.286180  1.005954
                               (2,)   kernel_00   -14.492555  ...   795.309203  1.011106
                               (3,)   kernel_00     9.366125  ...   959.538796  1.007047
                               (4,)   kernel_00    -1.528407  ...  1016.189974  1.002879
                               (5,)   kernel_00    -3.734825  ...  1079.376296  1.003831
                               (6,)   kernel_00     0.388581  ...   612.276170  1.008305
                               (7,)   kernel_00    -0.648466  ...  1058.549915  1.002327
                               (8,)   kernel_00    -0.645470  ...   626.463819  1.007817
    scale_np0_tau2_transformed ()     kernel_00     4.082952  ...  1515.335334  1.011839
    scale_p0_beta              (0,)   kernel_00     2.769565  ...  2374.286033  1.006084

    [22 rows x 10 columns]

    Error summary:

                                                                              count  relative
    kernel    error_code error_msg                                 phase
    kernel_00 1          divergent transition                      warmup      1383   0.06915
                                                                   posterior      7   0.00175
              2          maximum tree depth                        warmup     15865   0.79325
                                                                   posterior   3881   0.97025
              3          divergent transition + maximum tree depth warmup       447   0.02235
                                                                   posterior      3   0.00075

``` python
fig = gs.plot_trace(results, "loc_p0_beta")
```

<img src="04-mcycle_files/figure-gfm/nuts-traces-5.png" width="384" />

``` python
fig = gs.plot_trace(results, "loc_np0_tau2_transformed")
```

<img src="04-mcycle_files/figure-gfm/nuts-traces-6.png" width="384" />

``` python
fig = gs.plot_trace(results, "loc_np0_beta")
```

<img src="04-mcycle_files/figure-gfm/nuts-traces-7.png" width="960" />

``` python
fig = gs.plot_trace(results, "scale_p0_beta")
```

<img src="04-mcycle_files/figure-gfm/nuts-traces-8.png" width="384" />

``` python
fig = gs.plot_trace(results, "scale_np0_tau2_transformed")
```

<img src="04-mcycle_files/figure-gfm/nuts-traces-9.png" width="384" />

``` python
fig = gs.plot_trace(results, "scale_np0_beta")
```

<img src="04-mcycle_files/figure-gfm/nuts-traces-10.png" width="960" />

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

    Warning: Ignoring unknown parameters: linewidth

![](04-mcycle_files/figure-gfm/nuts-spline-17.png)
