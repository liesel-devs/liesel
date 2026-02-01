

# Linear Regression

In this tutorial, we build a linear regression model with Liesel and
estimate it with Goose. Our goal is to illustrate the most fundamental
features of the software in a straight-forward context.

## Imports

Before we can generate the data and build the model, we need to load
Liesel and a number of other packages. We usually import the model
building library `liesel.model` as `lsl`, and the MCMC library
`liesel.goose` as `gs`.

``` python
import jax
import jax.numpy as jnp
import numpy as np

# We use distributions and bijectors from tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

import liesel.goose as gs
import liesel.model as lsl

import matplotlib.pyplot as plt
```

## Generating the data

Now we can simulate 500 observations from the linear regression model
$y_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i, \;\sigma^2)$ with the true
parameters $\boldsymbol{\beta} = (\beta_0, \beta_1)' = (1, 2)'$ and
$\sigma = 1$. The relationship between the response $y_i$ and the
covariate $x_i$ is visualized in the following scatterplot.

``` python
rng = np.random.default_rng(42)

# sample size and true parameters
n = 500
true_beta = np.array([1.0, 2.0])
true_sigma = 1.0

# data-generating process
x0 = rng.uniform(size=n)
X_mat = np.column_stack([np.ones(n), x0])
eps = rng.normal(scale=true_sigma, size=n)
y_vec = X_mat @ true_beta + eps

# plot the simulated data
plt.scatter(x0, y_vec)
plt.title("Simulated data from the linear regression model")
plt.xlabel("Covariate x")
plt.ylabel("Response y")
plt.show()
```

![](01a-lin-reg_files/figure-commonmark/generate-data-1.png)

## Building the Model

As the most basic building blocks of a model, Liesel provides the
{class}`.Var` class for instantiating variables and the {class}`.Dist`
class for wrapping probability distributions. The {class}`.Var` class
comes with four constructors, namely {meth}`.Var.new_param` for
parameters, {meth}`.Var.new_obs` for observed data,
{meth}`.Var.new_calc` for variables that are deterministic functions of
other variables in the model, and {meth}`.Var.new_value` for fixed
values.

### The regression coefficients

Let’s assume the weakly informative prior
$\beta_0, \beta_1 \sim \mathcal{N}(0, 100^2)$ for the regression
coefficients. To define this in Liesel, we will be using the
{class}`.Dist` class. This class wraps distribution classes with the
TensorFlow Probability (TFP) API. Here, we use the TFP distribution
object
[(`tfd.Normal`)](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal),
and the two hyperparameters representing the parameters of the
distribution. TFP uses the names `loc` for the mean and `scale` for the
standard deviation, so we have to use the same names here. This is a
general feature of {class}`.Dist`, you should always use the parameter
names from TFP to refer to the parameters of your distribution.

``` python
beta_prior = lsl.Dist(tfd.Normal, loc=0.0, scale=100.0)
```

Now we can create our regression coefficient with the
{meth}`.Var.new_param` constructor:

``` python
beta = lsl.Var.new_param(
    value=jnp.array([0.0, 0.0]), distribution=beta_prior, name="beta"
)
```

### The standard deviation

We define the standard deviation using the weakly informative prior
$\sigma^2 \sim \text{InverseGamma}(a, b)$ with $a = b = 0.01$.

``` python
sigma_sq_prior = lsl.Dist(tfd.InverseGamma, concentration=0.01, scale=0.01)
sigma_sq = lsl.Var.new_param(value=1.0, distribution=sigma_sq_prior, name="sigma_sq")
```

Since we need to work not only with the variance, but with the scale, we
initialize the scale using {meth}`.Var.new_calc`, to compute the square
root.

``` python
sigma = lsl.Var.new_calc(jnp.sqrt, sigma_sq, name="sigma")
```

### Design matrix, fitted values, and response

To compute the matrix-vector product $\mathbf{X}\boldsymbol{\beta}$, we
use another variable instantiated via {meth}`.Var.new_calc`. We can view
our model as $y_i \sim \mathcal{N}(\mu_i, \;\sigma^2)$ with
$\mu_i = \beta_0 + \beta_1 x_i$, so we use the name `mu` for this
product.

``` python
X = lsl.Var.new_obs(X_mat, name="X")
mu = lsl.Var.new_calc(jnp.dot, X, beta, name="mu")
```

At last we can define our response, using our observed response values.
And since we assumed the model
$y_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i, \;\sigma^2)$, we also need
to specify the response’s distribution. We use our `sigma` and `mu` to
specify this distribution:

``` python
y_dist = lsl.Dist(tfd.Normal, loc=mu, scale=sigma)
y = lsl.Var.new_obs(y_vec, distribution=y_dist, name="y")
```

### Bringing the model together

Now, we can set up the {class}`.Model`. Here, we will only add the
response.

``` python
model = lsl.Model([y])
```

The {func}`.plot_vars()` function visualizes the model. More on that in
the [Model building with Liesel tutorial](01b-model.md) If the layout of
the graph looks messy for you, please make sure you have the
`pygraphviz` package installed.

``` python
lsl.plot_vars(model)
```

![](01a-lin-reg_files/figure-commonmark/plot-vars-3.png)

## MCMC inference with Goose

This section illustrates the basics of Liesel’s MCMC framework Goose. To
use Goose, the user needs to select one or more sampling algorithms,
called (transition) kernels, for the model parameters. Goose comes with
a number of standard kernels such as Hamiltonian Monte Carlo
({class}`~.goose.HMCKernel`) or the No U-Turn Sampler
({class}`~.goose.NUTSKernel`). Multiple kernels can be combined in one
sampling scheme and assigned to different parameters, and the user can
implement their own problem-specific kernels, as long as they are
compatible with the {class}`.Kernel` protocol. In any case, the user is
responsible for constructing a mathematically valid algorithm.

We start with a very simple sampling scheme, keeping $\sigma^2$ fixed at
the true value and using a NUTS sampler for $\boldsymbol{\beta}$. More
on sampling $\sigma^2$ can be found in the [Parameter transformations
tutorial](01c-transform.md) and the [Gibbs sampling
tutorial](01d-gibbs-sampling.md). The kernels are added to a
{class}`~.goose.Engine`, which coordinates the sampling, including the
kernel tuning during the warmup, and the MCMC bookkeeping. The engine
can be configured step by step with a {class}`.EngineBuilder`. Starting
from a Liesel model, it is straight-forward to obtain an engine builder
using the {class}`.LieselMCMC` helper. We then need to define the
kernels, and the sampling duration. Finally, we can call the
{meth}`.EngineBuilder.build` method, which returns a fully configured
MCMC engine.

``` python
builder = gs.LieselMCMC(model).get_engine_builder(seed=1337, num_chains=4)

builder.add_kernel(gs.NUTSKernel(["beta"]))
builder.set_duration(warmup_duration=1000, posterior_duration=1000)

engine = builder.build()
```

Now we can run the MCMC algorithm for the specified duration by calling
the {meth}`~.goose.Engine.sample_all_epochs` method. In a first step,
the model and the sampling algorithm are compiled, so don’t worry if you
don’t see an output right away. The subsequent samples will be generated
much faster.

``` python
engine.sample_all_epochs()
```


      0%|                                                  | 0/3 [00:00<?, ?chunk/s]
     33%|##############                            | 1/3 [00:03<00:06,  3.08s/chunk]
    100%|##########################################| 3/3 [00:03<00:00,  1.03s/chunk]

      0%|                                                  | 0/1 [00:00<?, ?chunk/s]
    100%|########################################| 1/1 [00:00<00:00, 1211.18chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|########################################| 2/2 [00:00<00:00, 1515.56chunk/s]

      0%|                                                  | 0/4 [00:00<?, ?chunk/s]
    100%|########################################| 4/4 [00:00<00:00, 1331.21chunk/s]

      0%|                                                  | 0/8 [00:00<?, ?chunk/s]
    100%|#########################################| 8/8 [00:00<00:00, 895.24chunk/s]

      0%|                                                 | 0/20 [00:00<?, ?chunk/s]
    100%|#######################################| 20/20 [00:00<00:00, 240.41chunk/s]

      0%|                                                  | 0/2 [00:00<?, ?chunk/s]
    100%|########################################| 2/2 [00:00<00:00, 1279.53chunk/s]

      0%|                                                 | 0/40 [00:00<?, ?chunk/s]
     60%|#######################4               | 24/40 [00:00<00:00, 228.33chunk/s]
    100%|#######################################| 40/40 [00:00<00:00, 196.56chunk/s]

Finally, we can extract the results and print a summary table.

``` python
results = engine.get_results()
summary = gs.Summary(results)
summary
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

beta
</th>

<th>

(0,)
</th>

<td>

kernel_00
</td>

<td>

0.982
</td>

<td>

0.087
</td>

<td>

0.840
</td>

<td>

0.983
</td>

<td>

1.126
</td>

<td>

4000
</td>

<td>

831.271
</td>

<td>

877.956
</td>

<td>

1.001
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

1.911
</td>

<td>

0.154
</td>

<td>

1.648
</td>

<td>

1.915
</td>

<td>

2.153
</td>

<td>

4000
</td>

<td>

860.853
</td>

<td>

1205.948
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

55
</td>

<td>

4000
</td>

<td>

4000
</td>

<td>

0.014
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

If we need more samples, we can append another epoch to the engine and
sample it by calling either the {meth}`~.goose.Engine.sample_next_epoch`
or the {meth}`~.goose.Engine.sample_all_epochs` method. The epochs are
described by {class}`.EpochConfig` objects.

``` python
engine.append_epoch(
    gs.EpochConfig(gs.EpochType.POSTERIOR, duration=1000, thinning=1, optional=None)
)
engine.sample_next_epoch()
```


      0%|                                                 | 0/40 [00:00<?, ?chunk/s]
     60%|#######################4               | 24/40 [00:00<00:00, 233.10chunk/s]
    100%|#######################################| 40/40 [00:00<00:00, 200.87chunk/s]

No compilation is required at this point, so this is pretty fast.

Here, we end this first tutorial. We have learned how to build a linear
regression model and seen how we can use Kernels for drawing MCMC
samples - that is quite a bit for the start. Now, have fun modelling
with Liesel!
