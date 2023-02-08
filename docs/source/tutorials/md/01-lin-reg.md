
# Linear regression

In this tutorial, we build a linear regression model with Liesel and
estimate it with Goose. Our goal is to illustrate the most important
features of the software in a straightforward context.

## Model building with Liesel

Liesel is based on the concept of probabilistic graphical models (PGMs)
to represent (primarily Bayesian) statistical models, so let us start
with a very brief look at what PGMs are and how they are implemented in
Liesel.

### Probabilistic graphical models

In a PGM, each variable is represented as a node. There are two basic
types of nodes in Liesel: strong and weak nodes. A strong node is a node
whose value is defined “outside” of the model, for example, if the node
represents some observed data or a parameter (parameters are usually set
by an inference algorithm such as an optimizer or sampler). In contrast,
a weak node is a node whose value is defined “within” the model, that
is, it is a deterministic function of some other nodes. An
exp-transformation mapping a real-valued parameter to a positive number,
for example, would be a weak node.

In addition, each node can have an optional probability distribution.
The probability density or mass function of the distribution evaluated
at the value of the node gives its log-probability. In a typical
Bayesian regression model, the response node would have a normal
distribution and the parameter nodes would have some prior distribution
(for example, a normal-inverse-gamma prior). The following table shows
the different node types and some examples of their use cases.

|                          | **Strong node**              | **Weak node**                                      |
|--------------------------|------------------------------|----------------------------------------------------|
| **With distribution**    | Response, parameter, …       | Copula, …                                          |
| **Without distribution** | Covariate, hyperparameter, … | Inverse link function, parameter transformation, … |

A PGM is essentially a collection of connected nodes. Two nodes can be
connected through a directed edge, meaning that the first node is an
input for the value or the distribution of the second node. Nodes
*without* an edge between them are assumed to be conditionally
independent, allowing us to factorize the model log-probability as

$$
\log p(\text{Model}) = \sum_{\text{Node $\in$ Model}} \log p(\text{Node} \mid \text{Inputs}(\text{Node})).
$$

### Generating the data

Before we can generate the data and build the model graph, we need to
load Liesel and a number of other packages. We usually import the model
building library `liesel.model` as `lsl`, and the MCMC library
`liesel.goose` as `gs`.

``` python
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import matplotlib.pyplot as plt
import numpy as np

# We use distributions and bijectors from tensorflow probability
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

rng = np.random.default_rng(42)
```

Now we can simulate 500 observations from the linear regression model
$y_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i, \;\sigma^2)$ with the true
parameters $\boldsymbol{\beta} = (\beta_0, \beta_1)' = (1, 2)'$ and
$\sigma = 1$. The relationship between the response $y_i$ and the
covariate $x_i$ is visualized in the following scatterplot.

``` python
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

![](01-lin-reg_files/figure-commonmark/unnamed-chunk-3-1.png)

### Building the model graph

The graph of a Bayesian linear regression model is a tree, where the
hyperparameters of the prior are the leaves and the response is the
root. To build this tree in Liesel, we need to start from the leaves and
work our way down to the root.

We can use the following classes to build the tree:

- {func}`.Obs`: We use this class to include observed data. Here, that
  means the response and the covariate values.
- {func}`.Param`: We use this class to include the model parameters that
  we want to estimate. Here, that includes the regression coefficients
  $\boldsymbol{\beta} = [\beta_0, \beta_1]^T$ and the variance
  $\sigma^2$.
- {class}`.Var`: We use this class to include other variables that we
  need for our model. Here, that will be the response and the
  hyperparameters of our priors for $\boldsymbol{\beta}$ and $\sigma^2$.

#### The regression coefficients

Let’s assume the weakly informative prior
$\beta_0, \beta_1 \sim \mathcal{N}(0, 100^2)$ for the regression
coefficients. To encode this assumption in Liesel, we need to create
hyperparameter nodes for the mean and the standard deviation of the
normal prior. Setting a name when creating a node is optional, but helps
to identify it later.

``` python
beta_loc = lsl.Var(0.0, name="beta_loc")
beta_scale = lsl.Var(100.0, name="beta_scale") # scale = sqrt(100^2)

beta_loc
```

    Var<beta_loc>

``` python
beta_scale
```

    Var<beta_scale>

Now, let us create the node for the regression coefficients.

To do so, we need to define its initial value and its node distribution
using the {class}`.Dist` class. This class wraps distribution classes
with the TensorFlow Probability (TFP) API to connect them to our node
classes. Here, the node distribution is initialized with three
arguments: the TFP distribution object
[(`tfd.Normal`)](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal),
and the two hyperparameter nodes representing the parameters of the
distribution. TFP uses the names `loc` for the mean and `scale` for the
standard deviation, so we have to use the same names here. This is a
general feature of {class}`.Dist`, you can always use the parameter
names from TFP to refer to the parameters of your distribution.

``` python
beta_dist = lsl.Dist(tfd.Normal, loc=beta_loc, scale=beta_scale)
```

With this distribution object, we can now create the node for our
regression coefficient with the {func}`.Param` class:

``` python
beta = lsl.Param(value=np.array([0.0, 0.0]), distribution=beta_dist,name="beta")
```

#### The standard deviation

The second branch of the tree contains the residual standard deviation.
We build it in a similar way, but this time, using the weakly
informative prior $\sigma \sim \text{InverseGamme}(0.01, 0.01)$. Again,
we use the parameter names based on TFP.

``` python
sigma_a = lsl.Var(0.01, name="a")
sigma_b = lsl.Var(0.01, name="b")

sigma_dist = lsl.Dist(tfd.InverseGamma, concentration=sigma_a, scale=sigma_b)
sigma = lsl.Param(value=10.0, distribution=sigma_dist, name="sigma")
```

#### Design matrix, fitted values, and response

All nodes we have seen so far are strong nodes. Before we can create a
weak node that computes the predictions
$\hat{\boldsymbol{y}} = \mathbf{X}\boldsymbol{\beta}$, we need to set up
one more strong node for the design matrix. This is done quickly with a
{func}`.Obs` node:

``` python
X = lsl.Obs(X_mat, name="X")
```

To compute the matrix-vector product
$\hat{\boldsymbol{y}} = \mathbf{X}\boldsymbol{\beta}$, we make our first
use of the {class}`.Calc` class. We can use this class to include
computations based on our nodes. It always takes a function as its first
argument, and the nodes to be used as function inputs as the following
arguments. In this case, we can create the calculator like this:

``` python
yhat_fn = lambda x, beta: jnp.dot(x, beta) # stick to JAX numpy functions in calculators
calc = lsl.Calc(yhat_fn, x=X, beta=beta)
```

With this calculator in place, we can create a corresponding node that
represents the fitted values. For this, the {class}`.Var` class is the
right choice. As the node’s value, we use the `calc` object we just
created.

``` python
y_hat = lsl.Var(calc, name="y_hat")
```

Finally, we can connect the branches of the tree in a response node. The
value of the node is the simulated response vector - our observed
response values. And since we assumed the model
$y_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i, \;\sigma^2)$, we also need
to specify the response’s distribution. For that, we use `y_hat` to
represent the mean (/location) and `sigma` to represent the standard
deviation (/scale).

``` python
y_dist = lsl.Dist(tfd.Normal, loc=y_hat, scale=sigma)
y = lsl.Var(y_vec, distribution=y_dist, name="y")
```

#### Bringing the model together

Now, to construct a full-fledged Liesel model from our individual node
objects, we can use the {class}`.GraphBuilder` class. Here, we will only
add the response node.

``` python
gb = lsl.GraphBuilder().add(y)
gb
```

    GraphBuilder<0 nodes, 1 vars>

Since all other nodes are directly or indirectly connected to this node,
the GraphBuilder will add those nodes automatically when it builds the
model. Let us do that now with a call to {meth}`.build_model()`. The
model that is returned by the builder provides a couple of convenience
function, for example, to evaluate the model log-probability, or to
update the nodes in a topological order.

``` python
model = gb.build_model()
```

    No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

``` python
model
```

    Model<24 nodes, 9 vars>

The {func}`.plot_vars()` function visualizes the graph of a model.
Strong nodes are shown in blue, weak nodes in red. Nodes with a
probability distribution are highlighted with a star. In the figure
below, we can see the tree-like structure of the graph and identify the
two branches for the mean and the standard deviation of the response. By
the way, if the layout of the graph looks messy for you, please make
sure you have the `pygraphviz` package installed.

``` python
lsl.plot_vars(model)
```

![](01-lin-reg_files/figure-commonmark/unnamed-chunk-15-3.png)

### Node and model log-probabilities

The log-probability of the model, which can be interpreted as the
(unnormalized) log-posterior in a Bayesian context, can be accessed with
the `log_prob` property.

``` python
model.log_prob
```

    Array(-1641.4343, dtype=float32)

The individual nodes also have a `log_prob` property. In fact, because
of the conditional independence assumption of the model, the
log-probability of the model is given by the sum of the
log-probabilities of the nodes with probability distributions. We take
the sum for the `.log_prob` attributes of `beta` and `y` because, per
default, the attributes return the individual log-probability
contributions of each element in the values of the nodes. So for `beta`
we would get two log-probability values, and for `y` we would get 500.

``` python
beta.log_prob.sum() + sigma.log_prob + y.log_prob.sum()
```

    Array(-1641.4343, dtype=float32)

Nodes without a probability distribution return a log-probability of
zero.

``` python
beta_loc.log_prob
```

    0.0

The log-probability of a node depends on its value and its inputs. Thus,
if we change the standard deviation of the response from 10 to 1, the
log-probability of the corresponding node, the log-probability of the
response node, and the log-probability of the model change as well.

``` python
print(f"Old value of sigma: {sigma.value}")
```

    Old value of sigma: 10.0

``` python
print(f"Old log-prob of sigma: {sigma.log_prob}")
```

    Old log-prob of sigma: -6.972140312194824

``` python
print(f"Old log-prob of y: {y.log_prob.sum()}\n")
```

    Old log-prob of y: -1623.4139404296875

``` python
sigma.value = 1.0

print(f"New value of sigma: {sigma.value}")
```

    New value of sigma: 1.0

``` python
print(f"New log-prob of sigma: {sigma.log_prob}")
```

    New log-prob of sigma: -4.655529975891113

``` python
print(f"New log-prob of y: {y.log_prob.sum()}\n")
```

    New log-prob of y: -1724.6702880859375

``` python
print(f"New model log-prob: {model.log_prob}")
```

    New model log-prob: -1740.3740234375

For most inference algorithms, we need the gradient of the model
log-probability with respect to the parameters. Liesel uses [the JAX
library for numerical computing and machine
learning](https://github.com/google/jax) to compute gradients using
automatic differentiation.

## MCMC inference with Goose

This section illustrates the key features of Liesel’s MCMC framework
Goose. To use Goose, the user needs to select one or more sampling
algorithms, called (transition) kernels, for the model parameters. Goose
comes with a number of standard kernels such as Hamiltonian Monte Carlo
({class}`.HMCKernel`) or the No U-Turn Sampler ({class}`.NUTSKernel`).
Multiple kernels can be combined in one sampling scheme and assigned to
different parameters, and the user can implement their own
problem-specific kernels, as long as they are compatible with the
{class}`.Kernel` protocol. In any case, the user is responsible for
constructing a *mathematically valid* algorithm.

We start with a very simple sampling scheme, keeping $\sigma$ fixed at
the true value and using a NUTS sampler for $\boldsymbol{\beta}$. The
kernels are added to a {class}`.Engine`, which coordinates the sampling,
including the kernel tuning during the warmup, and the MCMC bookkeeping.
The engine can be configured step by step with a
{class}`.EngineBuilder`. We need to inform the builder about the model,
the initial values, the kernels, and the sampling duration. Finally, we
can call the {meth}`.EngBuilder.build` method, which returns a fully
configured engine.

``` python
sigma.value = true_sigma

builder = gs.EngineBuilder(seed=1337, num_chains=4)

builder.set_model(lsl.GooseModel(model))
builder.set_initial_values(model.state)

builder.add_kernel(gs.NUTSKernel(["beta"]))

builder.set_duration(warmup_duration=1000, posterior_duration=1000)

engine = builder.build()
```

Now we can run the MCMC algorithm for the specified duration by calling
the {meth}`.sample_all_epochs` method on the engine. In a first step,
the model and the sampling algorithm are compiled, so don’t worry if you
don’t see an output right away. The subsequent samples will be generated
much faster. Finally, we can extract the results and print a summary
table.

``` python
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 3, 3, 2 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 2, 1, 2 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 2, 1, 1 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 1, 3, 2 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 1, 1, 2 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 500 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 3, 3, 5, 2 / 500 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 3, 4, 2 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch

``` python
results = engine.get_results()
gs.Summary.from_result(results)
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
0.980009
</td>
<td>
0.086085
</td>
<td>
0.837104
</td>
<td>
0.981508
</td>
<td>
1.122109
</td>
<td>
4000
</td>
<td>
1073.714254
</td>
<td>
1372.598600
</td>
<td>
1.003053
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
1.917174
</td>
<td>
0.149760
</td>
<td>
1.674593
</td>
<td>
1.916041
</td>
<td>
2.163207
</td>
<td>
4000
</td>
<td>
1086.454184
</td>
<td>
1194.724696
</td>
<td>
1.003285
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
61
</td>
<td>
0.01525
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

If we need more samples, we can append another epoch to the engine and
sample it by calling either the {meth}`.sample_next_epoch` or the
{meth}`.sample_all_epochs` method. The epochs are described by
{class}`.EpochConfig` objects.

``` python
engine.append_epoch(
    gs.EpochConfig(gs.EpochType.POSTERIOR, duration=1000, thinning=1, optional=None)
)

engine.sample_next_epoch()
```

    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch

No compilation is required at this point, so this is pretty fast.

### Using a Gibbs kernel

So far, we have not sampled our standard deviation parameter `sigma`; we
simply fixed it to zero. Now we extend our model with a Gibbs sampler
for `sigma`. Using a Gibbs kernel is a bit more complicated, because
Goose doesn’t automatically derive the full conditional from the model
graph. Hence, the user needs to provide a function to sample from the
full conditional. The function needs to accept a PRNG state and a model
state as arguments, and it needs to return a dictionary with the node
name as the key and the new node value as the value. We could also
update multiple parameters with one Gibbs kernel if we returned a
dictionary of length two or more.

To retrieve the values of our nodes from the `model_state`, we need to
add the suffix `_value` behind the nodes’ names. Likewise, the node name
in returned dictionary needs to have the added `_value` suffix.

``` python
def draw_sigma(prng_key, model_state):
    a_prior = model_state["a_value"].value
    b_prior = model_state["b_value"].value
    n = len(model_state["y_value"].value)

    resid = model_state["y_value"].value - model_state["y_hat_value"].value

    a_gibbs = a_prior + n / 2
    b_gibbs = b_prior + jnp.sum(resid**2) / 2
    draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)
    return {"sigma_value": draw}
```

We build the engine in a similar way as before, but this time adding the
Gibbs kernel as well.

``` python
builder = gs.EngineBuilder(seed=1338, num_chains=4)

builder.set_model(lsl.GooseModel(model))
builder.set_initial_values(model.state)

builder.add_kernel(gs.NUTSKernel(["beta"]))
builder.add_kernel(gs.GibbsKernel(["sigma"], draw_sigma))

builder.set_duration(warmup_duration=1000, posterior_duration=1000)

engine = builder.build()
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 5, 3, 4 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 1, 1, 1 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 3, 1, 2 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 3, 1, 2 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 2, 3, 2, 1 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 500 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 6, 2, 4, 1 / 500 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 4, 3, 1, 2 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch

Goose provides a couple of convenient numerical and graphical summary
tools. The {class}`.Summary` class computes several summary statistics
that can be either accessed programmatically or displayed as a summary
table.

``` python
results = engine.get_results()
gs.Summary.from_result(results)
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
0.983772
</td>
<td>
0.094300
</td>
<td>
0.829190
</td>
<td>
0.983157
</td>
<td>
1.141740
</td>
<td>
4000
</td>
<td>
981.401587
</td>
<td>
1000.792237
</td>
<td>
1.004109
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
1.908277
</td>
<td>
0.166134
</td>
<td>
1.635700
</td>
<td>
1.910057
</td>
<td>
2.182816
</td>
<td>
4000
</td>
<td>
1013.081369
</td>
<td>
998.073902
</td>
<td>
1.007787
</td>
</tr>
<tr>
<th>
sigma
</th>
<th>
()
</th>
<td>
kernel_01
</td>
<td>
1.043638
</td>
<td>
0.065514
</td>
<td>
0.940217
</td>
<td>
1.041130
</td>
<td>
1.155027
</td>
<td>
4000
</td>
<td>
3814.700615
</td>
<td>
3773.871037
</td>
<td>
1.000356
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
65
</td>
<td>
0.01625
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

</div>

We can plot the trace plots of the chains with {func}`.plot_trace()`.

``` python
g = gs.plot_trace(results)
```

![](01-lin-reg_files/figure-commonmark/unnamed-chunk-26-5.png)

We could also take a look at a kernel density estimator with
{func}`.plot_density()` and the estimated autocorrelation with
{func}`.plot_cor()`. Alternatively, we can output all three diagnostic
plots together with {func}`.plot_param()`. The following plot shows the
parameter $\beta_0$.

``` python
gs.plot_param(results, param="beta", param_index=0)
```

![](01-lin-reg_files/figure-commonmark/unnamed-chunk-27-7.png)

Here, we end this first tutorial. We have learned about a lot of
different classes and seen how we can flexibly use different Kernels for
drawing MCMC samples - that is quite a bit for the start. Now, have fun
modelling with Liesel!