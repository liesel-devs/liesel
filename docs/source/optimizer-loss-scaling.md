---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
---

(optimizer-likelihood-scaling)=

# Scale the loss

The default training loss combines likelihood and priors. A minibatch likelihood
is scaled up to represent its full training group; priors are not scaled up.
Each group gets its own factor, so different batch sizes do not change the
relative weight of the groups.

Validation and test likelihoods include only the split branches and are scaled
to the corresponding training size. Unsplit observed likelihoods contribute to
training only, including observations explicitly marked as passthrough.
{class}`LieselOptim <liesel.optim.LieselOptim>` then divides losses by the total training sample size by default;
use `scale_loss=False` to keep the sum. Sample size counts likelihood terms,
which need not equal the number of array elements.

{py:class}`~liesel.optim.LieselVI` applies the same training-likelihood scaling to
{py:class}`~liesel.optim.NegElboLoss`. Its default `scale_loss=True` divides the
complete negative ELBO by the training sample size, including entropy and prior
terms. For an explicitly constructed loss, set `scale` on that loss instead.

Validation leaves out priors by default. Use `validation_strategy="log_prob"`
to include them. See {py:class}`~liesel.optim.NegLogProbLoss` for details.

<iframe
  class="interactive-visualization"
  data-visualization="likelihood"
  src="_static/visualizations/likelihood-scaling.html"
  title="Interactive overview of likelihood scaling through splitting and batching"
  loading="lazy"
  sandbox="allow-scripts allow-same-origin">
</iframe>

[Open the likelihood-scaling overview in a separate page](_static/visualizations/likelihood-scaling.html).

## Set sample sizes

Observed distributions with `per_obs=False` sum their likelihood before
returning it, preventing automatic sample-size inference. Build a small
Normal model with this setting and construct its split explicitly:

```{code-cell} ipython3
import logging

import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt

logging.getLogger("liesel").setLevel(logging.WARNING)

loc = lsl.Var.new_param(0.0, name="loc")
y = lsl.Var.new_obs(
    jnp.array([1.0, 2.0, 3.0]), lsl.Dist(tfd.Normal, loc, 1.0), name="y"
)
y.dist_node.per_obs = False
model = lsl.Model(y)

split = opt.PositionSplit.from_model(
    model, infer_sample_sizes=False, multi_size="manager", shuffle=False
)
optim = opt.LieselOptim(
    model,
    split=split,
    optimizers=optax.adam(0.01),
    loss_monitor="train_full_data",
    show_progress=False,
)
```

```{code-cell} ipython3
split.train_sample_size
```

```{code-cell} ipython3
result = optim.fit()
```

```{code-cell} ipython3
result.position_min_monitor
```

This chooses split-axis counts for scaling. Alternatively, supply effective
`sample_sizes` to the split factory. Setting `scale_loss=False` on
{class}`LieselOptim <liesel.optim.LieselOptim>` only disables final loss normalization; it does not disable
split inference or specify batch scaling.

## Use a custom loss

Custom aggregate likelihood, prior, or probability nodes require a custom
{py:class}`~liesel.optim.Loss` and an explicit split. A manual split specifies data
grouping and scaling; it does not change the objective used by
{py:class}`~liesel.optim.NegLogProbLoss`. The built-in loss accepts the standard sums
of observed distribution factors and parameter priors, including weak observed
variables. Other distribution factors must be classified appropriately or handled
by a custom loss.

For the data setup, see {doc}`optimizer-splitting` and
{doc}`optimizer-batching`. {doc}`optimizer-weighted-batching` explains the
additional correction for unequal sampling probabilities.

<script src="_static/visualizations/resize-frames.js"></script>
