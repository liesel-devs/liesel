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

The default training loss combines likelihood and priors. Each factor whose
log-density depends on a batched key is scaled up to represent its full training
group. This includes local parameter priors as well as observed likelihoods.
Priors independent of batched keys are included once, without scaling.
Each group gets its own factor, so different batch sizes do not change the
relative weight of the groups.

Validation and test likelihoods include only the split branches and are scaled
to the corresponding training size. Unsplit observed likelihoods contribute to
training only, including observations explicitly marked as passthrough.
{class}`LieselOptim <liesel.optim.LieselOptim>` then divides losses by the total training sample size by default;
use `scale_loss=False` to keep the sum. Sample size counts likelihood terms,
which need not equal the number of array elements.

Validation leaves out priors by default. With `validation_strategy="log_prob"`,
global priors are included once, including priors computed from split data. The
built-in losses reject validation splits whose keys reach a local parameter
prior, one whose log-density shape changes with the split: held-out local latents
may receive no training gradient, so scoring them at their starting values gives
a misleading validation loss and can stop training too early. Use a custom `Loss`
that handles held-out latents, or train without a validation split. This
restriction applies to both validation strategies. Test parts are not scored
during fitting and are allowed. The check covers only priors: a per-unit
parameter without a prior, such as `z` in `y ~ N(z[idx], 1)`, passes it but gets
the same constant validation loss. See {py:class}`~liesel.optim.NegLogProbLoss`
for details.

Custom losses can reuse the split's `scaled_log_lik` and `scaled_log_prior`
helpers after inferring held-out local latents. These helpers only scale the
supplied state: covered local priors receive the same training-to-held-out
scale as their likelihoods, and global priors are included once.

## Batch local priors

A local prior can evaluate only the terms selected by a batched index while its
parameter retains all latent values:

```python
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt


class TermsAt:
    def __init__(self, idx):
        self.idx = idx

    def log_prob(self, z):
        return tfd.Normal(0.0, 1.0).log_prob(z[self.idx])


y_values = jnp.linspace(1.0, 3.0, 100)
idx = lsl.Var.new_obs(jnp.arange(100), name="idx")
z = lsl.Var.new_param(jnp.zeros(100), lsl.Dist(TermsAt, idx), name="z")
z_batch = lsl.Var.new_calc(lambda z, i: z[i], z, idx)
y = lsl.Var.new_obs(y_values, lsl.Dist(tfd.Normal, z_batch, 1.0), name="y")
model = lsl.Model(y)
split = opt.PositionSplit.from_model(model, position_keys=["idx", "y"])
batches = opt.Batches.from_split(split, batch_size=10)
```

Both the likelihood and local prior receive a factor of ten. The prior remains
in `_model_log_prior`, and `_model_log_lik` still contains only likelihoods.
The model's state totals themselves are unscaled; scaling occurs in the loss.

Dependency tracing follows both values and distribution inputs, including
computed inputs. For minibatches, indirectly selected factors must return an
array with an axis of batch-size length that has another length in the model's
full-data state. This rejects data-dependent global priors, such as a prior
whose scale is computed from a batched sample's standard deviation, even when
their length equals the batch size. Compute such global summaries from the full data and keep
them fixed as passthrough inputs, or use a custom loss. Full-data evaluation is
unchanged. Shape checks cannot prove that the terms decompose over selected
units: the model must still satisfy that requirement.

A factor depending on two independent
groups is rejected. Keep aligned keys in one group and shared inputs such as
spline knots fixed with `split_axes={"knots": None}`. Prior-only groups are also
supported; without likelihood contributions, sample sizes fall back to axis sizes.

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
