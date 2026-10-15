---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
---

# Choose optimizers

{py:class}`liesel.optim.LieselOptim` requires an explicit optimizer. Pass a
configured Optax transformation, such as `optimizers=optax.adam(0.01)`, to use
it for all parameters, or `optimizers="lbfgs"` for a full-data, deterministic
fit. L-BFGS cannot use minibatches.

The examples assume a Gaussian regression `model` with observed `X` and `y`,
as in the {doc}`first tutorial <tutorials/notebooks/09-liesel-optim-basic>`.

```{code-cell} ipython3
import logging

import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt
```

```{code-cell} ipython3
:tags: [remove-cell]

logging.getLogger("liesel").setLevel(logging.WARNING)

rng = np.random.default_rng(42)
x = np.linspace(-1.0, 1.0, 128)

X = lsl.Var.new_obs(jnp.asarray(x), name="X")
beta = lsl.Var.new_param(
    0.0,
    dist=lsl.Dist(tfd.Normal, 0.0, 5.0),
    name="beta",
)

log_sigma = lsl.Var.new_param(0.0, name="log_sigma")
sigma = lsl.Var.new_calc(jnp.exp, log_sigma, name="sigma")

mu = lsl.Var.new_calc(
    lambda X, beta: X * beta,
    X,
    beta,
    name="mu",
)

y = lsl.Var.new_obs(
    jnp.asarray(0.5 * x + rng.normal(scale=0.7, size=x.size)),
    dist=lsl.Dist(tfd.Normal, mu, sigma),
    name="y",
)
model = lsl.Model(y)
```

## Set the learning rate

Pass a configured Optax optimizer directly:

```{code-cell} ipython3
schedule = optax.exponential_decay(
    init_value=0.01,
    transition_steps=100,
    decay_rate=0.9,
)

result = opt.LieselOptim(
    model,
    optimizers=optax.adam(schedule),
    loss_monitor="train_full_data",
    show_progress=False,
).fit()
```

```{code-cell} ipython3
result.position_min_monitor
```

The schedule advances on optimizer updates, not epochs. With minibatches, it can
advance several times per epoch. For a constant rate, use `optax.adam(0.01)`.
Minibatch fits can retain optimization noise. For precise point estimates,
check convergence and consider a smaller rate or a final full-data fit.

## Choose parameters

Give the slope and log noise scale different learning rates:

```{code-cell} ipython3
optimizers = [
    opt.Optimizer(["beta"], optax.adam(0.01)),
    opt.Optimizer(["log_sigma"], optax.adam(0.001)),
]
```

Pass this list as `optimizers=optimizers` to {class}`LieselOptim <liesel.optim.LieselOptim>`. Each optimizer
updates its own parameters in list order on every batch. Their parameter names
must not overlap. Parameters left out of the list stay fixed.
Separate blocks each evaluate the loss and gradient, adding work.

L-BFGS must be the only optimizer because other parameter updates invalidate its
cached objective and curvature history. Use `optimizers="lbfgs"` for all model
parameters, or `optimizers=[opt.LBFGS(["beta", "log_sigma"])]` for a selected
subset.

## Choose the source

A prior may be attached to a weak parameter computed from a strong source variable.
That prior remains in the default loss, including its derivatives through the weak
parameter. Automatic parameter selection cannot decide which source to estimate
and raises an informative error. Supply the strong names explicitly, for example
`optimizers=[opt.Optimizer(["source"], optax.adam(0.01))]` or
`optimizers=[opt.LBFGS(["source"])]`. The source does not need to be marked as a
parameter. The weak parameter itself is recomputed during fitting.

## Switch to L-BFGS

Use two separate fits to switch from Adam to L-BFGS. The optimizer option
`activate_after_epochs` delays activation; it does not deactivate Adam.
Keep the same parameter names in both fits:

```{code-cell} ipython3
first = opt.LieselOptim(
    model,
    optimizers=optax.adam(0.01),
    loss_monitor="train_full_data",
    show_progress=False,
).fit()
model.state = model.update_state(first.position_final, model.state)

result = opt.LieselOptim(
    model,
    optimizers="lbfgs",
    loss_monitor="train_full_data",
    show_progress=False,
).fit()
```

```{code-cell} ipython3
result.position_min_monitor
```

Assign the returned model state before constructing the second fit:
`update_state` leaves the model unchanged by default. L-BFGS starts with fresh
optimizer state at Adam's final parameter values and uses full-data batches here.
Do not pass Adam's checkpoint to the L-BFGS fit.

## Save less history

Disable parameter history when you only need the final and best positions:

```{code-cell} ipython3
result = opt.LieselOptim(
    model,
    optimizers=optax.adam(0.01),
    loss_monitor="train_full_data",
    show_progress=False,
    save_position_history=False,
).fit()
```

History reserves memory for the maximum epoch budget before fitting, even when
early stopping ends the run sooner. Its parameter storage costs approximately
`epochs * total_parameter_bytes`: one million float32 parameters across 1,000
epochs take about 4 GB. Disabling it retains scalar losses and final/best positions.

Use {meth}`build_engine() <liesel.optim.LieselOptim.build_engine>` for settings beyond the wrapper's arguments.
See {py:class}`~liesel.optim.OptimEngine` for all settings, or
{doc}`optimizer-checkpointing` to pause and resume a fit.

## Write a custom loss

Inherit from {py:class}`liesel.optim.LossMixin` to obtain gradient helpers. Define
`split`, `position(keys)`, and `loss_train_batched(params, carry)`; also define
`loss_train` for full-training monitoring or `loss_monitor` for validation.
All three evaluation methods return `(value, proposed_state)`. For a stateless
loss, return `(value, None)`. The mixin differentiates only `value`.

For stateful losses, implement `init_state(params, carry)` and read the committed
state from `carry.loss_state`. Return proposals without changing that input.
The engine commits state only after a valid full-training evaluation at the
end of an epoch. Stateful losses require full-data batches and
`loss_monitor="train_full_data"`. See {py:meth}`~liesel.optim.LossMixin.init_state`
for the state contract.

Set `default_position_keys` to select the parameters used by {class}`LieselOptim <liesel.optim.LieselOptim>`
with a bare Optax transformation or `"lbfgs"`. The default `None` selects all
model parameters. Explicit optimizer blocks always retain their own keys.
