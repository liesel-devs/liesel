---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Configure sampling

An iteration applies every configured kernel once. A chain repeats those
iterations; an epoch is a run of iterations with the same phase and storage
settings.

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.goose as gs
import liesel.model as lsl
```

## Prepare the example

These examples require the regression `model`, including its transformation
and inference specifications, from {doc}`tutorials/md/01c-transform`.

```{code-cell} ipython3
:load: _examples/goose-regression.py.inc
:tags: [remove-cell]
```

## Set the phase lengths

For an existing `model` with complete inference specifications:

```{code-cell} ipython3
results = gs.LieselMCMC(model).run_for_epochs(
    seed=2026,
    num_chains=4,
    adaptation=1000,
    burnin=200,
    posterior=2000,
    show_progress=False,
)
```

These counts apply to **each chain**. Adaptation tunes kernel settings.
Optional burnin lets the chains continue with tuning finished. Both belong to
warmup. `get_posterior_samples()` returns only the posterior phase: here,
2,000 draws from each of four chains. The additional recorded initial state
is not a posterior draw.

<figure aria-label="Sampling phases from initialization to posterior draws">
  <div style="display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;margin:1rem 0">
    <div style="border:1px solid #999;padding:.7rem">Initial<br>state</div>
    <div style="border:2px solid #397e92;padding:.7rem;flex:2">
      <strong>Adaptation</strong><br>Fast → expanding slow epochs → fast
    </div>
    <div style="border:2px solid #397e92;padding:.7rem">Optional<br>burnin</div>
    <div style="border:2px solid #497a32;padding:.7rem;flex:2">
      <strong>Posterior sampling</strong><br>Draws used for inference
    </div>
  </div>
  <figcaption>Warmup includes adaptation and burnin. Box widths do not represent recommended durations.</figcaption>
</figure>

During adaptation, kernels can tune within iterations and between epochs.
NUTS and HMC tune step size and update their inverse mass matrix after slow
epochs. RW and IWLS tune step size; Gibbs has nothing to tune.
See {doc}`goose-engine` to customize the schedule.

The example lengths are a starting point. Use
{doc}`diagnostics <goose-diagnostics>` to assess the result. More posterior
iterations do not repair invalid starting values or a problematic model.

## Thin stored draws

Keep `posterior_thinning=1` unless storing the draws is too expensive.
For example, requesting 2,000 posterior iterations with
`posterior_thinning=2` retains 1,000 draws per chain; all 2,000 transitions
still run. Thinning discards information and does not fix poor mixing.

Avoid adaptation thinning for kernels that rely on sample history for tuning.
See {meth}`~liesel.goose.EngineBuilder.add_adaptation` for constraints.

(goose-record-tuning)=

## Inspect tuning

Kernel states must be recorded during sampling. For the single joint NUTS kernel in
{doc}`tutorials/md/01c-transform`, run:

```{code-cell} ipython3
recorded = gs.LieselMCMC(model).run_for_epochs(
    seed=2026,
    num_chains=4,
    adaptation=1000,
    posterior=1000,
    show_progress=False,
    store_kernel_states=True,
)

states = recorded.get_warmup_kernel_states()
kernel_id = recorded.get_kernels_by_pos_key()["beta"]
step_size = states[kernel_id]["step_size"]
```

```{code-cell} ipython3
step_size.shape
```

```{code-cell} ipython3
pd.DataFrame(
    {
        "Chain": range(step_size.shape[0]),
        "Last recorded step size": np.asarray(step_size[:, -1]),
    },
).round(3)
```

`step_size` has shape `(chains, warmup iterations)`. It contains the
adapting step sizes. The table shows the last recorded step size
for each chain, not convergence diagnostics. To investigate changes over
warmup, compare earlier entries along the second axis. Values may change at
epoch boundaries; the final dual-averaging update can also change the step
size used for posterior sampling.
Recording kernel states uses additional memory, especially for mass matrices.
It cannot be enabled retroactively for an existing result.

## Allow compilation time

Goose uses JAX to compile the sampling calculations. Compilation can make the
first part of a run slower than later iterations. Multiple chains have
separate states and random keys; they do not imply one physical CPU core per
chain. Record the backend and numerical environment for
{doc}`reproducibility <tutorials/md/05-reproducibility>`.
