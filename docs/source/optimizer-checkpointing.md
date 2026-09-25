---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  execution_mode: force
---

# Pause and resume

Build an engine to pause a run or save progress. This small Normal model
uses Adam so we can inspect it partway through fitting.

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
    jnp.array([1.0, 2.0, 3.0]),
    dist=lsl.Dist(tfd.Normal, loc, 1.0),
    name="y",
)
model = lsl.Model(y)

optim = opt.LieselOptim(
    model,
    optimizers=optax.adam(0.01),
    loss_monitor="train_full_data",
    stopper=opt.Stopper(epochs=300, patience=20),
    show_progress=False,
)
```

## Pause in memory

```{code-cell} ipython3
engine = optim.build_engine()
first = engine.fit(pause_after=10)
```

```{code-cell} ipython3
first.status
```

```{code-cell} ipython3
result = engine.fit(checkpoint=first.checkpoint)
```

```{code-cell} ipython3
result.position_min_monitor
```

`pause_after` limits additional epochs in that call. Early stopping still
applies. To extend the total budget, change `engine.stopper.epochs` before
resuming. Calling `fit()` without a checkpoint starts a new run.

The result's `status` tells you why fitting stopped: `"paused"`,
`"max_epochs"`, `"early_stopping"`, `"numerical_failure"`, or `"nan"`.
A numerical-failure or NaN result has no
checkpoint; recover from an earlier saved checkpoint instead.

## Save to disk

Use the same checkpoint path on the first job and after an interruption.
This example uses a temporary directory so rerunning the guide starts fresh:

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    checkpoint_path = Path(directory) / "optim.pkl"
    first = engine.fit(
        checkpoint=checkpoint_path,
        checkpoint_every=10,
        pause_after=10,
    )

    result = engine.fit(checkpoint=checkpoint_path, checkpoint_every=10)
```

```{code-cell} ipython3
result.status
```

For a real run, choose a persistent path such as `"optim.pkl"`.

A missing file starts a new run. An existing file resumes it. The engine saves
every ten epochs here, plus at deliberate pauses and normal completion. A crash
or timeout loses work since the last successful save. Failed writes and numerical
failures leave the previous file intact.

Use a different path for a new experiment and only one writer per path. The
parent directory must exist.

If the epoch budget is exhausted or early stopping applies, resuming warns
and returns without further updates. Increasing `engine.stopper.epochs`
extends the budget but does not override early stopping.

## Resume safely

* Recreate the same model, data, and optimizer settings. Checks cover structure,
  names, shapes, dtypes, and batching, but cannot detect changed data or learning
  rates. Weighted sampling resumes with its saved probabilities.
* Keep the same Liesel, JAX, jaxlib, Optax, and NumPy versions. The option
  `allow_version_mismatch=True` attempts recovery across versions without
  promising compatibility. Different hardware can also change results.
* Load only trusted files: checkpoints use Python pickle, which can execute code.
  They save run state, not model code, and are meant for recovery rather than
  long-term model storage.

History and monitoring continue across pauses. Earlier results stay unchanged;
keeping many snapshots uses extra memory. Treat checkpoint contents as read-only.
Custom losses must keep their model state and loss-state PyTrees compatible
and serializable with pickle. Committed and best loss states survive recovery.
For `LaplaceLoss`, keep the latent parameters and inner-solver controls unchanged;
checkpoint recovery checks these settings as well as state structure and dtype.

For manual snapshots, use {py:meth}`liesel.optim.OptimCheckpoint.save` and
{py:meth}`liesel.optim.OptimCheckpoint.load`. Passing a checkpoint object resumes
in memory; pass a path to `fit()` for automatic disk saves. See
{py:meth}`liesel.optim.OptimEngine.fit` for all recovery options.
