---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Reproducibility

Record both the random seed and the computational environment when saving an
analysis. A seed controls random draws; it does not by itself guarantee identical
floating-point results across environments.

## PRNG seeding

This example uses the regression model from {doc}`01c-transform`, including its
kernel assignments and jitter settings. It starts from that model's initial
state, independently of earlier notebook sessions.

```{code-cell} ipython3
:load: ../../_examples/goose-regression.py
:tags: [remove-cell]
```

```{code-cell} ipython3
results = gs.LieselMCMC(model).run_for_epochs(
    seed=1,
    num_chains=4,
    adaptation=500,
    posterior=500,
    show_progress=False,
)
samples = results.get_posterior_samples()
```

```{code-cell} ipython3
{name: value.shape for name, value in samples.items()}
```

The seed configures Goose's random-key stream, including chain initialization.
Keep data, initial values, jitter, kernel ordering, and iteration counts with
it. The lower-level {class}`~liesel.goose.EngineBuilder` also takes a seed.

[JAX uses explicit random keys](https://docs.jax.dev/en/latest/random-numbers.html).
When simulating data yourself, split a key for separate draws. Reusing a key
repeats its random stream; it does not produce an independent replicate.

```{code-cell} ipython3
import jax

x_key, noise_key = jax.random.split(jax.random.key(42))
x = jax.random.normal(x_key, (3,))
noise = jax.random.normal(noise_key, (3,))
```

```{code-cell} ipython3
{"x": x, "noise": noise}
```

<a id="practical-checklist"></a>

## Record the environment

Capture versions and numerical settings from the environment actually running
the analysis. Keep the source code and input data alongside this information.

```{code-cell} ipython3
import importlib.metadata
import platform

import pandas as pd

packages = ["liesel", "jax", "jaxlib", "tfp-nightly", "blackjax"]
environment = {
    "Python": platform.python_version(),
    "Platform": platform.platform(),
    "Backend": jax.default_backend(),
    "64-bit enabled": jax.config.jax_enable_x64,
    "PRNG implementation": jax.config.jax_default_prng_impl,
    **{name: importlib.metadata.version(name) for name in packages},
}
```

```{code-cell} ipython3
pd.Series(environment, name="Value").to_frame()
```

Also record the exact source revision if using development code, package lock
file, model and sampler settings, and data preparation. Save posterior draws
and diagnostics for a published analysis; rerunning code is not a substitute
for preserving the results being reported.

<a id="gpu-non-determinism"></a>
<a id="non-reproducibility-across-systems"></a>

## Understand the limits

Changing numerical libraries, hardware, compiler settings, or evaluation order
can change floating-point calculations. In MCMC, small differences can affect
adaptation or acceptance decisions and lead to different trajectories. CPU
execution is not a promise of bitwise equality across systems.

The [Stan reproducibility discussion](https://mc-stan.org/docs/reference-manual/reproducibility.html)
also distinguishes repeating a program in a fixed computational environment
from statistical reproducibility. When comparing runs in different environments,
assess posterior quantities relative to their Monte Carlo uncertainty and check
{doc}`../../goose-diagnostics`; do not require identical individual draws.

<a id="see-also"></a>

For this documentation, the repository lockfile and
{download}`build instructions <../../../README.md>`
specify the tested environment. See the
[JAX random configuration reference](https://docs.jax.dev/en/latest/config_options.html)
for PRNG options.
