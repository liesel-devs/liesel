
# Reproducibility

## PRNG seeding

Liesel uses [JAX’s functional pseudo-random number
generation](https://docs.jax.dev/en/latest/random-numbers.html). JAX
does not use a single global random state; random numbers are generated
from explicit PRNG keys. In the high-level Goose workflow, this is
usually handled by passing an integer seed to
{meth}`~.goose.LieselMCMC.run_for_epochs`:

``` python
results = gs.LieselMCMC(model).run_for_epochs(
    seed=1,
    num_chains=4,
    adaptation=1000,
    posterior=1000,
)
```

If you work with the lower-level {class}`~.goose.EngineBuilder`, you
also provide the seed when initializing the builder.

Current JAX versions use typed PRNG keys created with
`jax.random.key(seed)`. The older `jax.random.PRNGKey(seed)` API is
still widely seen in existing code, but the typed-key API is the current
interface. By default, JAX uses the `"threefry2x32"` PRNG
implementation, but this is configurable through JAX’s
`jax_default_prng_impl` setting. If exact reproducibility matters,
record the PRNG implementation together with the integer seed.

Even with the same seed, reproducible results cannot be guaranteed
across different systems or even for separate runs of the same program
on the same hardware.

## GPU non-determinism

Floating point operations on GPUs and TPUs are not always bitwise
deterministic. Different devices, kernels, compiler versions, or
evaluation orders can lead to small numerical differences. In MCMC,
small numerical differences can occasionally change adaptation,
acceptance decisions, or the trajectory of a chain. For bitwise
reproducibility, prefer running on the CPU and avoid changing JAX, XLA,
or hardware configuration between runs.

## Non-reproducibility across systems

[In our
experience](https://github.com/blackjax-devs/blackjax/issues/181),
results from Liesel, BlackJAX and JAX may differ across systems, even if
the exact same code is run on the CPU. Following [the Stan
documentation](https://mc-stan.org/docs/reference-manual/reproducibility.html),
we expect bitwise reproducibility only if all of the following
components are identical:

- the Liesel version,
- the Python version,
- the versions of JAX, jaxlib, TensorFlow Probability, BlackJAX, NumPy,
  SciPy, pandas, and all other relevant libraries,
- the operating system version,
- the computer hardware including CPU, motherboard and memory,
- the compilers, including versions, flags and libraries, used to build
  Python and all libraries Liesel depends on,
- the JAX backend and configuration, including whether 64-bit mode is
  enabled and which PRNG implementation is used,
- the program, including the seed, initialization, data, sampler
  configuration, and number of chains.

## Practical checklist

For reproducible analyses, we recommend recording at least the following
information:

- the exact code and data used for the analysis,
- all model initial values and jitter settings,
- the seed passed to {class}`~.goose.LieselMCMC` or
  {class}`~.goose.EngineBuilder`,
- the number of chains, adaptation iterations, posterior iterations,
  thinning, and stored positions,
- the installed versions of Liesel and its numerical dependencies,
- the JAX backend (`cpu`, `gpu`, or `tpu`), the `jax_enable_x64`
  setting, and the `jax_default_prng_impl` setting,
- the operating system and hardware.

For publication or long-term archiving, store the posterior samples and
the exact environment specification, for example a lock file or a
container image.

## See also

- [The JAX docs on pseudo-random
  numbers](https://docs.jax.dev/en/latest/random-numbers.html)
- [The JAX docs on PRNG
  configuration](https://docs.jax.dev/en/latest/config_options.html)
- [The PyTorch docs on
  reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [The Stan docs on
  reproducibility](https://mc-stan.org/docs/reference-manual/reproducibility.html)
