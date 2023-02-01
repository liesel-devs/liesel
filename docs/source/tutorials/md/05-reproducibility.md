
# Reproducibility

## PRNG seeding

Liesel uses [JAX’ splittable Threefry counter-based
PRNG](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers),
which does not have a global state, so you always need to seed it
explicitly, for example when initializing an
[`liesel.goose.EngineBuilder`](https://liesel-devs.github.io/liesel/liesel/goose.html#EngineBuilder).
However, even with the same seed, reproducible results cannot be
guaranteed across different systems or even for separate runs of the
same program on the same hardware.

## GPU non-determinism

Floating point operations on GPUs are generally not completely
deterministic, and even though JAX takes [some
precautions](https://github.com/google/jax/pull/4824#issuecomment-966514092)
against this surprising behavior, full reproducibility on GPUs should
not be assumed. See also [this (unmerged) page in the JAX
documentation](https://github.com/google/jax/blob/gpu-determinism-note/docs/gpu_determinism.rst).

## Non-reproducibility across systems

[In our
experience](https://github.com/blackjax-devs/blackjax/issues/181),
results from Liesel, BlackJAX and JAX may differ across systems, even if
the exact same code is run on the CPU. Following [the Stan
documentation](https://mc-stan.org/docs/reference-manual/reproducibility.html),
we expect reproducibility only on the CPU and only if all of the
following components are identical:

-   the Liesel version,
-   the Python version,
-   the versions of all libraries Liesel depends on,
-   the operating system version,
-   the computer hardware including CPU, motherboard and memory,
-   the compilers, including versions, flags and libraries, used to
    build Python and all libraries Liesel depends on,
-   the program, including the seed, initialization and data.

## See also

-   [The PyTorch docs on
    reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
-   [The Stan docs on
    reproducibility](https://mc-stan.org/docs/reference-manual/reproducibility.html)
