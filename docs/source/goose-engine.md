---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Use the engine

Use an {class}`~liesel.goose.EngineBuilder` when you need an explicit schedule
or want to add kernels yourself. For routine runs, start with
{doc}`sampling`.

## Prepare the example

These examples require the regression `model`, including its transformation
and inference specifications, from {doc}`tutorials/md/01c-transform`.

```{code-cell} ipython3
:load: _examples/goose-regression.py
:tags: [remove-cell]
```

## Build a schedule

For the `model` from {doc}`tutorials/md/01c-transform`:

```{code-cell} ipython3
import liesel.goose as gs

builder = gs.LieselMCMC(model).get_engine_builder(seed=2026, num_chains=4)
builder.positions_included = ["sigma_sq"]

builder.add_adaptation(1000)
builder.add_burnin(200)
builder.add_posterior(2000)
builder.show_progress = False
engine = builder.build()
```

```{code-cell} ipython3
engine.sample_all_epochs()
results = engine.get_results()
```

`get_engine_builder` supplies the model interface, initial state, kernels,
and configured jitter. The remaining calls choose the schedule, construct
the engine, run it, and retrieve its results.

## Customize adaptation

Use a new builder to change the fast and slow adaptation windows:

```{code-cell} ipython3
builder = gs.LieselMCMC(model).get_engine_builder(seed=2026, num_chains=4)
builder.add_adaptation(
    1000,
    init=100,
    term=100,
    base=50,
)
builder.add_posterior(1000)
builder.show_progress = False
engine = builder.build()
```

```{code-cell} ipython3
engine.sample_all_epochs()
results = engine.get_results()
```

Integer `init` and `term` values specify numbers of iterations; floats
specify fractions of the adaptation duration. `base` sets the first slow
window. See {meth}`~liesel.goose.EngineBuilder.add_adaptation` for the schedule.
Add adaptation before burnin and posterior sampling.

## Supply a log density

Goose can also sample a log density represented without a Liesel graph.
This independent example targets a standard normal variable:

```{code-cell} ipython3
import jax.numpy as jnp


def log_prob(state):
    return -0.5 * jnp.square(state["x"]).sum()


builder = gs.EngineBuilder(seed=2026, num_chains=4)
builder.set_model(gs.DictInterface(log_prob))
builder.set_initial_values({"x": jnp.array(0.0)})
builder.add_kernel(gs.NUTSKernel(["x"]))

builder.add_adaptation(1000)
builder.add_posterior(1000)
builder.show_progress = False
engine = builder.build()
```

```{code-cell} ipython3
engine.sample_all_epochs()
normal_results = engine.get_results()
```

Here `set_initial_values` broadcasts one state to all chains. For different
starts, pass state leaves with a leading chain axis and
`multiple_chains=True`. The interface provides log-density evaluation and
position updates. See {class}`~liesel.goose.ModelInterface` for the protocol.

The same engine can use other representations through
{class}`~liesel.goose.ModelInterface`. For custom update logic, start with
an {doc}`MH proposal <tutorials/md/08-custom-kernel>` before implementing a
complete kernel class.
