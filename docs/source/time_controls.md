# Time-controlled MCMC sampling

Goose offers two separate wall-clock controls:

- `max_wall_time` is a safety limit. Reaching it raises `TimeoutError` and preserves
  completed work in a resumable {class}`~liesel.goose.Engine`.
- {meth}`~liesel.goose.Engine.sample_for_time` and
  {meth}`~liesel.goose.LieselMCMC.run_for_time` stop normally at a requested sampling
  horizon or when their finite epoch schedule finishes.

Neither is a hard real-time deadline. Goose checks elapsed time only after a
synchronized JIT block. A positive horizon executes at least one block when work
remains, and the run may overshoot by one block plus any required epoch-end hooks and
adaptation tuning. Smaller blocks improve timing responsiveness but add host dispatch
and device synchronization overhead; larger blocks reduce that overhead but increase
possible overshoot.

`jit_block_size` is the number of transitions **per chain** in one compiled JAX call.
It must divide every non-initial epoch duration. The
{class}`~liesel.goose.EngineBuilder` still infers the greatest common divisor of the
configured durations for ordinary untimed runs, but all wall-clock controls require
an explicit value so the responsiveness tradeoff is deliberate.

The examples below assume `import liesel.goose as gs` and an existing Liesel
{class}`~liesel.model.Model` named `model`. Every sampled parameter in `model` must
have a complete {class}`~liesel.goose.MCMCSpec` annotation because these examples use
{class}`~liesel.goose.LieselMCMC` to construct the kernels.

## Timing scope and result metadata

An Engine timer starts when its public sampling method is called. It includes lazy
JIT compilation, sampling blocks, synchronization, and required epoch finalization.
At the {class}`~liesel.goose.LieselMCMC` level, model validation, jittering, Engine
construction, cache lookup, and result serialization are outside the sampling
horizon. Call {meth}`~liesel.goose.Engine.compile` beforehand when first-use
compilation should not consume a comparative sampling budget.

{class}`~liesel.goose.SamplingResults` records the latest top-level call in
`elapsed_wall_time` and `stop_reason`:

- `"completed"`: the requested epoch or remaining finite schedule finished;
- `"wall_time_reached"`: a normal timed horizon stopped the call;
- `"max_wall_time_reached"`: the safety limit stopped the call before
  `TimeoutError` was raised;
- `None`: no top-level sampling result is recorded, including legacy result files.

These fields describe only the latest call. Stored sample and transition arrays are
authoritative for counts.

If both limits are configured, `wall_time <= max_wall_time` uses normal time-based
termination. A shorter `max_wall_time` logs a warning, records the safety stop, and
raises `TimeoutError`.

## Recover from a safety stop

{meth}`~liesel.goose.LieselMCMC.run_for_epochs` exposes the safety limit without
turning an expected timeout into a successful result:

```python
mcmc = gs.LieselMCMC(model)

try:
    results = mcmc.run_for_epochs(
        seed=1,
        num_chains=4,
        adaptation=1_000,
        burnin=200,
        posterior=2_000,
        jit_block_size=1,
        max_wall_time=60.0,
    )
except TimeoutError:
    assert mcmc.engine is not None
    partial_results = mcmc.engine.get_results()
    assert partial_results.stop_reason == "max_wall_time_reached"

    # Disable or adjust the safety limit, then resume the active epoch.
    mcmc.engine.max_wall_time = None
    mcmc.engine.sample_all_epochs()
    results = mcmc.engine.get_results()
```

Epoch start hooks are not repeated when sampling resumes. An incomplete adaptation
epoch is not tuned or finalized until all its configured transitions finish. If the
block that reaches a limit also completes an epoch, its end hooks and adaptation
tuning finish before the method returns or raises.

`mcmc.engine` is assigned before sampling and remains available after success or
`TimeoutError`; a later run replaces it. Loading an existing `save_path` sets it to
`None`. Because a retained Engine may hold JAX device memory, clear it when no longer
needed:

```python
mcmc.engine = None
```

Safety failures are not saved by `LieselMCMC`. Normal timed results are saved using
the existing `save_path` behavior.

## Time all configured phases

{meth}`~liesel.goose.LieselMCMC.run_for_time` applies one horizon to the remaining
finite schedule. Adaptation, burn-in, and posterior phases all count if the sampler
reaches them:

```python
mcmc = gs.LieselMCMC(model)
results = mcmc.run_for_time(
    wall_time=300.0,
    jit_block_size=1,
    seed=1,
    num_chains=4,
    adaptation=1_000,
    burnin=200,
    posterior=10_000,
)

assert results.stop_reason in {"wall_time_reached", "completed"}
```

The iteration arguments are ceilings. When the schedule finishes before the horizon,
the call returns `"completed"`; Goose does not repeat any epoch to fill the time.

## Time only the posterior phase

Phase selection comes from the caller's schedule. To exclude adaptation and burn-in
from a timed budget, complete them on an explicit Engine, append a compatible
posterior epoch, and then start the timed call:

```python
builder = gs.LieselMCMC(model).get_engine_builder(seed=1, num_chains=4)
builder.add_adaptation(1_000)
builder.add_burnin(200)
builder.jit_block_size = 1

engine = builder.build()
engine.sample_all_epochs()
engine.append_epoch(
    gs.EpochConfig(gs.EpochType.POSTERIOR, 10_000, 1, None)
)
engine.sample_for_time(300.0)
results = engine.get_results()
```

An appended duration must be divisible by the Engine's fixed `jit_block_size`.

## Compare externally managed MAMBA arms

Liesel provides the timed Engine operation, but it does not create, score, prune, or
run MAMBA arms in parallel. Keep that orchestration in the tuning code. The following
is application-owned pseudocode, not a Liesel MAMBA API. It assumes a collection of
hyperparameter `configurations`, a sequence of per-arm `stage_budgets`, and
application functions for building Engines, computing KSD scores, and selecting
survivors. `make_engine()` must configure an explicit `jit_block_size` that divides
every non-initial epoch duration so `sample_for_time()` is valid. Compile every arm
before starting comparable budgets:

```python
# Application-owned pseudocode; these names are not provided by Liesel.
active_arms = [make_engine(config) for config in configurations]

for engine in active_arms:
    engine.compile()

for stage_budget in stage_budgets:
    # Every arm active in this stage receives the same additional time budget.
    for engine in active_arms:
        engine.sample_for_time(stage_budget)

    scores = [
        kernel_stein_discrepancy(engine.get_results()) for engine in active_arms
    ]
    active_arms = select_best_arms(active_arms, scores)
```

Compilation does not advance sampler state and is idempotent for an unchanged
signature. It prepares the exact sampling executable retained by the Engine, but
first dispatch, device allocation, and hardware-cache effects may still occur inside
the timed call. Apply the same preparation policy to every arm.
