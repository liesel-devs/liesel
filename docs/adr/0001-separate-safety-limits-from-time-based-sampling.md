# Separate safety limits from time-based sampling

## Context

Unexpectedly slow transitions need an exceptional safety limit, while algorithms
such as MAMBA need normal sampling for a compute-time budget. Both controls can only
observe completed JAX work at host synchronization points. They must preserve
adaptation state and completed samples so an interrupted epoch can resume.

## Decision

Liesel implements the controls as separate Engine operations:

- `max_wall_time: float | None` is configured through `EngineBuilder`, applies to
  each top-level Engine sampling call, and is exposed by both `LieselMCMC` run
  methods. Reaching it raises `TimeoutError` at a completed JIT-block boundary and
  leaves the Engine resumable. It is mutable on an Engine.
- `Engine.sample_for_time(wall_time)` and `LieselMCMC.run_for_time(wall_time)` stop
  normally when the horizon or finite epoch schedule is reached. A positive horizon
  executes at least one block when sampling work remains. `LieselMCMC.run_for_time()`
  mirrors the epoch scheduling and storage arguments of `run_for_epochs()` and
  requires `jit_block_size`.

Both limits are soft and checked after synchronized JIT blocks. The canonical
`jit_block_size` is fixed on an Engine and must be explicit for either timing
control; ordinary untimed sampling retains the previous GCD inference. The legacy
`Engine.__init__(jitted_sample_duration=...)` spelling remains accepted for
compatibility.

A partial epoch remains active at the last completed block. Later sampling resumes
it without repeating its start hook, and epoch-end hooks or adaptation tuning run
only when its configured transition count is complete. If the limiting block
completes an epoch, Engine synchronizes its finalization before stopping and includes
that work in elapsed time.

The Engine timer includes lazy JIT compilation and synchronized sampling.
`LieselMCMC` construction and result serialization are outside the sampling horizon.
`Engine.compile()` compiles and retains the exact executable without advancing state,
so external tuners can prepare comparable Engines before starting their clocks.

`LieselMCMC.engine` retains the most recently built Engine after successful or failed
sampling; a new run replaces it, and loading cached results sets it to `None`.
`SamplingResults` records `elapsed_wall_time` and the latest `stop_reason`:
`"completed"`, `"wall_time_reached"`, or `"max_wall_time_reached"`. Stored arrays
remain authoritative for counts.

When both thresholds are present, the smaller threshold controls. Equal thresholds
use normal time-based completion; `sample_for_time()` warns when `max_wall_time` is
shorter. Normal timed results use the existing `save_path` behavior, while safety
failures are not saved.

## Consequences

Wall-clock responsiveness is chosen explicitly through `jit_block_size`: smaller
blocks improve precision but add dispatch and synchronization overhead. No running
JAX block is interrupted, and MAMBA arm creation, scoring, pruning, and parallelism
remain application responsibilities. Callers that want to time only later phases
must finish earlier phases before starting the timed call.

See the [time-controlled sampling guide](../source/time_controls.md) for operational
examples and recovery workflows.
