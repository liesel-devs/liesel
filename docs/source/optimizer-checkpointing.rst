Pausing and checkpointing optimization
======================================

:meth:`liesel.optim.OptimEngine.fit` can return control after a chosen number of
epochs while retaining the optimizer state needed to continue the same run.
Construct an engine with ``optim.build_engine()`` when using
:class:`~liesel.optim.LieselOptim`.

Inspect and continue
--------------------

.. code-block:: python

   engine = optim.build_engine()
   first = engine.fit(pause_after=100)
   first.plot_loss()

   second = engine.fit(checkpoint=first.checkpoint, pause_after=100)
   second.plot_loss()  # History includes both segments.

   final = engine.fit(checkpoint=second.checkpoint)

``pause_after`` limits additional epochs in this call. The stopper alone controls
the total budget. To extend an exhausted budget, set ``engine.stopper.epochs``
before resuming. Automatic early stopping still applies. Calling ``fit()`` without
a checkpoint starts a fresh run.

The returned :class:`~liesel.optim.state.OptimResult` has a ``status`` of ``"paused"``,
``"max_epochs"``, ``"early_stopping"``, or ``"nan"``. A stopping condition takes
precedence over a coinciding pause boundary. A NaN result has ``checkpoint=None``;
use a previously retained or saved checkpoint for recovery.

Results contain cumulative history and active duration, excluding time spent
paused. Continuing leaves earlier results unchanged. Result and checkpoint share
their history arrays; retaining checkpoints also retains optimizer and runtime
state. Treat checkpoint contents as read-only. Extending history or retaining
many snapshots can require additional memory.

Recover after an HPC timeout
----------------------------

Reconstruct the same model, data, and optimization settings, then use the same
call on the initial run and on subsequent job allocations:

.. code-block:: python

   result = engine.fit(checkpoint="optim.pkl", checkpoint_every=10)

A missing file starts a persistent run. An existing file resumes that run and
receives subsequent saves. Invalid or incompatible files raise an error. Choose
a different path for a new experiment. The parent directory must exist.

Saving happens every ten completed epochs by default, independently of progress
display, and also at deliberate pauses and normal completion. Each successful
write atomically replaces the previous file. A failed write stops fitting and
preserves the previous file. A NaN failure also preserves the previous file.
Ctrl+C, crashes, and timeouts recover from the last successful save; work after
that checkpoint must be repeated. If no checkpoint was written, recovery starts
fresh. Use one writer per checkpoint path.

Recovery can use another compute node; saved arrays load onto the current JAX
device. Results need not be bitwise identical across different hardware.

Model identity and compatibility
--------------------------------

Restoration checks state structure, shapes, dtypes, and batch configuration.
It does not fingerprint data, loss functions, or optimizer settings: callers must
keep these consistent. Optimized parameters and variables in data splits need
stable names, shapes, and dtypes. With the built-in negative log-probability loss
and optimizers, anonymous internal nodes and constants do not need stable names:
their read-only evaluation state is rebuilt from the caller's model.

For weighted minibatches, saved sampling probabilities take precedence over newly
supplied weights. Reconstruct the same batch groups and sampling mode; the recovered
run continues with its original probabilities and random state.

Custom losses and optimizers can evolve ``carry.model_state``. For these, the
checkpoint retains that state and checks its keys, structure, shapes, and dtypes
against the reconstructed engine. This includes subclasses of the built-ins,
whose behavior may differ. Custom state must be pickle-compatible for disk saves.

Resumption requires identical versions of Liesel, JAX, jaxlib, Optax, and NumPy by
default. To attempt recovery across versions, pass
``allow_version_mismatch=True`` to ``fit()``. This warns about version differences
and leaves structural checks enabled; it does not guarantee compatibility.

Legacy EMA checkpoints that stored a numerator and accumulated weight are
converted to the normalized EMA when loaded. Conversion preserves the saved EMA
value but cannot undo previous rounding errors in that value, history, or stopping
state. Subsequent updates use the more stable calculation, so results can differ
from continuation with the old implementation. Newly created checkpoints retain
the EMA's rounding compensation and continue the same calculation as an
uninterrupted run under the same runtime and configuration.

Manual save and load
--------------------

.. code-block:: python

   from liesel.optim import OptimCheckpoint

   first.checkpoint.save("snapshot.pkl")
   snapshot = OptimCheckpoint.load("snapshot.pkl")
   result = engine.fit(checkpoint=snapshot)

Passing an object continues in memory without writing files. An object carries no
save destination. Pass a path to ``fit()`` whenever automatic disk saves are wanted.
Manual ``save()`` atomically replaces an existing file without rebinding the
object. Runtime-version checks occur when fitting resumes, so ``load()`` can also
be used to inspect a checkpoint.

The versioned checkpoint format uses Python pickle. Only load trusted files;
unpickling can execute code. Checkpoints contain state, not the model or optimizer
callables, and are intended for run recovery rather than portable model archives.
