Pause and resume a fit
======================

Build an engine to pause a run or save progress. Here, ``optim`` is a configured
:class:`liesel.optim.LieselOptim`.

Pause in memory
---------------

.. code-block:: python

   engine = optim.build_engine()
   first = engine.fit(pause_after=100)
   first.plot_loss_overview()
   result = engine.fit(checkpoint=first.checkpoint)

``pause_after`` limits additional epochs in that call. Early stopping still
applies. To extend the total budget, change ``engine.stopper.epochs`` before
resuming. Calling :meth:`fit() <liesel.optim.LieselOptim.fit>` without a checkpoint starts a new run.

The result's ``status`` tells you why fitting stopped: ``"paused"``,
``"max_epochs"``, ``"early_stopping"``, ``"numerical_failure"``, or ``"nan"``.
A numerical-failure or NaN result has no
checkpoint; recover from an earlier saved checkpoint instead.

Save progress to disk
---------------------

Run the same call on the first job and after an interruption:

.. code-block:: python

   result = engine.fit(checkpoint="optim.pkl", checkpoint_every=10)

A missing file starts a new run. An existing file resumes it. The engine saves
every ten epochs here, plus at deliberate pauses and normal completion. A crash
or timeout loses work since the last successful save. Failed writes and numerical
failures leave the previous file intact.

Use a different path for a new experiment and only one writer per path. The
parent directory must exist.

If the checkpoint permits no additional epochs under the current stopper
settings, :meth:`fit <liesel.optim.LieselOptim.fit>` warns and returns its result without further optimization.
This applies to both paths and in-memory checkpoints. Increasing
``engine.stopper.epochs`` allows continuation only if early stopping does not
apply. Use a new path or call :meth:`fit() <liesel.optim.LieselOptim.fit>` without a checkpoint for a fresh run,
including when changing data or optimizer settings for a new experiment.

Resume safely
-------------

* Recreate the same model, data, and optimizer settings. Checks cover structure,
  names, shapes, dtypes, and batching, but cannot detect changed data or learning
  rates. Weighted sampling resumes with its saved probabilities.
* Keep the same Liesel, JAX, jaxlib, Optax, and NumPy versions. The option
  ``allow_version_mismatch=True`` attempts recovery across versions without
  promising compatibility. Different hardware can also change results.
* Load only trusted files: checkpoints use Python pickle, which can execute code.
  They save run state, not model code, and are meant for recovery rather than
  long-term model storage.

History and monitoring continue across pauses. Earlier results stay unchanged;
keeping many snapshots uses extra memory. Treat checkpoint contents as read-only.
Custom losses must keep their model state and loss-state PyTrees compatible
and serializable with pickle. Committed and best loss states survive recovery.
For :class:`LaplaceLoss <liesel.optim.LaplaceLoss>`, keep the latent parameters and inner-solver controls unchanged;
checkpoint recovery checks these settings as well as state structure and dtype.

For manual snapshots, use :meth:`liesel.optim.OptimCheckpoint.save` and
:meth:`liesel.optim.OptimCheckpoint.load`. Passing a checkpoint object resumes
in memory; pass a path to :meth:`fit() <liesel.optim.LieselOptim.fit>` for automatic disk saves. See
:meth:`liesel.optim.OptimEngine.fit` for all recovery options.
