.. _optimizer-data-flow:

Splitting, batching, and loss scaling
=====================================

A split decides which data belong to training, validation, and testing.
Batches take smaller pieces of the training data for each update. These are
separate choices: shuffling batches never changes the held-out data.
Both operations require :ref:`likelihood contributions that can be evaluated
row by row <optimizer-row-wise>`. The optimizer does not check this property.

.. _optimizer-split-overview:

Keep matching rows together
---------------------------

Split responses and their covariates together. For an existing ``model``:

.. code-block:: python

   import optax

   import liesel.optim as opt

   split = opt.PositionSplit.from_model(
       model,
       position_keys=["X", "y"],
       validate_axis_share=0.2,
       test_axis_share=0.1,
       seed=42,
   )

This puts 70% of rows in training, 20% in validation, and 10% in testing, subject
to rounding. Splits with validation or test data shuffle by default; set
``shuffle=False`` for an ordered split. Without holdouts, splits preserve the
original row order and ignore the seed, even with ``shuffle=True``. This is also
the behavior of ``LieselOptim``'s automatic full-training split.

The split's ``seed`` chooses which rows go into each part. The seed passed to
``LieselOptim`` controls batch shuffling or random batch sampling during fitting.
Custom losses and optimizers can also use its random key through ``carry.key``.
Both seeds default to ``0``, making split and fitting randomness reproducible in
the same environment. Choose other integer seeds for different runs. Explicit
``seed=None`` opts into Unix time in whole seconds, so calls in the same second
can share a seed. Starting parameter values come from the model; seed any random
data or starting values separately.

For separate groups, make the grouping explicit:

.. code-block:: python

   split = opt.PositionSplitManager.from_model(
       model,
       position_keys=[["X_a", "y_a"], ["X_b", "y_b"]],
       validate_axis_share=0.2,
       seed=42,
   )

Arrays within a group share row indices. Different groups split independently,
even if their lengths happen to match. Flat or omitted ``position_keys`` group
observed arrays by length; use nested groups when equal length does not mean
matching rows. Every group must have validation data if any group does.

Automatic grouping requires an observed likelihood in each group. If a group
has none, specify its row keys as an explicit nested group, or mark shared data
as passthrough. Matching lengths alone do not establish row alignment: lookup
tables must be passthrough even when their length matches a response.

Set ``split_axes`` when creating a split for observations on an axis other than
zero. A value of ``None`` keeps a shared table unchanged in every split and out
of automatic batches. Keep per-observation covariates, weights, and offsets with
the response.
Set the corresponding ``batch_axes`` when creating batches too.

For a model with a lookup table ``z`` indexed by row-level group IDs, construct
``PositionSplit.from_model(model, split_axes={"z": None})`` and pass the result
as ``LieselOptim(..., split=split)``. This also works for scalar constants.
Automatic setup raises an informative error for scalars or inferred groups
without a likelihood instead of guessing how to split them.

``PositionSplit`` holds the split data. ``Split`` holds reusable row indices;
call ``split_position()`` to apply them. Manager classes handle several groups.
See :meth:`~liesel.optim.PositionSplit.from_model` for the factory options.

For observed distributions with ``per_obs=False``,
automatic sample-size inference is unavailable. Construct the split explicitly:

.. code-block:: python

   split = opt.PositionSplit.from_model(
       model, infer_sample_sizes=False, multi_size="manager", shuffle=False
   )
   optim = opt.LieselOptim(
       model, split=split, optimizers=optax.adam(0.01),
       loss_monitor="train_full_data",
   )

This chooses split-axis counts for scaling. Alternatively, supply effective
``sample_sizes`` to the split factory. Setting ``scale_loss=False`` on
``LieselOptim`` only disables final loss normalization; it does not disable
split inference or specify batch scaling.

Custom aggregate likelihood, prior, or probability nodes require a custom
:class:`~liesel.optim.Loss` and an explicit split. A manual split specifies data
grouping and scaling; it does not change the objective used by
:class:`~liesel.optim.NegLogProbLoss`. The built-in loss accepts the standard sums
of observed distribution factors and parameter priors, including weak observed
variables. Other distribution factors must be classified appropriately or handled
by a custom loss.

.. raw:: html

   <iframe
     class="interactive-visualization"
     data-visualization="split"
     src="_static/visualizations/split-api-overview.html"
     title="Interactive overview of the Liesel split API"
     loading="lazy"
     sandbox="allow-scripts allow-same-origin">
   </iframe>

`Open the split API overview in a separate page
<_static/visualizations/split-api-overview.html>`__.

.. _optimizer-batch-manager-overview:

Batch one or several groups
---------------------------

Create batches from the training split, then pass both to ``LieselOptim``:

.. code-block:: python

   batches = opt.Batches.from_split(split, batch_size=32)
   result = opt.LieselOptim(
       model,
       optimizers=optax.adam(0.01),
       split=split,
       batches=batches,
       loss_monitor="validation",
       seed=43,
   ).fit()

For default batch settings, pass ``batch_size=32`` directly to ``LieselOptim``.
This calls ``Batches.from_split(split, batch_size=32)``. Omit both ``batch_size``
and ``batches`` to use all training data in each update. For custom shuffling,
axes, or epoch policy, create ``batches`` explicitly instead.

The engine checks each training array's length along its batch axis before
fitting. Splitting and batching may use different axes. When holding out rows
on the same axis, build batches from the split so they use the training size.

Only complete batches are used. With shuffling, the leftover rows can change
between epochs. For multiple groups, a :class:`~liesel.optim.BatchManager`
supplies one batch from each group at every update.

The ``Batches.from_split`` default, ``epoch_size="max"``, follows the group with
the most batches. Smaller groups start another shuffled pass as needed. Other
choices are ``"min"`` (stop with the shortest group), ``"strict"`` (require equal batch
counts), or a positive number of steps. Direct managers default to ``"strict"``.

Pass ``sample_with_replacement=True`` to draw rows independently, allowing
duplicates. Managers also enable this for groups smaller than the requested batch
size. Such an epoch need not visit every row. For weighted sampling, pass
``sampling_weights`` in training-row order.

:meth:`~liesel.optim.Batches.from_split` also accepts custom sample sizes and
likelihood axes. :meth:`~liesel.optim.BatchManager.from_split` accepts the same
options and always returns a manager. For different settings per group, create
each child with ``Batches.from_split`` and combine them with ``BatchManager``.

.. raw:: html

   <iframe
     class="interactive-visualization"
     data-visualization="batch"
     src="_static/visualizations/batch-manager-overview.html"
     title="Interactive BatchManager epoch overview"
     loading="lazy"
     sandbox="allow-scripts allow-same-origin">
   </iframe>

`Open the BatchManager overview in a separate page
<_static/visualizations/batch-manager-overview.html>`__.

.. _optimizer-row-wise:

When row batching is valid
--------------------------

Selecting data rows must preserve each selected row's likelihood contribution.
This requirement applies to weak observations and to strong observations whose
distribution inputs are computed from the data. Graph dependencies identify
which group supplies a factor; matching array shapes, explicit groups and
``likelihood_axes`` do not establish that the factor can be evaluated row by row.
The optimizer does not verify this requirement.

For example, recomputing ``y[1:] - phi * y[:-1]`` or
``y - phi * jnp.roll(y, 1)`` after selecting arbitrary rows changes which values
are neighbors. Shuffled holdouts break these lag relationships even if fitting
then uses full-data batches. Turning off batch shuffling does not repair an
earlier split or restore the context lost at batch boundaries.

To evaluate such a model on its original ordered data, use no holdouts and
``batch_size=None``. When a conditional model can instead use fixed response/lag
pairs, prepare those pairs from the original series and keep them in one group.
The following example compares a batch's conditional likelihood contributions
with those of the corresponding full-data rows:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import tensorflow_probability.substrates.jax.distributions as tfd
   import liesel.model as lsl
   import liesel.optim as opt

   series = jnp.array([0.0, 0.7, 0.4, -0.1, 0.3, 0.9, 0.5])
   lag = lsl.Var.new_obs(series[:-1], name="lag")
   phi = lsl.Var.new_param(0.5, name="phi")
   mean = lsl.Var.new_calc(lambda lag, phi: phi * lag, lag, phi)
   response = lsl.Var.new_obs(
       series[1:], lsl.Dist(tfd.Normal, mean, 1.0), name="y"
   )
   ar_model = lsl.Model([response])
   ar_split = opt.PositionSplit.from_model(
       ar_model, position_keys=[["y", "lag"]]
   )
   ar_batches = opt.Batches.from_split(ar_split, batch_size=2).start_epoch(
       jax.random.key(1)
   )
   batch = ar_batches.get_batched_position(ar_split.train, 0)
   state = ar_model.update_state(batch, ar_model.state)
   full_terms = ar_model.state["y_log_prob"].value
   expected = -ar_batches.batch_sample_scale * jnp.sum(
       full_terms[ar_batches.batch_indices[0]]
   )
   actual = -ar_batches.scaled_log_lik(ar_model, state, batch_index=0)
   print(bool(jnp.allclose(actual, expected)))

The output is ``True``: the selected pairs retain their full-data likelihood
contributions.

These are conditional likelihood terms with fixed observed lags. Choose temporal
holdouts according to the intended prediction task; this example does not define
a forecasting-validation procedure. Dependencies that require other batching or
held-out semantics need a custom :class:`~liesel.optim.Loss` and suitable data
configuration. Substituting a custom loss while retaining incorrect row slicing
does not repair the objective.

Weak observed variables
-----------------------

Weak observed variables, such as copula observations computed from marginal PITs,
are supported. The model factories select strong observed inputs as writable data;
weak values and their likelihoods are recomputed as parameters or data change.
Automatic grouping also supports inputs whose only likelihood comes from a weak
observed variable; the strong inputs need not have their own distributions.
Explicitly selecting a weak variable or its value node for splitting or batching
raises an error: select its strong source data instead.

Keep aligned inputs of a weak observation in one group, for example
``position_keys=[["x1", "x2"]]`` for two copula margins. The weak likelihood then
shares that group's split, scaling, and weighted sampling corrections. It also
contributes to validation and test scores when its source group is held out.
A factor spanning independently split or batched groups is rejected.

Graph dependencies identify which group supplies a weak observation, but do not
establish arbitrary transformation or row semantics. Supply explicit grouping
and axes when needed. In weighted batches, a multidimensional weak likelihood
requires ``likelihood_axes={"copula": 0}`` (with the appropriate factor name and
likelihood axis); the strong input's axis is not assumed to be its likelihood axis.
The :ref:`row-wise requirement <optimizer-row-wise>` also applies here. Use a
custom loss and suitable data configuration when the transformation does not
preserve the group's likelihood contributions.

Copula likelihoods are sensitive to starting values: PITs can round to 0 or 1,
producing infinite inverse-CDF values and non-finite likelihoods. Initialize
marginal parameters near the data, for example with a margins-only fit. Higher
precision can help but cannot guarantee finite tails. For a configured
``LieselOptim`` builder, enable first-NaN reproduction capture on its engine:

.. code-block:: python

   engine = builder.build_engine()
   engine.debug_nans = True
   result = engine.fit()
   debug_info = result.nan_debug

``debug_nans`` is an engine setting, not a ``LieselOptim`` constructor argument.
The captured information helps reproduce the first detected NaN; it does not
correct poor starting values.

Weak parameters and priors
--------------------------

A prior may be attached to a weak parameter computed from a strong source variable.
That prior remains in the default loss, including its derivatives through the weak
parameter. Automatic optimizer selection cannot decide which source to estimate
and raises an informative error. Supply the strong names explicitly, for example
``optimizers=[opt.Optimizer(["source"], optax.adam(0.01))]`` or
``optimizers=[opt.LBFGS(["source"])]``. The source does not need to be marked as a
parameter. The weak parameter itself is recomputed during fitting.

.. _optimizer-likelihood-scaling:

What the loss means
-------------------

The default training loss combines likelihood and priors. A minibatch likelihood
is scaled up to represent its full training group; priors are not scaled up.
Each group gets its own factor, so different batch sizes do not change the
relative weight of the groups.

Validation and test likelihoods include only the split branches and are scaled
to the corresponding training size. Unsplit observed likelihoods contribute to
training only, including observations explicitly marked as passthrough.
``LieselOptim`` then divides losses by the total training sample size by default;
use ``scale_loss=False`` to keep the sum. Sample size counts likelihood terms,
which need not equal the number of array elements.

Validation leaves out priors by default. Use ``validation_strategy="log_prob"``
to include them. See :class:`~liesel.optim.NegLogProbLoss` for details.

.. raw:: html

   <iframe
     class="interactive-visualization"
     data-visualization="likelihood"
     src="_static/visualizations/likelihood-scaling.html"
     title="Interactive overview of likelihood scaling through splitting and batching"
     loading="lazy"
     sandbox="allow-scripts allow-same-origin">
   </iframe>

`Open the likelihood-scaling overview in a separate page
<_static/visualizations/likelihood-scaling.html>`__.

.. raw:: html

   <script>
     window.addEventListener("message", (event) => {
       if (event.data?.type !== "liesel:visualization-height") return;
       const frame = document.querySelector(
         `[data-visualization="${event.data.visualization}"]`
       );
       if (
         !frame ||
         event.source !== frame.contentWindow?.frames[0] ||
         !Number.isFinite(event.data.height) ||
         event.data.height < 0 ||
         event.data.height > 10000
       ) return;
       frame.style.height = `${Math.ceil(event.data.height) + 32}px`;
     });
   </script>
