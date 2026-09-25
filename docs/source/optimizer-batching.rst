Create minibatches
==================

Batches select smaller pieces of the training data for each update. Use them
when full-data updates are too expensive. Each selected row must retain its
:ref:`likelihood contribution <optimizer-row-wise>`; the optimizer does not
check this property. See :doc:`optimizer-loss-scaling` for how a batch
represents its full training group.

.. _optimizer-batch-manager-overview:

Batch one or several groups
---------------------------

For an existing ``model`` and a :doc:`split with validation data
<optimizer-splitting>`, create batches from its training rows and pass both
to ``LieselOptim``:

.. code-block:: python

   import optax
   import liesel.optim as opt

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
``sampling_weights`` in training-row order; see :doc:`optimizer-weighted-batching`.

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
``batch_size=None``. Dependencies that require other batching or held-out
semantics need a custom :class:`~liesel.optim.Loss` and suitable data
configuration. Substituting a custom loss while retaining incorrect row slicing
does not repair the objective.

Prepare aligned response and lag pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a conditional model with fixed observed lags, prepare response/lag pairs
from the original series. Keep both arrays in one group so selecting a response
also selects its original lag:

.. code-block:: python

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
   ar_batches = opt.Batches.from_split(ar_split, batch_size=2)

Pass ``ar_split`` and ``ar_batches`` to ``LieselOptim`` to fit this conditional
model. Choose temporal holdouts according to the intended prediction task;
this example does not define a forecasting-validation procedure.

Compare a batch with the full-data contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the model and batches above, select a batch and compare its scaled
likelihood with the corresponding terms evaluated on the original data:

.. code-block:: python

   import jax

   ar_batches = ar_batches.start_epoch(jax.random.key(1))
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

.. _optimizer-computed-data:

Batch precomputed values
-----------------------

An expensive computed variable, such as a callback-based design matrix, can be
batched by selecting its cached values. For a model with a computed ``basis``,
batch its rows alongside a raw covariate ``x2`` and the response ``y``:

.. code-block:: python

   split = opt.PositionSplit.from_model(
       model, position_keys=["basis", "x2", "y"], validate_axis_share=0.2
   )
   batches = opt.Batches.from_split(split, batch_size=32)

This trades memory for less computation: the full basis matrix must fit in memory.
Its values must remain valid as parameters change, and selecting rows must preserve
their :ref:`likelihood contributions <optimizer-row-wise>`. Raw covariates can still
be used for inexpensive JAX calculations in the same fit.

Use the computed variable's name. Transient variables and calculation-node keys
are rejected. A computed value cannot be selected together with an ancestor or
descendant data key; for example, choose either ``basis`` or the covariate it uses.

.. _optimizer-weak-observations:

Batch weak observed variables
------------------------------

Weak observed variables, such as copula observations computed from marginal PITs,
are supported. By default, model factories select strong observed inputs as data;
weak values and their likelihoods are recomputed as parameters or data change.

Mark the weak variable carrying the copula likelihood as observed, for example
``copula.observed = True``. On an existing model, use
``model.vars["copula"].observed = True``. An intermediate PIT without its own
likelihood does not need this flag. See the
:class:`~liesel.distributions.GaussianCopula` example for a complete model.

Automatic grouping also supports inputs whose only likelihood comes from a weak
observed variable; the strong inputs need not have their own distributions.
Values such as PITs that depend on fitted parameters must remain computed from
their strong source data. Fixed computed values can be selected explicitly as
:ref:`precomputed data <optimizer-computed-data>`.

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
precision can help but cannot guarantee finite tails. To investigate a failed
fit, see :ref:`optimizer-debug-nans`.

.. raw:: html

   <script src="_static/visualizations/resize-frames.js"></script>
