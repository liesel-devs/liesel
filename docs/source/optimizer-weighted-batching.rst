Weighted minibatch sampling
===========================

Use ``sampling_weights`` on :class:`~liesel.optim.Batches` to make some training
observations more likely to appear in a minibatch. ``NegLogProbLoss`` corrects
likelihood contributions automatically, preserving the original objective in
expectation. Sampling priorities do not change the importance of observations in
the statistical model.

Configuring a run
-----------------

Construct batches explicitly and pass them to :class:`~liesel.optim.OptimEngine`:

.. code-block:: python

   import jax.numpy as jnp
   import optax
   import tensorflow_probability.substrates.jax.distributions as tfd
   import liesel.model as lsl
   import liesel.optim as opt

   loc = lsl.Var.new_param(
       jnp.array(0.0), lsl.Dist(tfd.Normal, 0.0, 1.0), name="loc"
   )
   y = lsl.Var.new_obs(
       jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 3.0]),
       lsl.Dist(tfd.Normal, loc, 1.0),
       name="y",
   )
   model = lsl.Model([y])
   split = opt.PositionSplit.from_model(model, position_keys=["y"])
   batches = opt.Batches(
       ["y"],
       axis_size=split.train_axis_size,
       batch_size=2,
       sample_with_replacement=True,
       sampling_weights=[1, 1, 1, 1, 1, 5],
   )
   loss = opt.NegLogProbLoss(model, split, scale=True)
   engine = opt.OptimEngine(
       loss=loss,
       batches=batches,
       optimizers=[opt.Optimizer(["loc"], optax.adam(0.01))],
       stopper=opt.Stopper(epochs=100, patience=20),
       seed=42,
       initial_state=model.state,
       loss_monitor="train_full_data",
       show_progress=False,
   )
   result = engine.fit()

Weights must be finite, strictly positive, and have length ``axis_size``. They
are normalized internally and exposed as ``batches.sampling_probabilities``.
Supply them in the **training split's current order**, after splitting or
shuffling. One vector applies to all aligned arrays in a group; include responses
and their per-observation covariates together in ``position_keys``.

Sampling is independent with replacement: duplicates within a batch are allowed,
even when ``batch_size == axis_size``. Epoch length and
:class:`~liesel.optim.BatchManager` epoch policies remain unchanged. An epoch does
not guarantee that every observation appears. A manager can combine weighted
groups with different probability vectors and ordinary uniform groups.

Probabilities are fixed throughout a run. On checkpoint recovery, the saved
probabilities and alias table take precedence over newly supplied weights. Batch
structure and sampling mode must remain compatible. Adaptive priorities,
weighted sampling without replacement, and automatic weight routing through
``LieselOptim`` are outside this API.

Sampling and numerical precision
--------------------------------

Weighted groups prepare a `Vose alias table
<https://www.keithschwarz.com/darts-dice-coins/>`_ once on the host using NumPy
float64 arithmetic. Each draw selects a uniform integer table entry and flips
a biased coin to choose that entry or its stored alternative. Preparation and
table storage are linear in the observation count; draws do not scan the weight
vector. The engine draws an epoch's indices together, then slices them into
minibatches, reusing the table across epochs.

The table occupies 16 bytes per observation, in addition to the probability
vector (4 bytes per observation, or 8 with float64) and the epoch's indices.
Thus one million observations require about 16 MB for the alias table.
Integer rejection sampling avoids modulo bias when selecting entries. Coin
decisions use random integer bits, drawing additional words when needed, rather
than comparing a float32 uniform draw with a tiny threshold. Sampling uses JAX
keys throughout and works with JAX's 64-bit mode disabled.

Coin decisions are exact for the stored binary thresholds, assuming uniform
independent random bits. Normalization and table construction still involve
ordinary floating-point rounding; this is not exact arithmetic for the original
weight ratios. Probabilities and importance corrections use at least float32,
including when input weights are float16. Float64 inputs retain float64 when
JAX's 64-bit mode is enabled. There is no probability floor or correction cap;
representability checks still apply.

Checkpoints save the alias table along with the probabilities and random state,
allowing continuation to match an uninterrupted run under the same supported
runtime. Weighted checkpoints require alias state to resume, even with
``allow_version_mismatch=True``.

How correction works
--------------------

For a group of size :math:`N`, a selected index :math:`i` with probability
:math:`p_i` receives an extra factor :math:`1/(N p_i)`. This is applied before
summing likelihood values, in addition to the existing
``sample_size / batch_sample_size`` scale (ordinarily :math:`N/b`). Thus the
ordinary corrected estimate is :math:`\frac{1}{b}\sum_{i\text{ drawn}}\ell_i/p_i`.
Uniform weights reproduce the existing configured objective.

Priors, unbatched likelihood terms, and overall loss normalization remain
unchanged. Full-training and validation evaluations use the full relevant split,
without sampling correction. No probability floor or correction cap is imposed;
weights that produce zero or nonfinite probabilities or factors in the working
dtype are rejected.

Likelihood axes
---------------

The correction must act along the likelihood dimension that enumerates the
sampled observations. Standard distribution semantics determine this after
trailing event reduction and leading broadcasting. Examples include data shaped
``(n, k)`` with multivariate likelihood shaped ``(n,)``, or data shaped
``(k, n, d)`` batched along axis 1 with likelihood shaped ``(k, n)``.

For custom reductions or transpositions, declare the likelihood axis explicitly:

.. code-block:: python

   batches = opt.Batches(
       ["y"], axis_size=n, batch_size=128,
       batch_axes={"y": 1},        # data has shape (k, n)
       likelihood_axes={"y": 0},   # custom log_prob has shape (n,)
       sample_with_replacement=True,
       sampling_weights=weights,
   )

``likelihood_axes`` is keyed by observed-variable name, including when
``position_keys`` uses a value-node name. A declared axis must enumerate the
sampled units in their batch order. Its length must equal ``batch_size``. Shape
coincidences alone cannot establish this correspondence. A scalar likelihood
already summed over observations cannot be corrected and is rejected; keep
``Dist.per_obs=True``. Sampling within a multivariate event does not produce a
per-observation likelihood and cannot be corrected this way.

Other losses
------------

Custom losses can call
``carry.batches.scaled_log_lik(model, updated_state, batch_index=carry.i_batch)``.
For other model interfaces, apply ``batches.correction_factors(batch_index)``
along the known likelihood axis before reducing, and retain the ordinary
``batch_sample_scale``. This method returns only the extra per-index factors;
uniform groups return ones. ``BatchManager.correction_factors`` returns a tuple
in child order. Applying these factors is the custom loss's responsibility.
Models with a custom aggregate ``log_lik_node`` also require a custom loss:
the built-in helper cannot infer how to decompose that aggregation.
