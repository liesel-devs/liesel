Impute missing values
=====================

This guide demonstrates a technical pattern for missing-value models: represent
missing covariate values as parameters and combine them with observed values in
a calculated variable. MCMC samples the missing values alongside the regression
parameters; optimization estimates them jointly. Run the examples in order;
the sampling and full-data optimization sections both use the same model.

.. admonition:: Scope of this example
   :class: note

   The example assumes missingness independent of the data, a fully specified
   ``Normal(0, 1)`` covariate model, and a fixed response noise scale of 0.5.
   It demonstrates implementation in Liesel, not how to choose an imputation
   strategy. Whether these assumptions
   or an estimator suit your application, and whether you can ignore its
   missingness mechanism, are outside this guide's scope.

Define a model with missing covariates
--------------------------------------

Generate a covariate, a response, and missing entries. Compute the missing
indices before fitting so their number stays fixed during JAX compilation.

.. plot::
   :context: reset
   :nofigs:
   :include-source: True
   :show-source-link: False

   import jax.numpy as jnp
   import numpy as np
   import tensorflow_probability.substrates.jax.distributions as tfd

   import liesel.model as lsl

   rng = np.random.default_rng(12)
   n = 120
   x_true = rng.normal(size=n)
   y_data = jnp.asarray(1.0 + 2.0 * x_true + rng.normal(scale=0.5, size=n))
   x_data = np.where(rng.random(n) < 0.25, np.nan, x_true)

   missing = np.flatnonzero(np.isnan(x_data))
   base = jnp.asarray(np.where(np.isnan(x_data), 0.0, x_data))
   x_missing = lsl.Var.new_param(
       jnp.zeros(len(missing)),
       lsl.Dist(tfd.Normal, 0.0, 1.0),
       name="x_missing",
   )
   x = lsl.Var.new_calc(
       lambda z: base.at[missing].set(z), x_missing, name="x"
   )

Of the covariate entries, only the missing ones are estimated. The calculation
replaces the placeholders in ``base`` and preserves every observed value. The
distribution of ``x_missing`` specifies the model for the missing covariate;
the response likelihood also informs its values. Here the covariate distribution
is fixed. If you estimate its parameters, include the observed covariates'
likelihood too.

Use ``x`` in the regression.

.. plot::
   :context: close-figs
   :include-source: True
   :show-source-link: False
   :alt: Model graph showing x_missing feeding the completed covariate x, which combines with alpha and beta in mu to determine the distribution of y.

   alpha = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="alpha")
   beta = lsl.Var.new_param(1.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="beta")
   mu = lsl.Var.new_calc(lambda a, b, x: a + b * x, alpha, beta, x, name="mu")
   y = lsl.Var.new_obs(y_data, lsl.Dist(tfd.Normal, mu, 0.5), name="y")
   model = lsl.Model([y])
   model.plot(width=8, height=6)

The graph shows how ``x_missing`` enters the regression through ``x`` and ``mu``.
The observed covariate entries stay fixed inside the calculation of ``x``.

Sample the missing values
-------------------------

Assign the continuous parameters to one NUTS kernel. Each posterior draw includes
a draw for every missing covariate entry.

.. code-block:: python

   import liesel.goose as gs

   for var in model.parameters.values():
       var.inference = gs.MCMCSpec(
           gs.NUTSKernel, kernel_group="joint", jitter_dist=tfd.Normal(0.0, 0.1)
       )

   results = gs.LieselMCMC(model).run_for_epochs(
       seed=1, num_chains=4, adaptation=1000, posterior=1000
   )
   samples = results.get_posterior_samples()
   missing_mean = samples["x_missing"].mean(axis=(0, 1))
   completed_mean = base.at[missing].set(missing_mean)

The first two sample axes are chain and draw. ``completed_mean`` contains
posterior means at the missing entries.

.. note::

   These posterior-mean imputations depend on the response. Treating them as
   observed covariates can bias later estimates involving the response and
   understates uncertainty. For later analyses, use the individual posterior
   draws of the missing values and propagate their uncertainty.

Inspect the chains with :func:`~liesel.goose.plot_trace`
and :class:`~liesel.goose.Summary` before interpreting the imputations.
For discrete missing covariates, use a suitable discrete sampler or marginalize
them; NUTS requires continuous parameters.

Optimize on the full data
-------------------------

Pass the same model to :class:`~liesel.optim.LieselOptim`. All parameters,
including ``x_missing``, are optimized together to obtain a joint maximum
a posteriori (MAP) estimate.

.. _missing-values-joint-map:

.. admonition:: Joint optimization is a different statistical target
   :class: warning

   Joint MAP differs from integrating missing values out before optimization.
   In this normal model it systematically inflates the slope magnitude. The bias
   grows with the noise level and share of missing values; for a nonzero slope,
   it does not vanish as the sample grows. Estimating the noise scale or the
   covariate distribution's scale jointly with missing values also biases those
   scales downward. Sampling the missing values with MCMC, as above, avoids this
   joint-MAP bias.

   The code demonstrates the technical pattern, not a recommendation for this
   estimator. Optimized missing values are single, shrunk imputations that
   depend on the response. Treating them as observed covariates can bias later
   estimates involving the response and understates uncertainty. For later
   analyses, use the per-draw imputations from MCMC and propagate their
   uncertainty.

.. code-block:: python

   import liesel.optim as opt

   result = opt.LieselOptim(
       model, optimizers="lbfgs", loss_monitor="train_full_data"
   ).fit()
   fitted = result.position_min_monitor
   completed_fit = base.at[missing].set(fitted["x_missing"])
   print(float(fitted["alpha"]), float(fitted["beta"]))

For these data, the joint-MAP coefficients are about 0.98 and 1.95.
``completed_fit`` contains the corresponding fitted covariate values, not
posterior draws. Fitting returns the values separately and does not change
``model``.

Keep rows aligned when splitting and batching
---------------------------------------------

For minibatches, the reconstructed covariate must follow the response rows.
Batch the observed values and a lookup index together with the response.
First choose the training rows, then assign each missing training value an
index into the parameter vector. Use ``-1`` for rows without a latent value.

.. plot::
   :context: close-figs
   :include-source: True
   :show-source-link: False
   :alt: Model graph showing x_observed, missing_index, and x_missing feeding the batch covariate x, which combines with alpha and beta in mu to determine the distribution of y.

   import liesel.optim as opt

   splitter = opt.Split(axis_size=n, validate_axis_size=24, seed=42)
   train_ids = np.asarray(splitter.indices_train)
   missing_train = np.sort(train_ids[np.isnan(x_data[train_ids])])
   lookup = np.full(n, -1, dtype=np.int32)
   lookup[missing_train] = np.arange(len(missing_train))
   split = splitter.split_position({
       "x_observed": jnp.asarray(np.where(np.isnan(x_data), 0.0, x_data)),
       "missing_index": jnp.asarray(lookup),
       "y": y_data,
   })

   x_missing_train = lsl.Var.new_param(
       jnp.zeros(len(missing_train)),
       lsl.Dist(tfd.Normal, 0.0, 1.0),
       name="x_missing",
   )
   x_observed = lsl.Var.new_obs(split.train["x_observed"], name="x_observed")
   missing_index = lsl.Var.new_obs(split.train["missing_index"], name="missing_index")
   x_train = lsl.Var.new_calc(
       lambda observed, index, z: jnp.where(index >= 0, z[jnp.maximum(index, 0)], observed),
       x_observed,
       missing_index,
       x_missing_train,
       name="x",
   )
   alpha = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="alpha")
   beta = lsl.Var.new_param(1.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="beta")
   mu = lsl.Var.new_calc(lambda a, b, x: a + b * x, alpha, beta, x_train, name="mu")
   y = lsl.Var.new_obs(split.train["y"], lsl.Dist(tfd.Normal, mu, 0.5), name="y")
   train_model = lsl.Model([y])
   train_model.plot(width=8, height=6)

The calculation receives only the current batch's observed values and lookup
indices, plus the missing-value parameter vector. It neither captures ``base``
nor reconstructs the full covariate for each batch. The parameter vector and
its optimizer state keep the same order throughout fitting. This example has
missing training values; if there are none, use ``x_observed`` directly.

Use Adam for the minibatch updates. The existing loss scales each batch's
response likelihood to the training size and adds the full parameter prior,
including the distribution of ``x_missing``.
Batching optimizes the same joint target, with the
:ref:`same statistical limitations <missing-values-joint-map>`.

.. code-block:: python

   import optax

   batches = opt.Batches.from_split(split, batch_size=16, shuffle=True)
   batch_result = opt.LieselOptim(
       train_model,
       split=split,
       batches=batches,
       optimizers=optax.adam(0.02),
       loss_monitor=opt.EmaTrainLossMonitor(effective_window=2.0),
       stopper=opt.Stopper(epochs=300, patience=50),
       save_position_history=False,
       seed=7,
   ).fit()
   batch_fit = batch_result.position_min_monitor
   completed_train = x_data.copy()
   completed_train[missing_train] = np.asarray(batch_fit["x_missing"])

``completed_train`` preserves observed values and leaves missing validation
covariates as ``NaN``. The :doc:`EMA monitor <optimizer-monitoring>` reuses batch
losses instead of evaluating the full training data after every epoch. Disabling
position history avoids saving the entire latent vector at every epoch.
The split data, latent parameters, and dense optimizer state still stay in memory.

See :doc:`optimizer-batching` for batch configuration and
:doc:`optimizer-loss-scaling` for scaling details.

Score held-out responses
------------------------

Integrate missing validation covariates over their covariate model, without
using the validation responses. Here the integral is available exactly:
for missing :math:`x_i`, the predictive response has mean :math:`\alpha` and
variance :math:`0.5^2 + \beta^2`.

.. note::

   The training model's default validation loss plugs in the placeholder value
   and uses only the response variance, omitting missing-covariate uncertainty.
   Use the scoring calculation below for this particular normal model. It
   integrates over missing covariates, conditional on the fitted coefficients.

.. code-block:: python

   valid_ids = np.asarray(splitter.indices_validate)
   valid_missing = np.isnan(x_data[valid_ids])
   x_mean = jnp.where(valid_missing, 0.0, split.validate["x_observed"])
   mean = batch_fit["alpha"] + batch_fit["beta"] * x_mean
   sd = jnp.sqrt(0.5**2 + valid_missing * batch_fit["beta"] ** 2)
   validation_nll = -tfd.Normal(mean, sd).log_prob(split.validate["y"]).mean()
   print(float(validation_nll))

The zero in ``x_mean`` uses the specified ``Normal(0, 1)`` covariate mean,
and the added :math:`\beta^2` in the predictive variance uses its variance of 1.
Neither depends on the placeholder used to store missing data.
Lower mean negative log predictive density indicates better predictions on the
same held-out data. This score does not include uncertainty in the coefficients.

Handle missing responses
------------------------

For independent regression responses with ignorable missingness, fit using the
observed response terms and predict the missing responses afterward. Omit only
the missing response term: retain the row's observed covariates in the covariate
likelihood if that model has estimated parameters. With MCMC, draw missing
responses from the response distribution for each posterior parameter draw.
After optimization, predictions are conditional on the fitted parameters.
Dependent responses, or missing responses that feed other observed variables,
may instead need explicit latent variables in the model.
