Impute missing values
=====================

This guide demonstrates a technical pattern for missing-value models: represent
missing covariate values as parameters and combine them with observed values in
a calculated variable. MCMC samples the missing values alongside the regression
parameters; optimization integrates them out with :class:`~liesel.optim.LaplaceLoss`.
Run the examples in order; sampling and optimization use the same model.

.. admonition:: Scope of this example
   :class: note

   The example assumes missingness independent of the data and a fully specified
   ``Normal(0, 1)`` covariate model. It demonstrates implementation in Liesel.
   Choosing an imputation strategy or deciding whether your application's
   missingness mechanism can be ignored is outside this guide's scope.

Define a model with missing covariates
--------------------------------------

Generate a covariate, a response, and missing entries. Compute the missing
indices before fitting so their number stays fixed during JAX compilation.

.. plot::
   :context: reset
   :nofigs:
   :include-source: True
   :show-source-link: False

   import jax
   import jax.numpy as jnp
   import numpy as np
   import tensorflow_probability.substrates.jax.bijectors as tfb
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

Use ``x`` in the regression and estimate the response noise scale ``sigma``.
The exponential bijector gives it a log-scale parameter named ``h(sigma)``.
These examples use Liesel's default float32 precision.

.. plot::
   :context: close-figs
   :include-source: True
   :show-source-link: False
   :alt: Model graph showing x_missing feeding the completed covariate x, alpha and beta determining mu, and the positive scale sigma and mu determining the distribution of y.

   alpha = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="alpha")
   beta = lsl.Var.new_param(1.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="beta")
   sigma = lsl.Var.new_param(
       1.0, lsl.Dist(tfd.HalfCauchy, 0.0, 5.0), bijector=tfb.Exp(), name="sigma"
   )
   mu = lsl.Var.new_calc(lambda a, b, x: a + b * x, alpha, beta, x, name="mu")
   y = lsl.Var.new_obs(y_data, lsl.Dist(tfd.Normal, mu, sigma), name="y")
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

Integrate the missing values out
--------------------------------

Pass ``LaplaceLoss`` to :class:`~liesel.optim.LieselOptim` to integrate
``x_missing`` out and optimize the marginal posterior for ``alpha``, ``beta``,
and ``h(sigma)``. Priors and the transformation Jacobian remain in the objective,
so the maximum a posteriori (MAP) estimate is defined on this parameter scale.

.. code-block:: python

   import liesel.optim as opt

   loss = opt.LaplaceLoss(model, latent=["x_missing"])
   result = opt.LieselOptim(
       model,
       loss=loss,
       optimizers="lbfgs",
       loss_monitor="train_full_data",
       show_progress=False,
   ).fit()
   fitted = result.position_min_monitor
   sigma_fit = jnp.exp(fitted["h(sigma)"])
   print(
       f"alpha = {fitted['alpha']:.3f}, beta = {fitted['beta']:.3f}, "
       f"sigma = {sigma_fit:.3f}"
   )

.. code-block:: text

   alpha = 0.982, beta = 1.916, sigma = 0.488

The MCMC posterior means are about 0.983, 1.916, and 0.497, respectively.
Posterior means and a marginal mode summarize different aspects of the posterior.

.. admonition:: Limits of Laplace integration
   :class: note

   Integration is exact here because the missing values have a normal
   conditional distribution. With other responses, such as binary responses,
   it is an approximation. Its error need not vanish as the number of rows
   grows: each missing covariate is still informed by only its own row.
   Check against MCMC. Discrete missing covariates require another approach.

   Dense curvature limits this implementation to a few hundred missing values;
   see :doc:`optimizer-laplace`. For more, use MCMC. In this normal model, an
   analytically integrated likelihood can instead be written as ordinary per-row
   terms and minibatched exactly. ``LaplaceLoss`` itself requires full-data fits.

.. warning::

   Without ``LaplaceLoss``, the default loss optimizes missing values jointly
   with the other parameters. Here that gives a joint-MAP slope of about 1.943
   and noise scale of 0.441. In this normal model, joint MAP systematically
   inflates nonzero slope magnitudes and biases jointly estimated noise or
   covariate scales downward. The bias does not vanish as the sample grows;
   slope bias grows with noise and the share of missing values. Sampling the
   missing values with MCMC avoids this joint-MAP bias.

Recover imputations and uncertainty
-----------------------------------

The fitted loss state holds the missing values at their conditional modes,
matched to ``fitted``. Reconstruct the covariate at this fitted parameter set:

.. code-block:: python

   missing_mode = result.loss_state_min_monitor.latent_position["x_missing"]
   completed_mode = base.at[missing].set(missing_mode)

Like posterior means, these are single imputations that depend on the response.
Treating them as observed covariates can bias subsequent estimates involving
the response and understates uncertainty.

For imputations with uncertainty, construct a joint Gaussian approximation and
draw the parameters and missing values together. ``Model.predict`` reconstructs
the completed covariate and transforms the scale for each draw:

.. code-block:: python

   posterior = loss.approximate_joint_posterior(result)
   draws = posterior.sample(1000, seed=jax.random.key(42))
   predicted = model.predict(draws, predict=["x", "sigma"])
   completed_draws = predicted["x"]
   sigma_draws = predicted["sigma"]

``completed_draws`` has shape ``(1000, n)`` and preserves observed covariates
in every draw. Propagate the variation across draws in later analyses.
The joint posterior is approximate even though integrating the missing values
is exact in this example. See :doc:`optimizer-laplace` for diagnostics and
numerical controls. Fitting and prediction leave ``model`` unchanged.

Keep rows aligned when splitting
--------------------------------

Choose the training rows before assigning each missing training value an index
into the parameter vector. Split the observed values and lookup indices together
with the response, so the reconstructed covariate follows the shuffled rows.
Use ``-1`` for rows without a latent value.

.. plot::
   :context: close-figs
   :include-source: True
   :show-source-link: False
   :alt: Model graph showing x_observed, missing_index, and x_missing reconstructing the training covariate x, alpha and beta determining mu, and sigma controlling the response noise.

   import liesel.optim as opt

   splitter = opt.Split(axis_size=n, validate_axis_size=24, seed=42)
   train_ids = np.asarray(splitter.indices_train)
   missing_train = np.sort(train_ids[np.isnan(x_data[train_ids])])
   lookup = np.full(n, -1, dtype=np.int32)
   lookup[missing_train] = np.arange(len(missing_train))
   split = splitter.split_position(
       {
           "x_observed": jnp.asarray(np.where(np.isnan(x_data), 0.0, x_data)),
           "missing_index": jnp.asarray(lookup),
           "y": y_data,
       }
   )

   x_missing_train = lsl.Var.new_param(
       jnp.zeros(len(missing_train)),
       lsl.Dist(tfd.Normal, 0.0, 1.0),
       name="x_missing",
   )
   x_observed = lsl.Var.new_obs(split.train["x_observed"], name="x_observed")
   missing_index = lsl.Var.new_obs(split.train["missing_index"], name="missing_index")
   x_train = lsl.Var.new_calc(
       lambda observed, index, z: jnp.where(
           index >= 0, z[jnp.maximum(index, 0)], observed
       ),
       x_observed,
       missing_index,
       x_missing_train,
       name="x",
   )
   alpha = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="alpha")
   beta = lsl.Var.new_param(1.0, lsl.Dist(tfd.Normal, 0.0, 5.0), name="beta")
   sigma = lsl.Var.new_param(
       1.0, lsl.Dist(tfd.HalfCauchy, 0.0, 5.0), bijector=tfb.Exp(), name="sigma"
   )
   mu = lsl.Var.new_calc(lambda a, b, x: a + b * x, alpha, beta, x_train, name="mu")
   y = lsl.Var.new_obs(split.train["y"], lsl.Dist(tfd.Normal, mu, sigma), name="y")
   train_model = lsl.Model([y])
   train_model.plot(width=8, height=6)

The calculation receives the selected rows' observed values and lookup indices
explicitly. The latent vector contains only missing training values and keeps
a fixed order. If no training values are missing, use ``x_observed`` directly
and the default optimization loss.

Integrate over the missing training values while fitting the full training split:

.. code-block:: python

   train_loss = opt.LaplaceLoss(train_model, split=split, latent=["x_missing"])
   train_result = opt.LieselOptim(
       train_model,
       split=split,
       loss=train_loss,
       optimizers="lbfgs",
       loss_monitor="train_full_data",
       show_progress=False,
   ).fit()
   train_fit = train_result.position_min_monitor
   train_sigma = jnp.exp(train_fit["h(sigma)"])
   train_missing_mode = train_result.loss_state_min_monitor.latent_position[
       "x_missing"
   ]
   completed_train = x_data.copy()
   completed_train[missing_train] = np.asarray(train_missing_mode)

``completed_train`` preserves observed values and leaves missing validation
covariates as ``NaN``. The validation responses do not enter this fit.

Score held-out responses
------------------------

Integrate missing validation covariates over their covariate model, without
using the validation responses. Here the integral is available exactly:
for missing :math:`x_i`, the predictive response has mean :math:`\alpha` and
variance :math:`\sigma^2 + \beta^2`.

.. note::

   ``LaplaceLoss`` requires ``loss_monitor="train_full_data"`` and does not
   evaluate the validation loss. Score the held-out responses separately.
   The calculation below integrates over missing covariates, conditional on
   the fitted regression coefficients and noise scale.

.. code-block:: python

   valid_ids = np.asarray(splitter.indices_validate)
   valid_missing = np.isnan(x_data[valid_ids])
   x_mean = jnp.where(valid_missing, 0.0, split.validate["x_observed"])
   mean = train_fit["alpha"] + train_fit["beta"] * x_mean
   sd = jnp.sqrt(train_sigma**2 + valid_missing * train_fit["beta"] ** 2)
   validation_nll = -tfd.Normal(mean, sd).log_prob(split.validate["y"]).mean()
   print(f"{validation_nll:.3f}")

.. code-block:: text

   0.929

The zero in ``x_mean`` uses the specified ``Normal(0, 1)`` covariate mean,
and the added :math:`\beta^2` in the predictive variance uses its variance of 1.
Neither depends on the placeholder used to store missing data.
Lower mean negative log predictive density indicates better predictions on the
same held-out data. This score does not include uncertainty in the fitted
coefficients or noise scale.

Handle missing responses
------------------------

For independent regression responses with ignorable missingness, fit using the
observed response terms and predict the missing responses afterward. Omit only
the missing response term: retain the row's observed covariates in the covariate
likelihood if that model has estimated parameters. With MCMC, draw missing
responses from the response distribution for each posterior parameter draw.
After a Laplace fit, use approximate joint-posterior draws in the same way.
Predictions from just the fitted parameters omit parameter uncertainty.
Dependent responses, or missing responses that feed other observed variables,
may instead need explicit latent variables in the model.
