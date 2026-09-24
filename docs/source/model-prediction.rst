Predict quantities at new data
==============================

:meth:`~liesel.model.Model.predict` evaluates named quantities at supplied
parameter draws. It propagates their uncertainty through the calculations in
the graph. For new random responses, use :doc:`model-simulation`.

Choose outputs and supply draws
-------------------------------

These examples use ``model`` and posterior ``samples`` from
:doc:`tutorials/notebooks/12-model-predictions`. Both the mean ``mu`` and
standard deviation ``sigma`` depend on ``x`` in that model.

.. code-block:: python

   import jax.numpy as jnp

   x_grid = jnp.linspace(-1.0, 1.0, 60)
   predicted = model.predict(
       samples, predict=["mu", "sigma"], newdata={"x": x_grid}
   )
   print({name: value.shape for name, value in predicted.items()})

Each array has shape ``(4, 500, 60)`` for the tutorial's four chains and
500 retained draws. Its last axis indexes grid points. The ``predict`` entries
are variable or node names, not distribution argument names such as ``loc``.
Values saved for weak variables in ``samples`` are ignored and recalculated.

Change the design deliberately
------------------------------

``newdata`` replaces named inputs for the evaluation. Omit it to use the current
data; omitted inputs keep their current values. Supply compatible arrays for
all covariates that change together. Keys must not overlap between ``samples``
and ``newdata``. Include draws for every parameter whose posterior uncertainty
you intend to propagate: missing strong inputs keep their current values.

Selecting only ``mu`` and ``sigma`` allows prediction to use their parental
submodel, so the old response length does not matter. Predicting response
log densities also requires response values matching the new covariates.

With default outputs or model-total nodes, the current implementation uses the
full model and applies the new-data state to it. Use ``model.copy().predict(...)``
when you need to preserve its state on those paths. Explicit output selections
as above operate on a copied parental submodel.

Summarize the requested quantity
--------------------------------

.. code-block:: python

   mean_interval = jnp.quantile(predicted["mu"], jnp.array([0.05, 0.95]), axis=(0, 1))
   print(mean_interval.shape)

The shape is ``(2, 60)``: lower and upper limits for each grid point. These are
pointwise 90% credible intervals for the conditional mean. They describe neither
new response variation nor simultaneous coverage of the entire curve.
Summarize ``predicted["sigma"]`` similarly to inspect uncertainty in the spread.

See :doc:`model-distributions` for pointwise log likelihoods. For large prediction
jobs, the ``chunk_size`` option bounds parallel intermediate calculations; it
does not reduce the memory needed for the returned arrays.
