Transform constrained parameters
================================

A positive parameter can be represented by an unconstrained source while the
likelihood still receives its positive value. Use a bijection when you want to
preserve a prior already defined on the positive scale.

Keep the positive value in the model
------------------------------------

Starting with ``model`` from :doc:`tutorials/notebooks/11-model-building`, copy
it and transform its variance:

.. code-block:: python

   import tensorflow_probability.substrates.jax.bijectors as tfb

   transformed = model.copy()
   transformed.plot(width=8, height=5)
   variance = transformed.vars["variance"]
   variance.biject(tfb.Exp(), name="log_variance")
   log_variance = variance.bijected_var
   print(log_variance.name, float(log_variance.value), float(variance.value))
   transformed.plot(width=8, height=5)

The new strong parameter is ``log_variance``. The original ``variance`` is now
weak and calculates ``exp(log_variance)``. The exponential bijector's forward
direction goes from the unconstrained value to the positive value; its inverse
computes the initial log variance. The graph gains that value dependency.

``biject`` returns the original variable, which is useful for chaining. Read
``.bijected_var`` to obtain the new parameter. :meth:`~liesel.model.Var.transform`
returns the transformed variable instead. ``biject("auto")`` selects the
distribution's default event-space bijector; an explicit bijector makes your
choice visible.

Preserve the intended density
-----------------------------

The prior moves to the unconstrained source and includes the change-of-variables
Jacobian. If the original prior is on variance :math:`v` and :math:`u=\log(v)`,
the transformed log density is :math:`\log p(\exp(u)) + u`.
The observed/parameter flags move to the source as well.

By comparison, ``new_calc(jnp.exp, log_variance)`` only creates a calculation.
A normal prior placed directly on ``log_variance`` defines a log-normal prior
on variance; it does not automatically transform an arbitrary prior on variance.
The second tutorial uses this direct modeling choice for a log-scale predictor.

Density modes depend on the parameterization. Even with the correct Jacobian,
optimizing the transformed density need not give the transform of the original
density's mode.

Choose which parameters to transform
------------------------------------

For an existing distribution, :meth:`~liesel.model.Dist.biject_parameters` can
transform its strong parameter inputs. For example, this selects the scale
input of a normal likelihood:

.. code-block:: python

   import liesel.model as lsl
   import tensorflow_probability.substrates.jax.distributions as tfd

   scale = lsl.Var.new_param(
       1.0, lsl.Dist(tfd.LogNormal, loc=0.0, scale=0.5), name="scale"
   )
   likelihood = lsl.Dist(tfd.Normal, loc=0.0, scale=scale)
   likelihood.biject_parameters({"scale": tfb.Exp()})
   print(scale.bijected_var.name)

These keys name distribution arguments, not model variables. For precise control
over the source name and its inference configuration, use ``Var.biject``.

Carry inference settings deliberately
-------------------------------------

If a variable already has inference settings, supply settings for the new source
or explicitly drop the old ones with ``inference="drop"``. For an untransformed
variance prepared for NUTS, the corresponding call is:

.. code-block:: python

   import liesel.goose as gs

   prepared = model.copy()
   prepared.vars["variance"].biject(
       tfb.Exp(), name="log_variance", inference=gs.MCMCSpec(gs.NUTSKernel)
   )

Sample or optimize ``log_variance`` and interpret the back-transformed
``variance``. Configure the other parameters too before running inference.
See :doc:`tutorials/md/01c-transform` for a complete MCMC example and
:doc:`optimizer-customization` for parameter selection in optimization.
