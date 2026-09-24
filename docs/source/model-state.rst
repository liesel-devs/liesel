Evaluate model states and derivatives
=====================================

A position is a dictionary of selected variable or node names and values.
A model state also contains the cached values and update flags of its
computational nodes. Use ``model`` from
:doc:`tutorials/notebooks/11-model-building` below.

Evaluate a position before applying it
--------------------------------------

.. code-block:: python

   import jax.numpy as jnp

   position = model.extract_position(["beta", "variance"])
   position["beta"] = jnp.array([0.5, 1.5])
   model.update()
   candidate = model.update_state(position, model.state)
   print(model.extract_position(["beta"], candidate))
   print(model.extract_position(["beta"]))

The first result contains the candidate coefficients; the second contains the
model's existing coefficients. ``update_state`` leaves the model unchanged by
default. Its input state must already be up to date: it only triggers updates
through the supplied position. Call ``update()`` first if automatic updates
have been disabled. Weak variables normally remain calculated, read-only values.

To apply the candidate explicitly:

.. code-block:: python

   model.state = candidate

This is also how you can apply a position returned by an optimizer. See
:doc:`optimizer-customization` for a fit-to-fit example and
:meth:`~liesel.model.Model.update_state` for the in-place option.

Evaluate a density and its gradient
-----------------------------------

.. code-block:: python

   import liesel.model as lsl

   log_prob = lsl.LogProb(model)
   position = model.extract_position(["beta", "variance"])
   print(float(log_prob(position)))
   print(log_prob.grad(position))

The gradient has the same dictionary keys and parameter shapes as ``position``.
It describes the local change in the unnormalized log density, not a fitted
parameter value. Use ``component="log_lik"`` or ``"log_prior"`` to inspect
one contribution, and :class:`~liesel.model.FlatLogProb` for an interface that
accepts a flat parameter array.

Write a calculator JAX can evaluate
-----------------------------------

Pass every changing dependency into the calculation:

.. code-block:: python

   def linear_mean(x, beta):
       return beta[0] + beta[1] * x

   predictor = lsl.Var.new_calc(
       linear_mean, model.vars["x"], model.vars["beta"], name="prediction_check"
   )
   print(predictor.value[:3])

Use JAX array operations in calculations that will be differentiated or compiled.
Avoid reading mutable globals or capturing ``beta.value`` in a closure: those
values would not be explicit graph inputs. A pure function depends only on its
inputs and has no side effects.

Python control flow over static values can work. Branching on a traced array
value needs JAX control flow such as ``jax.lax.cond`` or an elementwise
``jnp.where``. Keep data preparation outside the graph when it need not be
re-evaluated, and test any calculation that must also run on new prediction data.

By default, calculator variables cache their results and recompute when their
inputs change. ``new_calc(..., cache=False)`` avoids storing the result, but
repeats the calculation on access. Consider it for a cheap reshaping operation, not an
expensive matrix decomposition. See :meth:`~liesel.model.Var.new_calc` for details.
