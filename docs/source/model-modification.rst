Change an existing model
========================

Use ``model`` from :doc:`tutorials/notebooks/11-model-building` for these
examples. Make a copy when you want to compare model variants.

Change a value
--------------

.. code-block:: python

   import jax.numpy as jnp

   changed = model.copy()
   changed.vars["beta"].value = jnp.array([1.0, 2.0])
   print(changed.vars["mu"].value[:3])
   print(float(changed.log_lik))

The mean and likelihood update automatically. The connections stay the same.
Retrieve variables from ``changed`` when editing the copy: a Python variable
that points to the original ``beta`` still belongs to the original model.

Replace an input or a prior
---------------------------

.. code-block:: python

   import liesel.model as lsl
   import tensorflow_probability.substrates.jax.distributions as tfd

   new_width = lsl.Var.new_value(1.0, name="new_prior_scale")
   changed.vars["beta"].dist_node["scale"] = new_width
   print(changed.vars["beta"].dist_node["scale"].name)

The coefficient prior now depends on ``new_prior_scale``. To replace the whole
prior, assign a new distribution node:

.. code-block:: python

   changed.vars["beta"].dist_node = lsl.Dist(tfd.Normal, loc=0.0, scale=1.5)
   changed.rebuild_graph()
   print(list(changed.vars))

The old prior-scale variables disappear because they are no longer inputs of
the model's seed response ``y``. ``rebuild_graph()`` rediscovers the graph from
its seeds; components added with ``add_to_seeds=False`` can also disappear.
``update()`` instead recomputes outdated values without pruning the graph.

The same input indexing works for calculations. Here, ``sigma`` takes its
variance as its first positional input:

.. code-block:: python

   changed.vars["sigma"].value_node[0] = lsl.Var.new_value(0.5, name="fixed_variance")
   changed.rebuild_graph()
   print(float(changed.vars["sigma"].value))

The response standard deviation is now about 0.707. This changes one connection.
It also removes the unused variance parameter and its prior from this model.

Replace a variable throughout the model
---------------------------------------

Start from another copy when comparing this change with the previous one:

.. code-block:: python

   fixed = model.copy()
   fixed.replace("variance", lsl.Var.new_value(0.5, name="variance"))
   print(list(fixed.parameters))

Only ``beta`` remains a parameter. :meth:`~liesel.model.Model.replace` redirects
uses of the replaced variable throughout the model, whereas assigning one
``value_node`` or ``dist_node`` input changes just that connection. Both choices
change the statistical model here: variance is fixed and its prior is removed.
